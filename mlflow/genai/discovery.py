from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import pandas as pd
import pydantic

import mlflow
from mlflow.entities.span_status import SpanStatusCode
from mlflow.entities.trace import Trace
from mlflow.genai.evaluation.entities import EvaluationResult
from mlflow.genai.judges.make_judge import make_judge
from mlflow.genai.judges.utils import get_default_model
from mlflow.genai.judges.utils.invocation_utils import (
    get_chat_completions_with_structured_output,
)
from mlflow.genai.scorers.base import Scorer
from mlflow.tracking.fluent import _get_experiment_id
from mlflow.types.llm import ChatMessage
from mlflow.utils.annotations import experimental

_logger = logging.getLogger(__name__)

_DEFAULT_SAMPLE_SIZE = 100
_MAX_SUMMARIES_FOR_CLUSTERING = 50
_MIN_FREQUENCY_THRESHOLD = 0.01


# ---- Pydantic schemas for LLM structured output ----


class _IdentifiedIssue(pydantic.BaseModel):
    name: str = pydantic.Field(description="snake_case identifier for the issue")
    description: str = pydantic.Field(description="What the issue is")
    root_cause: str = pydantic.Field(description="Why this issue occurs")
    detection_instructions: str = pydantic.Field(
        description="Instructions for a judge to detect this issue from a {{ trace }}"
    )
    example_indices: list[int] = pydantic.Field(
        description="Indices into the input trace summary list that exemplify this issue"
    )


class _IssueClusteringResult(pydantic.BaseModel):
    issues: list[_IdentifiedIssue] = pydantic.Field(
        description="Distinct issues identified from the failing traces"
    )


class _SummaryResult(pydantic.BaseModel):
    summary: str = pydantic.Field(description="Markdown-formatted summary of the discovered issues")


# ---- Data models ----


@dataclass
class Issue:
    """A distinct issue discovered in the experiment's traces."""

    name: str
    description: str
    root_cause: str
    example_trace_ids: list[str]
    scorer: Scorer
    frequency: float
    rationale_examples: list[str] = field(default_factory=list)


@dataclass
class DiscoverIssuesResult:
    """Result of the discover_issues pipeline."""

    issues: list[Issue]
    triage_evaluation: EvaluationResult
    validation_evaluation: EvaluationResult | None
    summary: str
    total_traces_analyzed: int


# ---- Internal helpers ----


def _build_default_satisfaction_scorer(model: str | None) -> Scorer:
    return make_judge(
        name="satisfaction",
        instructions=(
            "Evaluate whether the user's goals in this {{ conversation }} were achieved "
            "efficiently and completely. Consider:\n"
            "- Did the assistant understand the user's request?\n"
            "- Was the response accurate and complete?\n"
            "- Were there unnecessary steps, errors, or confusion?\n"
            "- Did the conversation reach a satisfactory resolution?\n\n"
            "Return True if the user's goals were met satisfactorily, False otherwise."
        ),
        model=model,
        feedback_value_type=bool,
    )


@dataclass
class _TraceSummary:
    index: int
    trace_id: str
    inputs_preview: str
    outputs_preview: str
    span_names: list[str]
    error_spans: list[str]
    satisfaction_rationale: str
    execution_duration_ms: int | None


def _build_trace_summary(
    index: int,
    trace: Trace,
    satisfaction_rationale: str,
) -> _TraceSummary:
    request_preview = trace.info.request_preview or ""
    response_preview = trace.info.response_preview or ""

    span_names = [span.name for span in trace.data.spans]
    error_spans = [
        span.name for span in trace.data.spans if span.status.status_code == SpanStatusCode.ERROR
    ]

    return _TraceSummary(
        index=index,
        trace_id=trace.info.trace_id,
        inputs_preview=request_preview[:500],
        outputs_preview=response_preview[:500],
        span_names=span_names,
        error_spans=error_spans,
        satisfaction_rationale=satisfaction_rationale,
        execution_duration_ms=trace.info.execution_duration,
    )


def _format_summaries_for_clustering(summaries: list[_TraceSummary]) -> str:
    parts = [
        f"[{s.index}] trace_id={s.trace_id}\n"
        f"  Input: {s.inputs_preview}\n"
        f"  Output: {s.outputs_preview}\n"
        f"  Spans: {', '.join(s.span_names)}\n"
        f"  Error spans: {', '.join(s.error_spans) if s.error_spans else 'none'}\n"
        f"  Duration: {s.execution_duration_ms}ms\n"
        f"  Satisfaction rationale: {s.satisfaction_rationale}"
        for s in summaries
    ]
    return "\n\n".join(parts)


# ---- Pipeline phases ----


def _check_scorer_errors(
    df: pd.DataFrame,
    scorer_name: str,
) -> None:
    """Raise if a scorer produced all null values, indicating a systemic error."""
    value_col = f"{scorer_name}/value"
    if value_col not in df.columns:
        return

    if df[value_col].isna().all() or df[value_col].apply(lambda v: v is None).all():
        error_msg = _extract_scorer_error(df, scorer_name)
        raise mlflow.exceptions.MlflowException(
            f"Scorer '{scorer_name}' failed on all traces. "
            f"This usually means the scorer's model is misconfigured or "
            f"a required dependency is missing.\n"
            f"Error: {error_msg}"
        )


def _extract_scorer_error(df: pd.DataFrame, scorer_name: str) -> str:
    """Try to extract an error message from the assessments column."""
    if "assessments" not in df.columns:
        return "No error details available."

    for assessments_raw in df["assessments"]:
        if assessments_raw is None:
            continue
        try:
            assessments = (
                json.loads(assessments_raw) if isinstance(assessments_raw, str) else assessments_raw
            )
            if not isinstance(assessments, list):
                continue
            for assessment in assessments:
                if not isinstance(assessment, dict):
                    continue
                if assessment.get("assessment_name") == scorer_name:
                    feedback = assessment.get("feedback", {})
                    error = feedback.get("error", {})
                    if error_message := error.get("error_message"):
                        return error_message
        except (json.JSONDecodeError, TypeError, AttributeError):
            continue

    return "No error details available."


def _phase1_triage(
    traces: list[Trace],
    satisfaction_scorer: Scorer,
    model_id: str | None,
) -> tuple[EvaluationResult, list[Trace], dict[str, str]]:
    """Run satisfaction scorer on sampled traces.

    Returns (eval_result, failing_traces, rationale_map).
    """
    eval_result = mlflow.genai.evaluate(
        data=traces,
        scorers=[satisfaction_scorer],
        model_id=model_id,
    )

    scorer_name = satisfaction_scorer.name

    failing_traces = []
    rationale_map: dict[str, str] = {}

    if eval_result.result_df is not None:
        df = eval_result.result_df

        _check_scorer_errors(df, scorer_name)

        value_col = f"{scorer_name}/value"
        rationale_col = f"{scorer_name}/rationale"

        if value_col in df.columns:
            for _, row in df.iterrows():
                val = row.get(value_col)
                if val is False or val in {"no", "No"}:
                    trace = row.get("trace")
                    if trace is not None:
                        if isinstance(trace, str):
                            trace = Trace.from_json(trace)
                        failing_traces.append(trace)
                        rationale = row.get(rationale_col, "")
                        rationale_map[trace.info.trace_id] = str(rationale) if rationale else ""

    return eval_result, failing_traces, rationale_map


def _phase2_cluster(
    failing_traces: list[Trace],
    rationale_map: dict[str, str],
    model: str,
    max_issues: int,
) -> list[_IdentifiedIssue]:
    """Cluster failing traces into distinct issues using an LLM."""
    if not failing_traces:
        return []

    summaries = []
    for i, trace in enumerate(failing_traces[:_MAX_SUMMARIES_FOR_CLUSTERING]):
        rationale = rationale_map.get(trace.info.trace_id, "")
        summaries.append(_build_trace_summary(i, trace, rationale))

    summaries_text = _format_summaries_for_clustering(summaries)

    messages = [
        ChatMessage(
            role="system",
            content=(
                "You are an expert at analyzing AI application failures. "
                "Given a list of failing trace summaries from an AI application, "
                "identify the distinct issues that are causing failures. "
                "Group similar failures together into coherent issues. "
                f"Identify at most {max_issues} distinct issues. "
                "For each issue, provide:\n"
                "- A snake_case name\n"
                "- A clear description of what the issue is\n"
                "- The root cause of why it occurs\n"
                "- Detection instructions that a judge could use to identify this issue "
                "from a single trace (the judge will receive the full trace via {{ trace }})\n"
                "- Indices of example traces from the input list"
            ),
        ),
        ChatMessage(
            role="user",
            content=f"Here are the failing trace summaries:\n\n{summaries_text}",
        ),
    ]

    result = get_chat_completions_with_structured_output(
        model_uri=model,
        messages=messages,
        output_schema=_IssueClusteringResult,
    )

    return result.issues[:max_issues]


def _phase3_validate(
    validation_traces: list[Trace],
    identified_issues: list[_IdentifiedIssue],
    model: str,
    model_id: str | None,
) -> tuple[EvaluationResult | None, dict[str, float], dict[str, list[str]]]:
    """Run issue scorers on validation traces.

    Returns (eval_result, frequency_map, rationale_map).
    """
    if not identified_issues or not validation_traces:
        return None, {}, {}

    issue_scorers = [
        make_judge(
            name=issue.name,
            instructions=issue.detection_instructions,
            model=model,
            feedback_value_type=bool,
        )
        for issue in identified_issues
    ]

    eval_result = mlflow.genai.evaluate(
        data=validation_traces,
        scorers=issue_scorers,
        model_id=model_id,
    )

    frequency_map: dict[str, float] = {}
    rationale_examples: dict[str, list[str]] = {}

    if eval_result.result_df is not None:
        df = eval_result.result_df
        total = len(df)

        for issue in identified_issues:
            _check_scorer_errors(df, issue.name)

            value_col = f"{issue.name}/value"
            rationale_col = f"{issue.name}/rationale"

            if value_col in df.columns:
                affected = df[value_col].apply(lambda v: v is True or v in {"yes", "Yes"}).sum()
                frequency_map[issue.name] = affected / total if total > 0 else 0.0

                examples = []
                if rationale_col in df.columns:
                    for _, row in df.iterrows():
                        val = row.get(value_col)
                        if (val is True or val in {"yes", "Yes"}) and len(examples) < 3:
                            if r := row.get(rationale_col, ""):
                                examples.append(str(r))
                rationale_examples[issue.name] = examples

    return eval_result, frequency_map, rationale_examples


def _phase4_summarize(
    issues: list[Issue],
    total_traces: int,
    model: str,
) -> str:
    """Generate a markdown summary of the discovered issues."""
    if not issues:
        return (
            f"## Issue Discovery Summary\n\n"
            f"Analyzed {total_traces} traces. No significant issues were identified."
        )

    issues_text = "\n".join(
        f"- **{issue.name}** (frequency: {issue.frequency:.1%}): {issue.description}\n"
        f"  Root cause: {issue.root_cause}"
        for issue in issues
    )

    messages = [
        ChatMessage(
            role="system",
            content=(
                "You are an expert at summarizing AI application quality analysis. "
                "Generate a concise markdown summary of the discovered issues. "
                "Include an overview, issues ranked by frequency, root causes, "
                "and actionable recommendations."
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                f"Total traces analyzed: {total_traces}\n\n"
                f"Discovered issues:\n{issues_text}\n\n"
                "Generate a markdown summary with sections for Overview, "
                "Issues (ranked by frequency), and Recommendations."
            ),
        ),
    ]

    result = get_chat_completions_with_structured_output(
        model_uri=model,
        messages=messages,
        output_schema=_SummaryResult,
    )
    return result.summary


# ---- Main API ----


@experimental(version="3.1.0")
def discover_issues(
    experiment_id: str | None = None,
    model_id: str | None = None,
    satisfaction_scorer: Scorer | None = None,
    model: str | None = None,
    sample_size: int = _DEFAULT_SAMPLE_SIZE,
    validation_sample_size: int | None = None,
    max_issues: int = 10,
    filter_string: str | None = None,
) -> DiscoverIssuesResult:
    """Automatically discover quality and operational issues in an experiment's traces.

    Runs a multi-phase pipeline:
    1. **Triage**: Scores a sample of traces for user satisfaction
    2. **Cluster**: Groups failing traces into distinct issues using an LLM
    3. **Validate**: Runs generated issue scorers on a broader trace set
    4. **Summarize**: Produces a markdown report of findings

    Args:
        experiment_id: Experiment to analyze. Defaults to the active experiment.
        model_id: Scope traces to a specific model.
        satisfaction_scorer: Custom Scorer for triage. Defaults to a built-in
            conversation-level satisfaction judge.
        model: LLM for judge and analysis calls. Defaults to ``get_default_model()``.
        sample_size: Number of traces for the triage phase.
        validation_sample_size: Number of traces for validation.
            Defaults to ``min(5 * sample_size, total_traces)``.
        max_issues: Maximum distinct issues to identify.
        filter_string: Filter string passed to ``search_traces``.

    Returns:
        A :class:`DiscoverIssuesResult` containing the discovered issues,
        evaluation results, and a summary report.

    Example:

        .. code-block:: python

            import mlflow

            mlflow.set_experiment("my-genai-app")
            result = mlflow.genai.discover_issues()  # clint: disable=unknown-mlflow-function

            for issue in result.issues:
                print(f"{issue.name}: {issue.frequency:.1%} of traces affected")

            print(result.summary)
    """
    if model is None:
        model = get_default_model()

    exp_id = experiment_id or _get_experiment_id()
    if exp_id is None:
        raise mlflow.exceptions.MlflowException(
            "No experiment specified and no active experiment found. "
            "Please set an active experiment with mlflow.set_experiment() "
            "or pass experiment_id explicitly."
        )

    locations = [exp_id]

    if satisfaction_scorer is None:
        satisfaction_scorer = _build_default_satisfaction_scorer(model)

    # Fetch triage sample
    _logger.info("Phase 1: Fetching %d traces for triage...", sample_size)
    triage_traces = mlflow.search_traces(
        filter_string=filter_string,
        max_results=sample_size,
        model_id=model_id,
        return_type="list",
        locations=locations,
    )

    if not triage_traces:
        _logger.warning("No traces found in experiment %s", exp_id)
        empty_eval = EvaluationResult(run_id="", metrics={}, result_df=None)
        return DiscoverIssuesResult(
            issues=[],
            triage_evaluation=empty_eval,
            validation_evaluation=None,
            summary=f"## Issue Discovery Summary\n\nNo traces found in experiment {exp_id}.",
            total_traces_analyzed=0,
        )

    # Phase 1: Triage
    _logger.info("Phase 1: Running satisfaction scorer on %d traces...", len(triage_traces))
    triage_eval, failing_traces, rationale_map = _phase1_triage(
        triage_traces, satisfaction_scorer, model_id
    )

    _logger.info(
        "Phase 1 complete: %d/%d traces flagged as unsatisfactory",
        len(failing_traces),
        len(triage_traces),
    )

    if not failing_traces:
        summary = (
            f"## Issue Discovery Summary\n\n"
            f"Analyzed {len(triage_traces)} traces. "
            f"All traces passed the satisfaction check — no issues found."
        )
        return DiscoverIssuesResult(
            issues=[],
            triage_evaluation=triage_eval,
            validation_evaluation=None,
            summary=summary,
            total_traces_analyzed=len(triage_traces),
        )

    # Phase 2: Cluster
    _logger.info("Phase 2: Clustering %d failing traces into issues...", len(failing_traces))
    identified_issues = _phase2_cluster(failing_traces, rationale_map, model, max_issues)
    _logger.info("Phase 2 complete: %d distinct issues identified", len(identified_issues))

    if not identified_issues:
        summary = (
            f"## Issue Discovery Summary\n\n"
            f"Analyzed {len(triage_traces)} traces. "
            f"{len(failing_traces)} traces failed satisfaction check, "
            f"but no distinct issue patterns could be identified."
        )
        return DiscoverIssuesResult(
            issues=[],
            triage_evaluation=triage_eval,
            validation_evaluation=None,
            summary=summary,
            total_traces_analyzed=len(triage_traces),
        )

    # Phase 3: Validate
    total_traces = _get_total_trace_count(locations, model_id, filter_string)
    if validation_sample_size is None:
        validation_sample_size = min(5 * sample_size, total_traces)

    _logger.info("Phase 3: Fetching %d traces for validation...", validation_sample_size)
    validation_traces = mlflow.search_traces(
        filter_string=filter_string,
        max_results=validation_sample_size,
        model_id=model_id,
        return_type="list",
        locations=locations,
    )

    _logger.info(
        "Phase 3: Running %d issue scorers on %d traces...",
        len(identified_issues),
        len(validation_traces),
    )
    validation_eval, frequency_map, rationale_examples_map = _phase3_validate(
        validation_traces, identified_issues, model, model_id
    )

    # Build Issue objects
    issues: list[Issue] = []
    for ident_issue in identified_issues:
        frequency = frequency_map.get(ident_issue.name, 0.0)
        if frequency < _MIN_FREQUENCY_THRESHOLD:
            _logger.debug(
                "Discarding issue '%s' with frequency %.3f (below threshold %.3f)",
                ident_issue.name,
                frequency,
                _MIN_FREQUENCY_THRESHOLD,
            )
            continue

        example_trace_ids = [
            failing_traces[idx].info.trace_id
            for idx in ident_issue.example_indices
            if 0 <= idx < len(failing_traces)
        ]

        issue_scorer = make_judge(
            name=ident_issue.name,
            instructions=ident_issue.detection_instructions,
            model=model,
            feedback_value_type=bool,
        )

        issues.append(
            Issue(
                name=ident_issue.name,
                description=ident_issue.description,
                root_cause=ident_issue.root_cause,
                example_trace_ids=example_trace_ids,
                scorer=issue_scorer,
                frequency=frequency,
                rationale_examples=rationale_examples_map.get(ident_issue.name, []),
            )
        )

    issues.sort(key=lambda i: i.frequency, reverse=True)

    # Phase 4: Summarize
    total_analyzed = len(validation_traces) if validation_traces else len(triage_traces)
    _logger.info("Phase 4: Generating summary report...")
    summary = _phase4_summarize(issues, total_analyzed, model)

    return DiscoverIssuesResult(
        issues=issues,
        triage_evaluation=triage_eval,
        validation_evaluation=validation_eval,
        summary=summary,
        total_traces_analyzed=total_analyzed,
    )


def _get_total_trace_count(
    locations: list[str],
    model_id: str | None,
    filter_string: str | None,
) -> int:
    """Get total trace count in the experiment by fetching minimal data."""
    traces = mlflow.search_traces(
        filter_string=filter_string,
        model_id=model_id,
        return_type="list",
        locations=locations,
        include_spans=False,
    )
    return len(traces)
