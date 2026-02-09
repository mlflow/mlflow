from __future__ import annotations

import logging
from dataclasses import dataclass, field

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

_DEFAULT_SAMPLE_SIZE = 50
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


def _format_trace_for_clustering(index: int, trace: Trace, rationale: str) -> str:
    request = (trace.info.request_preview or "")[:500]
    response = (trace.info.response_preview or "")[:500]
    spans = ", ".join(s.name for s in trace.data.spans)
    errors = ", ".join(
        s.name for s in trace.data.spans if s.status.status_code == SpanStatusCode.ERROR
    )
    return (
        f"[{index}] trace_id={trace.info.trace_id}\n"
        f"  Input: {request}\n"
        f"  Output: {response}\n"
        f"  Spans: {spans}\n"
        f"  Error spans: {errors or 'none'}\n"
        f"  Duration: {trace.info.execution_duration}ms\n"
        f"  Failure rationale: {rationale}"
    )


def _extract_failing_traces(
    eval_result: EvaluationResult,
    scorer_name: str,
) -> tuple[list[Trace], dict[str, str]]:
    """Extract traces that failed a boolean scorer and their rationales."""
    failing = []
    rationales: dict[str, str] = {}
    df = eval_result.result_df
    if df is None:
        return failing, rationales

    value_col = f"{scorer_name}/value"
    rationale_col = f"{scorer_name}/rationale"
    if value_col not in df.columns:
        return failing, rationales

    for _, row in df.iterrows():
        if row.get(value_col) is not False:
            continue
        trace = row.get("trace")
        if trace is None:
            continue
        if isinstance(trace, str):
            trace = Trace.from_json(trace)
        failing.append(trace)
        rationales[trace.info.trace_id] = str(row.get(rationale_col, "") or "")

    return failing, rationales


def _compute_frequencies(
    eval_result: EvaluationResult,
    scorer_names: list[str],
) -> tuple[dict[str, float], dict[str, list[str]]]:
    """Compute frequency and sample rationales for each boolean scorer."""
    frequencies: dict[str, float] = {}
    rationale_examples: dict[str, list[str]] = {}
    df = eval_result.result_df
    if df is None:
        return frequencies, rationale_examples

    total = len(df)
    for name in scorer_names:
        value_col = f"{name}/value"
        rationale_col = f"{name}/rationale"
        if value_col not in df.columns:
            continue

        affected = df[value_col].apply(lambda v: v is True).sum()
        frequencies[name] = affected / total if total > 0 else 0.0

        examples = []
        if rationale_col in df.columns:
            for _, row in df.iterrows():
                if row.get(value_col) is True and len(examples) < 3:
                    if r := row.get(rationale_col):
                        examples.append(str(r))
        rationale_examples[name] = examples

    return frequencies, rationale_examples


def _build_summary(issues: list[Issue], total_traces: int) -> str:
    if not issues:
        return f"## Issue Discovery Summary\n\nAnalyzed {total_traces} traces. No issues found."

    lines = [
        "## Issue Discovery Summary\n",
        f"Analyzed **{total_traces}** traces. Found **{len(issues)}** issues:\n",
    ]
    for i, issue in enumerate(issues, 1):
        lines.append(
            f"### {i}. {issue.name} ({issue.frequency:.0%} of traces)\n\n"
            f"{issue.description}\n\n"
            f"**Root cause:** {issue.root_cause}\n"
        )
    return "\n".join(lines)


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
    """Discover quality and operational issues in an experiment's traces.

    Runs a multi-phase pipeline:
    1. **Triage**: Scores a sample of traces for user satisfaction
    2. **Cluster**: Groups failing traces into distinct issues via LLM
    3. **Validate**: Runs generated issue scorers on a broader trace set

    Args:
        experiment_id: Experiment to analyze. Defaults to the active experiment.
        model_id: Scope traces to a specific model.
        satisfaction_scorer: Custom scorer for triage. Defaults to a built-in
            conversation-level satisfaction judge.
        model: LLM for judge and analysis calls (e.g. ``"openai:/gpt-4.1"``).
        sample_size: Number of traces for the triage phase.
        validation_sample_size: Number of traces for validation.
            Defaults to ``5 * sample_size``.
        max_issues: Maximum distinct issues to identify.
        filter_string: Filter string passed to ``search_traces``.

    Returns:
        A :class:`DiscoverIssuesResult` with discovered issues, evaluation
        results, and a summary report.

    Example:

        .. code-block:: python

            import mlflow

            mlflow.set_experiment("my-genai-app")
            result = mlflow.genai.discover_issues()

            for issue in result.issues:
                print(f"{issue.name}: {issue.frequency:.1%} of traces affected")
                # Each issue has a scorer you can reuse
                mlflow.genai.evaluate(data=traces, scorers=[issue.scorer])
    """
    model = model or get_default_model()
    exp_id = experiment_id or _get_experiment_id()
    if exp_id is None:
        raise mlflow.exceptions.MlflowException(
            "No experiment specified. Use mlflow.set_experiment() or pass experiment_id."
        )

    locations = [exp_id]
    search_kwargs = {
        "filter_string": filter_string,
        "model_id": model_id,
        "return_type": "list",
        "locations": locations,
    }

    if satisfaction_scorer is None:
        satisfaction_scorer = _build_default_satisfaction_scorer(model)

    # Phase 1: Triage — score a sample for user satisfaction
    _logger.info("Phase 1: Scoring %d traces for satisfaction...", sample_size)
    triage_traces = mlflow.search_traces(max_results=sample_size, **search_kwargs)
    if not triage_traces:
        empty_eval = EvaluationResult(run_id="", metrics={}, result_df=None)
        return DiscoverIssuesResult(
            issues=[],
            triage_evaluation=empty_eval,
            validation_evaluation=None,
            summary=f"No traces found in experiment {exp_id}.",
            total_traces_analyzed=0,
        )

    triage_eval = mlflow.genai.evaluate(
        data=triage_traces,
        scorers=[satisfaction_scorer],
        model_id=model_id,
    )
    failing_traces, rationale_map = _extract_failing_traces(triage_eval, satisfaction_scorer.name)
    _logger.info(
        "Phase 1 complete: %d/%d traces unsatisfactory", len(failing_traces), len(triage_traces)
    )

    if not failing_traces:
        return DiscoverIssuesResult(
            issues=[],
            triage_evaluation=triage_eval,
            validation_evaluation=None,
            summary=_build_summary([], len(triage_traces)),
            total_traces_analyzed=len(triage_traces),
        )

    # Phase 2: Cluster — identify distinct issues from failing traces
    _logger.info("Phase 2: Clustering %d failing traces into issues...", len(failing_traces))
    summaries_text = "\n\n".join(
        _format_trace_for_clustering(i, t, rationale_map.get(t.info.trace_id, ""))
        for i, t in enumerate(failing_traces[:_MAX_SUMMARIES_FOR_CLUSTERING])
    )

    clustering_result = get_chat_completions_with_structured_output(
        model_uri=model,
        messages=[
            ChatMessage(
                role="system",
                content=(
                    "You are an expert at analyzing AI application failures. "
                    "Given failing trace summaries, identify distinct issue categories. "
                    f"Identify at most {max_issues} issues.\n\n"
                    "For each issue provide:\n"
                    "- A snake_case name\n"
                    "- A clear description\n"
                    "- The root cause\n"
                    "- Detection instructions for a judge that receives a {{ trace }} "
                    "and returns True if the issue is present\n"
                    "- Indices of example traces from the input"
                ),
            ),
            ChatMessage(
                role="user",
                content=f"Failing trace summaries:\n\n{summaries_text}",
            ),
        ],
        output_schema=_IssueClusteringResult,
    )

    identified = clustering_result.issues[:max_issues]
    _logger.info("Phase 2 complete: %d issues identified", len(identified))

    if not identified:
        return DiscoverIssuesResult(
            issues=[],
            triage_evaluation=triage_eval,
            validation_evaluation=None,
            summary=_build_summary([], len(triage_traces)),
            total_traces_analyzed=len(triage_traces),
        )

    # Phase 3: Validate — run issue scorers on a broader sample
    if validation_sample_size is None:
        validation_sample_size = 5 * sample_size

    _logger.info(
        "Phase 3: Validating %d issues on %d traces...", len(identified), validation_sample_size
    )
    validation_traces = mlflow.search_traces(max_results=validation_sample_size, **search_kwargs)

    issue_scorers = [
        make_judge(
            name=issue.name,
            instructions=issue.detection_instructions,
            model=model,
            feedback_value_type=bool,
        )
        for issue in identified
    ]

    validation_eval = mlflow.genai.evaluate(
        data=validation_traces,
        scorers=issue_scorers,
        model_id=model_id,
    )

    frequencies, rationale_examples_map = _compute_frequencies(
        validation_eval, [i.name for i in identified]
    )

    # Build final Issue objects, reusing the scorers from validation
    issues: list[Issue] = []
    scorer_by_name = {s.name: s for s in issue_scorers}
    for ident in identified:
        freq = frequencies.get(ident.name, 0.0)
        if freq < _MIN_FREQUENCY_THRESHOLD:
            continue

        example_ids = [
            failing_traces[idx].info.trace_id
            for idx in ident.example_indices
            if 0 <= idx < len(failing_traces)
        ]
        issues.append(
            Issue(
                name=ident.name,
                description=ident.description,
                root_cause=ident.root_cause,
                example_trace_ids=example_ids,
                scorer=scorer_by_name[ident.name],
                frequency=freq,
                rationale_examples=rationale_examples_map.get(ident.name, []),
            )
        )

    issues.sort(key=lambda i: i.frequency, reverse=True)
    total_analyzed = len(validation_traces)
    summary = _build_summary(issues, total_analyzed)
    _logger.info("Done. Found %d issues across %d traces.", len(issues), total_analyzed)

    return DiscoverIssuesResult(
        issues=issues,
        triage_evaluation=triage_eval,
        validation_evaluation=validation_eval,
        summary=summary,
        total_traces_analyzed=total_analyzed,
    )
