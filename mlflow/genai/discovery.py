from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

import pydantic

import mlflow
from mlflow.entities.assessment import Feedback
from mlflow.entities.span_status import SpanStatusCode
from mlflow.entities.trace import Trace
from mlflow.genai.evaluation.entities import EvaluationResult
from mlflow.genai.judges.make_judge import make_judge
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
_MIN_CONFIDENCE = 75
_MIN_EXAMPLES = 2

_DEFAULT_JUDGE_MODEL = "openai:/gpt-5-mini"
_DEFAULT_ANALYSIS_MODEL = "openai:/gpt-5"
_DEFAULT_SCORER_NAME = "_issue_discovery_judge"

_TEMPLATE_VARS = {
    "{{ trace }}",
    "{{ inputs }}",
    "{{ outputs }}",
    "{{ conversation }}",
    "{{ expectations }}",
}


# ---- Pydantic schemas for LLM structured output ----


class _TraceAnalysis(pydantic.BaseModel):
    trace_index: int = pydantic.Field(description="Index of the trace in the input list")
    failure_category: str = pydantic.Field(
        description=(
            "Category of failure: tool_error, hallucination, latency, "
            "incomplete_response, error_propagation, wrong_tool_use, "
            "context_loss, or other"
        )
    )
    failure_summary: str = pydantic.Field(description="Brief summary of what went wrong")
    root_cause_hypothesis: str = pydantic.Field(
        description="Hypothesis about why this failure occurred based on span-level evidence"
    )
    affected_spans: list[str] = pydantic.Field(
        description="Names of spans most relevant to the failure"
    )
    severity: int = pydantic.Field(
        description="Severity of the failure (1=minor, 3=moderate, 5=critical)"
    )


class _BatchTraceAnalysisResult(pydantic.BaseModel):
    analyses: list[_TraceAnalysis] = pydantic.Field(description="Analysis for each failing trace")


class _ScorerInstructions(pydantic.BaseModel):
    detection_instructions: str = pydantic.Field(
        description=(
            "Instructions for a judge to detect this issue from a {{ trace }}. "
            "MUST contain the literal text '{{ trace }}'."
        )
    )


class _IdentifiedIssue(pydantic.BaseModel):
    name: str = pydantic.Field(description="snake_case identifier for the issue")
    description: str = pydantic.Field(description="What the issue is")
    root_cause: str = pydantic.Field(description="Why this issue occurs")
    example_indices: list[int] = pydantic.Field(
        description="Indices into the input trace summary list that exemplify this issue"
    )
    confidence: int = pydantic.Field(
        description=(
            "Confidence that this is a real, distinct issue (0-100). "
            "0=not confident, 25=might be real, 50=moderately confident, "
            "75=highly confident and real, 100=absolutely certain"
        ),
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
    confidence: int
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
        name=_DEFAULT_SCORER_NAME,
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


def _ensure_template_var(instructions: str) -> str:
    """Ensure instructions contain at least one template variable, defaulting to {{ trace }}."""
    if any(var in instructions for var in _TEMPLATE_VARS):
        return instructions
    return f"Analyze the following {{{{ trace }}}} and determine:\n\n{instructions}"


def _get_existing_score(trace: Trace, scorer_name: str) -> bool | None:
    """Return True/False if the trace already has a boolean score with the given name."""
    for assessment in trace.info.assessments:
        if isinstance(assessment, Feedback) and assessment.name == scorer_name:
            if isinstance(assessment.value, bool):
                return assessment.value
    return None


def _partition_by_existing_scores(
    traces: list[Trace],
    scorer_name: str,
) -> tuple[list[Trace], list[Trace], list[Trace]]:
    """Split traces into (negative, positive, needs_scoring) based on existing scores."""
    negative: list[Trace] = []
    positive: list[Trace] = []
    needs_scoring: list[Trace] = []
    for trace in traces:
        score = _get_existing_score(trace, scorer_name)
        if score is True:
            positive.append(trace)
        elif score is False:
            negative.append(trace)
        else:
            needs_scoring.append(trace)
    return negative, positive, needs_scoring


def _test_scorer(scorer: Scorer, trace: Trace) -> None:
    """Run scorer on a single trace to verify it works. Raises on failure."""
    result = mlflow.genai.evaluate(data=[trace], scorers=[scorer])

    if result.result_df is None:
        return

    value_col = f"{scorer.name}/value"
    if value_col not in result.result_df.columns:
        return

    if result.result_df[value_col].iloc[0] is not None:
        return

    # Scorer returned None — check the assessment on the trace for the actual error
    error_msg = None
    if "trace" in result.result_df.columns:
        result_trace = result.result_df["trace"].iloc[0]
        if isinstance(result_trace, str):
            result_trace = Trace.from_json(result_trace)
        if result_trace is not None:
            for assessment in result_trace.info.assessments:
                if isinstance(assessment, Feedback) and assessment.name == scorer.name:
                    error_msg = assessment.error_message
                    break

    raise mlflow.exceptions.MlflowException(
        f"Scorer '{scorer.name}' failed on test trace "
        f"{trace.info.trace_id}: {error_msg or 'unknown error (check model API logs)'}"
    )


def _build_span_tree(spans: list[object]) -> str:
    """Build an indented span tree string from a flat list of spans."""
    if not spans:
        return "    (no spans)"

    # Index spans by span_id for parent lookup
    span_by_id: dict[str, object] = {}
    children: dict[str | None, list[object]] = {}
    for span in spans:
        span_by_id[span.span_id] = span
        children.setdefault(span.parent_id, []).append(span)

    def _format_span(span, depth: int) -> list[str]:
        indent = "    " + "  " * depth
        duration_ms = ""
        if span.end_time_ns is not None and span.start_time_ns is not None:
            duration_ms = f", {(span.end_time_ns - span.start_time_ns) // 1_000_000}ms"

        status = span.status.status_code.value if span.status else "UNKNOWN"
        span_type = getattr(span, "span_type", "UNKNOWN") or "UNKNOWN"
        model = getattr(span, "model_name", None)
        model_str = f", model={model}" if model else ""

        line = f"{indent}{span.name} ({span_type}, {status}{duration_ms}{model_str})"
        lines = [line]

        # Add error details
        if span.status and span.status.status_code == SpanStatusCode.ERROR:
            if span.status.description:
                lines.append(f"{indent}  ERROR: {span.status.description[:200]}")
            for event in getattr(span, "events", []):
                if event.name == "exception":
                    exc_type = event.attributes.get("exception.type", "")
                    exc_msg = event.attributes.get("exception.message", "")
                    if exc_type or exc_msg:
                        lines.append(f"{indent}  EXCEPTION: {exc_type}: {exc_msg}"[:250])

        # Add span I/O (truncated)
        inputs = getattr(span, "inputs", None)
        outputs = getattr(span, "outputs", None)
        if inputs:
            lines.append(f"{indent}  in: {str(inputs)[:200]}")
        if outputs:
            lines.append(f"{indent}  out: {str(outputs)[:200]}")

        for child in children.get(span.span_id, []):
            lines.extend(_format_span(child, depth + 1))
        return lines

    # Start from root spans (parent_id is None)
    roots = children.get(None, [])
    # Fallback: if no roots found, treat all spans as flat
    if not roots:
        roots = spans

    result_lines: list[str] = []
    for root in roots:
        result_lines.extend(_format_span(root, 0))
    return "\n".join(result_lines)


def _build_enriched_trace_summary(index: int, trace: Trace, rationale: str) -> str:
    request = (trace.info.request_preview or "")[:500]
    response = (trace.info.response_preview or "")[:500]
    duration = trace.info.execution_duration or 0
    span_tree = _build_span_tree(trace.data.spans)
    return (
        f"[{index}] trace_id={trace.info.trace_id}\n"
        f"  Input: {request}\n"
        f"  Output: {response}\n"
        f"  Duration: {duration}ms | Failure rationale: {rationale}\n"
        f"  Span tree:\n{span_tree}"
    )


def _run_deep_analysis(
    enriched_summaries: list[str],
    analysis_model: str,
) -> list[_TraceAnalysis]:
    result = get_chat_completions_with_structured_output(
        model_uri=analysis_model,
        messages=[
            ChatMessage(
                role="system",
                content=(
                    "You are an expert at diagnosing AI application failures. "
                    "Given enriched trace summaries with span-level detail, analyze each "
                    "failing trace individually.\n\n"
                    "For each trace, identify:\n"
                    "- The failure category (tool_error, hallucination, latency, "
                    "incomplete_response, error_propagation, wrong_tool_use, "
                    "context_loss, or other)\n"
                    "- A brief failure summary\n"
                    "- A root cause hypothesis based on the span evidence\n"
                    "- Which spans are most relevant to the failure\n"
                    "- Severity (1=minor, 3=moderate, 5=critical)"
                ),
            ),
            ChatMessage(
                role="user",
                content=(
                    "Analyze each of the following failing traces:\n\n"
                    + "\n\n".join(enriched_summaries)
                ),
            ),
        ],
        output_schema=_BatchTraceAnalysisResult,
    )
    return result.analyses


def _format_analysis_for_clustering(
    index: int,
    analysis: _TraceAnalysis,
    enriched_summary: str,
) -> str:
    return (
        f"[{index}] Category: {analysis.failure_category} | "
        f"Severity: {analysis.severity}/5\n"
        f"  Summary: {analysis.failure_summary}\n"
        f"  Root cause: {analysis.root_cause_hypothesis}\n"
        f"  Affected spans: {', '.join(analysis.affected_spans)}\n"
        f"  ---\n"
        f"  {enriched_summary}"
    )


def _generate_scorer_instructions(
    issue: _IdentifiedIssue,
    example_analyses: list[_TraceAnalysis],
    judge_model: str,
) -> str:
    examples_text = "\n".join(
        f"- [{a.failure_category}] {a.failure_summary} (affected: {', '.join(a.affected_spans)})"
        for a in example_analyses
    )
    result = get_chat_completions_with_structured_output(
        model_uri=judge_model,
        messages=[
            ChatMessage(
                role="system",
                content=(
                    "You are an expert at writing detection instructions for AI quality judges. "
                    "Given an issue description and example failures, write concise instructions "
                    "that a judge can use to detect this issue in a trace.\n\n"
                    "CRITICAL: Your detection_instructions MUST contain the literal text "
                    "'{{ trace }}' (with double curly braces) as a template variable — "
                    "this is how the judge receives the trace data.\n"
                    "Example: 'Analyze the {{ trace }} to determine if...'"
                ),
            ),
            ChatMessage(
                role="user",
                content=(
                    f"Issue: {issue.name}\n"
                    f"Description: {issue.description}\n"
                    f"Root cause: {issue.root_cause}\n\n"
                    f"Example failures:\n{examples_text}\n\n"
                    "Write detection instructions for this issue."
                ),
            ),
        ],
        output_schema=_ScorerInstructions,
    )
    return _ensure_template_var(result.detection_instructions)


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
        val = row.get(value_col)
        if val is None or bool(val):
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

        affected = df[value_col].eq(True).sum()
        frequencies[name] = affected / total if total > 0 else 0.0

        examples = []
        if rationale_col in df.columns:
            for _, row in df.iterrows():
                val = row.get(value_col)
                if val is not None and bool(val) and len(examples) < 3:
                    if r := row.get(rationale_col):
                        examples.append(str(r))
        rationale_examples[name] = examples

    return frequencies, rationale_examples


def _log_artifacts(result: DiscoverIssuesResult, experiment_id: str) -> str | None:
    """Log discovery results as MLflow artifacts under a timestamped run. Returns run_id."""
    run_name = f"discover_issues_{int(time.time())}"
    try:
        with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
            mlflow.log_text(result.summary, "summary.md")
            issues_data = [
                {
                    "name": issue.name,
                    "description": issue.description,
                    "root_cause": issue.root_cause,
                    "frequency": issue.frequency,
                    "confidence": issue.confidence,
                    "example_trace_ids": issue.example_trace_ids,
                    "rationale_examples": issue.rationale_examples,
                    "scorer_name": issue.scorer.name,
                }
                for issue in result.issues
            ]
            mlflow.log_text(json.dumps(issues_data, indent=2), "issues.json")
            mlflow.log_text(
                json.dumps(
                    {
                        "total_traces_analyzed": result.total_traces_analyzed,
                        "num_issues": len(result.issues),
                        "timestamp": int(time.time()),
                        "triage_run_id": result.triage_evaluation.run_id,
                        "validation_run_id": (
                            result.validation_evaluation.run_id
                            if result.validation_evaluation
                            else None
                        ),
                    },
                    indent=2,
                ),
                "metadata.json",
            )
            if result.triage_evaluation.result_df is not None:
                mlflow.log_text(
                    result.triage_evaluation.result_df.to_csv(index=False),
                    "triage_results.csv",
                )
            if (
                result.validation_evaluation is not None
                and result.validation_evaluation.result_df is not None
            ):
                mlflow.log_text(
                    result.validation_evaluation.result_df.to_csv(index=False),
                    "validation_results.csv",
                )
            return run.info.run_id
    except Exception:
        _logger.warning("Failed to log discovery artifacts", exc_info=True)
        return None


def _build_summary(issues: list[Issue], total_traces: int) -> str:
    if not issues:
        return f"## Issue Discovery Summary\n\nAnalyzed {total_traces} traces. No issues found."

    lines = [
        "## Issue Discovery Summary\n",
        f"Analyzed **{total_traces}** traces. Found **{len(issues)}** issues:\n",
    ]
    for i, issue in enumerate(issues, 1):
        lines.append(
            f"### {i}. {issue.name} ({issue.frequency:.0%} of traces, "
            f"confidence: {issue.confidence}/100)\n\n"
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
    judge_model: str | None = None,
    analysis_model: str | None = None,
    sample_size: int = _DEFAULT_SAMPLE_SIZE,
    validation_sample_size: int | None = None,
    max_issues: int = 10,
    filter_string: str | None = None,
) -> DiscoverIssuesResult:
    """Discover quality and operational issues in an experiment's traces.

    Runs a multi-phase pipeline:
    1. **Triage**: Scores a sample of traces for user satisfaction
    2. **Deep Analysis**: Extracts enriched span data and analyzes each failure
    3. **Cluster**: Groups analyses into distinct issue categories
    4. **Generate Scorers**: Writes detection instructions per issue
    5. **Validate**: Runs generated issue scorers on a broader trace set

    Args:
        experiment_id: Experiment to analyze. Defaults to the active experiment.
        model_id: Scope traces to a specific model.
        satisfaction_scorer: Custom scorer for triage. Defaults to a built-in
            conversation-level satisfaction judge.
        judge_model: LLM used for scoring traces (satisfaction + issue detection).
            Defaults to ``"openai:/gpt-5-mini"``.
        analysis_model: LLM used for clustering failures into issues.
            Defaults to ``"openai:/gpt-5"``.
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
    judge_model = judge_model or _DEFAULT_JUDGE_MODEL
    analysis_model = analysis_model or _DEFAULT_ANALYSIS_MODEL
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
        satisfaction_scorer = _build_default_satisfaction_scorer(judge_model)

    # Phase 1: Triage — score a sample for user satisfaction
    _logger.info("Phase 1: Fetching %d traces...", sample_size)
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

    # Check for existing scores from a prior discover_issues run
    scorer_name = satisfaction_scorer.name
    already_negative, _already_positive, needs_scoring = _partition_by_existing_scores(
        triage_traces, scorer_name
    )
    if already_negative:
        _logger.info(
            "Found %d traces with existing '%s' = False, %d need scoring",
            len(already_negative),
            scorer_name,
            len(needs_scoring),
        )

    # Test the scorer on one trace before running the full batch
    _logger.info("Phase 1: Testing scorer on one trace...")
    _test_scorer(satisfaction_scorer, triage_traces[0])

    # Score only traces without existing scores
    if needs_scoring:
        _logger.info("Phase 1: Scoring %d traces...", len(needs_scoring))
        triage_eval = mlflow.genai.evaluate(
            data=needs_scoring,
            scorers=[satisfaction_scorer],
            model_id=model_id,
        )
        scored_failing, rationale_map = _extract_failing_traces(
            triage_eval, satisfaction_scorer.name
        )
    else:
        triage_eval = EvaluationResult(run_id="", metrics={}, result_df=None)
        scored_failing = []
        rationale_map = {}

    # Combine: previously-negative traces + scorer-failing traces
    failing_traces = already_negative + scored_failing
    for trace in already_negative:
        rationale_map.setdefault(trace.info.trace_id, "Previously scored as unsatisfactory")

    _logger.info(
        "Phase 1 complete: %d/%d traces unsatisfactory (%d existing, %d newly scored)",
        len(failing_traces),
        len(triage_traces),
        len(already_negative),
        len(scored_failing),
    )

    if not failing_traces:
        return DiscoverIssuesResult(
            issues=[],
            triage_evaluation=triage_eval,
            validation_evaluation=None,
            summary=_build_summary([], len(triage_traces)),
            total_traces_analyzed=len(triage_traces),
        )

    # Phase 2: Deep Analysis — enriched span extraction + batched LLM analysis
    capped_failing = failing_traces[:_MAX_SUMMARIES_FOR_CLUSTERING]
    _logger.info("Phase 2: Deep analysis of %d failing traces...", len(capped_failing))
    enriched_summaries = [
        _build_enriched_trace_summary(i, t, rationale_map.get(t.info.trace_id, ""))
        for i, t in enumerate(capped_failing)
    ]
    analyses = _run_deep_analysis(enriched_summaries, analysis_model)
    _logger.info("Phase 2 complete: %d trace analyses produced", len(analyses))

    # Phase 3: Cluster — group analyses into distinct issues
    _logger.info("Phase 3: Clustering analyses into issues...")
    # Build lookup from trace_index -> analysis
    analysis_by_index = {a.trace_index: a for a in analyses}
    clustering_texts = "\n\n".join(
        _format_analysis_for_clustering(
            i,
            analysis_by_index.get(i, analyses[i] if i < len(analyses) else analyses[0]),
            enriched_summaries[i],
        )
        for i in range(len(enriched_summaries))
    )

    clustering_result = get_chat_completions_with_structured_output(
        model_uri=analysis_model,
        messages=[
            ChatMessage(
                role="system",
                content=(
                    "You are an expert at analyzing AI application failures. "
                    "Given per-trace analyses with failure categories and root causes, "
                    f"group them into at most {max_issues} distinct issue categories.\n\n"
                    "For each issue provide:\n"
                    "- A snake_case name\n"
                    "- A clear description\n"
                    "- The root cause\n"
                    "- Indices of example traces from the input\n"
                    "- A confidence score 0-100 indicating how confident you are this "
                    "is a real, distinct issue (0=not confident, 50=moderate, "
                    "75=highly confident, 100=certain). Be rigorous — only score "
                    "75+ if multiple traces clearly demonstrate the same pattern."
                ),
            ),
            ChatMessage(
                role="user",
                content=f"Per-trace analyses:\n\n{clustering_texts}",
            ),
        ],
        output_schema=_IssueClusteringResult,
    )

    identified = [
        issue
        for issue in clustering_result.issues[:max_issues]
        if issue.confidence >= _MIN_CONFIDENCE and len(issue.example_indices) >= _MIN_EXAMPLES
    ]
    _logger.info(
        "Phase 3 complete: %d issues identified (%d filtered out by confidence/examples)",
        len(identified),
        len(clustering_result.issues) - len(identified),
    )

    if not identified:
        return DiscoverIssuesResult(
            issues=[],
            triage_evaluation=triage_eval,
            validation_evaluation=None,
            summary=_build_summary([], len(triage_traces)),
            total_traces_analyzed=len(triage_traces),
        )

    # Phase 4: Generate Scorers — write detection instructions per issue
    _logger.info("Phase 4: Generating scorers for %d issues...", len(identified))
    issue_scorers: list[Scorer] = []
    for issue in identified:
        example_analyses = [
            analysis_by_index[idx] for idx in issue.example_indices if idx in analysis_by_index
        ]
        instructions = _generate_scorer_instructions(issue, example_analyses, judge_model)
        scorer = make_judge(
            name=issue.name,
            instructions=instructions,
            model=judge_model,
            feedback_value_type=bool,
        )
        issue_scorers.append(scorer)
    _logger.info("Phase 4 complete: %d scorers generated", len(issue_scorers))

    # Phase 5: Validate — run issue scorers on a broader sample
    if validation_sample_size is None:
        validation_sample_size = 5 * sample_size

    _logger.info(
        "Phase 5: Validating %d issues on %d traces...", len(identified), validation_sample_size
    )
    validation_traces = mlflow.search_traces(max_results=validation_sample_size, **search_kwargs)

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
                confidence=ident.confidence,
                rationale_examples=rationale_examples_map.get(ident.name, []),
            )
        )

    issues.sort(key=lambda i: i.frequency, reverse=True)
    total_analyzed = len(validation_traces)
    summary = _build_summary(issues, total_analyzed)
    _logger.info("Done. Found %d issues across %d traces.", len(issues), total_analyzed)

    result = DiscoverIssuesResult(
        issues=issues,
        triage_evaluation=triage_eval,
        validation_evaluation=validation_eval,
        summary=summary,
        total_traces_analyzed=total_analyzed,
    )

    # Log artifacts so each run's results are preserved
    _log_artifacts(result, exp_id)

    return result
