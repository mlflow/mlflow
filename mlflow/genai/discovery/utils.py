from __future__ import annotations

import logging

import mlflow
from mlflow.entities.assessment import Feedback
from mlflow.entities.span_status import SpanStatusCode
from mlflow.entities.trace import Trace
from mlflow.genai.discovery.constants import (
    _DEFAULT_SCORER_NAME,
    _SESSION_SATISFACTION_INSTRUCTIONS,
    _TEMPLATE_VARS,
    _TRACE_SATISFACTION_INSTRUCTIONS,
)
from mlflow.genai.discovery.schemas import (
    _BatchTraceAnalysisResult,
    _IdentifiedIssue,
    _ScorerInstructionsResult,
    _ScorerSpec,
    _TraceAnalysis,
)
from mlflow.genai.evaluation.entities import EvaluationResult
from mlflow.genai.judges.make_judge import make_judge
from mlflow.genai.judges.utils.invocation_utils import (
    get_chat_completions_with_structured_output,
)
from mlflow.genai.scorers.base import Scorer
from mlflow.types.llm import ChatMessage

_logger = logging.getLogger(__name__)


def _has_session_ids(traces: list[Trace]) -> bool:
    for t in traces:
        if (t.info.tags or {}).get("mlflow.trace.session_id"):
            return True
        if (t.info.trace_metadata or {}).get("mlflow.trace.session"):
            return True
    return False


def _build_default_satisfaction_scorer(model: str | None, use_conversation: bool) -> Scorer:
    instructions = (
        _SESSION_SATISFACTION_INSTRUCTIONS if use_conversation else _TRACE_SATISFACTION_INSTRUCTIONS
    )
    return make_judge(
        name=_DEFAULT_SCORER_NAME,
        instructions=instructions,
        model=model,
        feedback_value_type=bool,
    )


def _ensure_template_var(instructions: str) -> str:
    if any(var in instructions for var in _TEMPLATE_VARS):
        return instructions
    return f"Analyze the following {{{{ trace }}}} and determine:\n\n{instructions}"


def _get_existing_score(trace: Trace, scorer_name: str) -> bool | None:
    for assessment in trace.info.assessments:
        if isinstance(assessment, Feedback) and assessment.name == scorer_name:
            if isinstance(assessment.value, bool):
                return assessment.value
    return None


def _partition_by_existing_scores(
    traces: list[Trace],
    scorer_name: str,
) -> tuple[list[Trace], list[Trace], list[Trace]]:
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
    result = mlflow.genai.evaluate(data=[trace], scorers=[scorer])

    if result.run_id:
        try:
            mlflow.MlflowClient().delete_run(result.run_id)
        except Exception:
            _logger.debug("Failed to delete test run %s", result.run_id)

    if result.result_df is None:
        return

    value_col = f"{scorer.name}/value"
    if value_col not in result.result_df.columns:
        return

    if result.result_df[value_col].iloc[0] is not None:
        return

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
    if not spans:
        return "    (no spans)"

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

        if span.status and span.status.status_code == SpanStatusCode.ERROR:
            if span.status.description:
                lines.append(f"{indent}  ERROR: {span.status.description[:200]}")
            for event in getattr(span, "events", []):
                if event.name == "exception":
                    exc_type = event.attributes.get("exception.type", "")
                    exc_msg = event.attributes.get("exception.message", "")
                    if exc_type or exc_msg:
                        lines.append(f"{indent}  EXCEPTION: {exc_type}: {exc_msg}"[:250])

        inputs = getattr(span, "inputs", None)
        outputs = getattr(span, "outputs", None)
        if inputs:
            lines.append(f"{indent}  in: {str(inputs)[:200]}")
        if outputs:
            lines.append(f"{indent}  out: {str(outputs)[:200]}")

        for child in children.get(span.span_id, []):
            lines.extend(_format_span(child, depth + 1))
        return lines

    roots = children.get(None, [])
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


def _generate_scorer_specs(
    issue: _IdentifiedIssue,
    example_analyses: list[_TraceAnalysis],
    judge_model: str,
) -> list[_ScorerSpec]:
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
                    "IMPORTANT: The judge returns yes/no (pass/fail). A passing trace (yes) "
                    "means the trace is FREE of this issue. A failing trace (no) means "
                    "the issue WAS detected. Write instructions so that 'yes' = clean/good "
                    "and 'no' = issue found.\n\n"
                    "CRITICAL RULE ON SPLITTING SCORERS:\n"
                    "Each scorer MUST test exactly ONE criterion. If the issue involves "
                    "multiple independent criteria, you MUST split them into separate scorers. "
                    "Indicators that you need to split:\n"
                    "- The word 'and' joining two distinct checks "
                    "(e.g. 'is slow AND hallucinates')\n"
                    "- The word 'or' joining two distinct checks "
                    "(e.g. 'truncates OR omits data')\n"
                    "- Multiple failure modes that could occur independently\n"
                    "For example, 'response is truncated and uses wrong API' should become TWO "
                    "scorers: one for truncation and one for wrong API usage.\n\n"
                    "CRITICAL: Each scorer's detection_instructions MUST contain the literal text "
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
                    "Write detection scorer(s) for this issue. Use one scorer if the issue is "
                    "a single criterion, or multiple scorers if it involves independent criteria."
                ),
            ),
        ],
        output_schema=_ScorerInstructionsResult,
    )
    for spec in result.scorers:
        spec.detection_instructions = _ensure_template_var(spec.detection_instructions)
    return result.scorers


def _extract_failing_traces(
    eval_result: EvaluationResult,
    scorer_name: str,
) -> tuple[list[Trace], dict[str, str]]:
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

        affected = df[value_col].eq(False).sum()
        frequencies[name] = affected / total if total > 0 else 0.0

        examples = []
        if rationale_col in df.columns:
            for _, row in df.iterrows():
                val = row.get(value_col)
                if val is not None and not bool(val) and len(examples) < 3:
                    if r := row.get(rationale_col):
                        examples.append(str(r))
        rationale_examples[name] = examples

    return frequencies, rationale_examples


def _log_discovery_artifacts(run_id: str, artifacts: dict[str, str]) -> None:
    if not run_id:
        return
    client = mlflow.MlflowClient()
    for filename, content in artifacts.items():
        try:
            client.log_text(run_id, content, filename)
        except Exception:
            _logger.warning("Failed to log %s to run %s", filename, run_id, exc_info=True)


def _build_summary(issues: list[object], total_traces: int) -> str:
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
