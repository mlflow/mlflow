from __future__ import annotations

import logging
from collections import defaultdict

import mlflow
from mlflow.entities.assessment import Feedback
from mlflow.entities.trace import Trace
from mlflow.genai.discovery.constants import (
    TRACE_CONTENT_TRUNCATION,
)
from mlflow.genai.discovery.entities import Issue, _ConversationAnalysis, _IdentifiedIssue
from mlflow.genai.scorers.base import Scorer
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.tracing.provider import trace_disabled

_logger = logging.getLogger(__name__)


def get_session_id(trace: Trace) -> str | None:
    return (trace.info.trace_metadata or {}).get(TraceMetadataKey.TRACE_SESSION)


def group_traces_by_session(
    traces: list[Trace],
) -> dict[str, list[Trace]]:
    """
    Group traces by session ID.

    Traces without a session become standalone single-trace "sessions"
    keyed by their trace_id. Each group is sorted by timestamp_ms.

    Note: mlflow.genai.evaluation.session_utils has a similar function, but it
    operates on EvalItem objects and drops traces without sessions. This version
    works on raw Trace objects and keeps sessionless traces as standalone groups,
    which is required for the discovery pipeline's frequency calculations.
    """
    groups: dict[str, list[Trace]] = defaultdict(list)
    for trace in traces:
        session_id = get_session_id(trace) or trace.info.trace_id
        groups[session_id].append(trace)

    for traces_in_group in groups.values():
        traces_in_group.sort(key=lambda trace: trace.info.timestamp_ms)

    return dict(groups)


def _compute_percentiles(values: list[float], percentiles: list[int | float]) -> list[float]:
    """
    Compute percentiles using NumPy if available, otherwise linear interpolation.

    This implementation matches NumPy's default linear interpolation method.

    Args:
        values: List of numeric values to compute percentiles from.
        percentiles: List of percentile values to compute (e.g., [50, 75, 90, 95, 99]).

    Returns:
        List of computed percentile values in the same order as requested.
    """
    if not values:
        return []

    try:
        import numpy as np

        return [float(p) for p in np.percentile(values, percentiles)]
    except ImportError:
        # Fall back to linear interpolation (matches NumPy's default method)
        values_sorted = sorted(values)
        n = len(values_sorted)
        result = []

        for p in percentiles:
            # NumPy default: linear interpolation between data points
            # Formula: value = lower + fraction * (upper - lower)
            # where fraction comes from (n - 1) * p / 100
            idx_float = (n - 1) * p / 100
            idx_lower = int(idx_float)
            idx_upper = min(idx_lower + 1, n - 1)

            fraction = idx_float - idx_lower
            lower_val = values_sorted[idx_lower]
            upper_val = values_sorted[idx_upper]
            result.append(lower_val + fraction * (upper_val - lower_val))

        return result


def compute_latency_percentiles(traces: list[Trace]) -> dict[str, float | int] | None:
    """
    Compute latency percentiles from a list of traces for relative threshold context.

    Args:
        traces: List of traces to analyze.

    Returns:
        Dictionary with percentile values (p50, p75, p90, p95, p99) in seconds
        and count as an integer, or None if no valid durations found.
    """
    durations_ms = [
        trace.info.execution_duration
        for trace in traces
        if trace.info.execution_duration is not None
    ]

    if not durations_ms:
        return None

    durations_s = [d / 1000 for d in durations_ms]
    percentile_values = _compute_percentiles(durations_s, [50, 75, 90, 95, 99])

    return {
        "p50": round(percentile_values[0], 2),
        "p75": round(percentile_values[1], 2),
        "p90": round(percentile_values[2], 2),
        "p95": round(percentile_values[3], 2),
        "p99": round(percentile_values[4], 2),
        "count": len(durations_s),
    }


def build_summary(issues: list[Issue], total_traces: int) -> str:
    if not issues:
        return f"Analyzed {total_traces} traces. No issues found."

    lines = [
        f"Analyzed **{total_traces}** traces. Found **{len(issues)}** issues:\n",
    ]
    for i, issue in enumerate(issues, 1):
        root_causes = "; ".join(issue.root_causes) if issue.root_causes else "Unknown"
        categories = ", ".join(issue.categories) if issue.categories else "None"
        lines.append(
            f"### {i}. {issue.name} (severity: {issue.severity})\n\n"
            f"{issue.description}\n\n"
            f"**Root causes:** {root_causes}\n\n"
            f"**Categories:** {categories}\n"
        )
    return "\n".join(lines)


def log_discovery_artifacts(run_id: str, artifacts: dict[str, str]) -> None:
    if not run_id:
        return
    client = mlflow.MlflowClient()
    for filename, content in artifacts.items():
        try:
            client.log_text(run_id, content, filename)
        except Exception:
            _logger.warning("Failed to log %s to run %s", filename, run_id, exc_info=True)


@trace_disabled
def verify_scorer(
    scorer: Scorer,
    trace: Trace,
    session: list[Trace] | None = None,
) -> None:
    """
    Verify a scorer works on a single trace (or session) before running the full pipeline.

    Calls the scorer and checks that the returned Feedback has a non-null value.

    Args:
        scorer: The scorer to test.
        trace: A trace to run the scorer on (used for trace-based scorers).
        session: If provided, pass as ``session=`` to the scorer instead of ``trace=``.
            Used for conversation-based scorers that require ``{{ conversation }}``.

    Raises:
        MlflowException: If the scorer produces no feedback or returns a null value.
    """
    try:
        feedback = scorer(session=session) if session is not None else scorer(trace=trace)
    except Exception as exc:
        raise mlflow.exceptions.MlflowException(
            f"Scorer '{scorer.name}' failed verification on trace {trace.info.trace_id}: {exc}"
        ) from exc

    if not isinstance(feedback, Feedback):
        raise mlflow.exceptions.MlflowException(
            f"Scorer '{scorer.name}' returned {type(feedback).__name__} instead of Feedback"
        )
    if feedback.value is None:
        error = feedback.error_message or "unknown error (check model API logs)"
        raise mlflow.exceptions.MlflowException(
            f"Scorer '{scorer.name}' returned null value: {error}"
        )


def collect_affected_trace_ids(
    issue: _IdentifiedIssue,
    analyses: list[_ConversationAnalysis],
) -> list[str]:
    """
    Collect all affected trace IDs for an identified issue.

    Args:
        issue: The identified issue containing indices into the analyses list.
        analyses: List of per-session analyses with affected trace IDs.

    Returns:
        A list of all trace IDs affected by this issue.
    """
    trace_ids = []
    for idx in issue.example_indices:
        if 0 <= idx < len(analyses):
            trace_ids.extend(analyses[idx].affected_trace_ids)
    return trace_ids


def format_trace_content(trace: Trace, include_timing: bool = False) -> str:
    """
    Format trace content for annotation prompts.

    Args:
        trace: The trace to format.
        include_timing: If True, include timing information (duration and slowest spans).
                       Should be enabled when latency detection is active.

    Returns:
        Formatted trace content string.
    """
    from mlflow.genai.discovery.extraction import extract_execution_path, extract_span_errors

    # import here to avoid circular import
    from mlflow.genai.utils.trace_utils import _extract_trace_timing_info

    parts = []
    if request := trace.data.request:
        parts.append(f"Input: {str(request)[:TRACE_CONTENT_TRUNCATION]}")
    if response := trace.data.response:
        parts.append(f"Output: {str(response)[:TRACE_CONTENT_TRUNCATION]}")

    if include_timing:
        if timing_info := _extract_trace_timing_info(trace):
            parts.append(f"Total duration: {timing_info['duration_s']:.2f}s")
            if slowest_spans_formatted := timing_info["slowest_spans_formatted"]:
                parts.append(f"Slowest spans: {slowest_spans_formatted}")

    if (exec_path := extract_execution_path(trace)) and exec_path != "(no routing)":
        parts.append(f"Execution path: {exec_path}")
    if errors := extract_span_errors(trace):
        parts.append(f"Errors: {errors}")
    return "\n".join(parts) if parts else "(trace content not available)"


def format_annotation_prompt(
    issue: Issue,
    trace_content: str,
    triage_rationale: str,
    categories: list[str] | None = None,
) -> str:
    prompt = (
        f"=== ISSUE ===\n"
        f"Name: {issue.name}\n"
        f"Description: {issue.description}\n"
        f"Root causes: {'; '.join(issue.root_causes or [])}\n\n"
        f"=== TRACE ===\n"
        f"{trace_content}\n\n"
        f"=== TRIAGE JUDGE RATIONALE ===\n"
        f"{triage_rationale or '(not available)'}"
    )
    if categories:
        prompt += (
            f"\n\n=== RELEVANT CATEGORIES ===\n"
            f"{', '.join(categories)}\n"
            f"Reference these categories in your rationale where applicable."
        )
    return prompt
