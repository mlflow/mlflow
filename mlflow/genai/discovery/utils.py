from __future__ import annotations

import logging
import math
import statistics
from collections import defaultdict

import mlflow
from mlflow.entities.assessment import Feedback
from mlflow.entities.trace import Trace
from mlflow.entities.view_type import ViewType
from mlflow.genai.discovery.constants import (
    TRACE_CONTENT_TRUNCATION,
)
from mlflow.genai.discovery.entities import (
    CostAnomaly,
    Issue,
    _ConversationAnalysis,
    _IdentifiedIssue,
)
from mlflow.genai.scorers.base import Scorer
from mlflow.tracing.constant import TraceMetadataKey, TraceTagKey
from mlflow.tracing.provider import trace_disabled

_logger = logging.getLogger(__name__)


def in_flight_detection_run(store, experiment_id: str, now_ms: int | None = None) -> str | None:
    """
    The id of a detection run still going in this experiment, or ``None``.

    Ad-hoc and scheduled detection are kept from overlapping. Two runs over the same window
    analyze the same traces and file the same findings: cross-run matching collapses the
    duplicates afterwards, but both runs have already paid the model for the same work, and a
    concurrent refresh of one issue from two runs is a race there is no reason to carry.

    Checked here rather than with the job engine's exclusive lock, which cannot express it: the
    lock key is ``f"{job_name}:{params_hash}"`` and on the Huey engine it lives in a per-job-name
    store, so it cannot span two job functions -- and its contention behaviour is to cancel the
    loser, which would silently discard a run the user had just asked for.

    A run is only counted while its job could still be alive. Past the job engine's default
    timeout the job is gone, so a run left RUNNING by a worker that died cannot block detection
    for this experiment forever.

    Args:
        store: The tracking store to search.
        experiment_id: The experiment whose detection runs to look at.
        now_ms: The instant to measure the staleness bound against. Defaults to now; the
            dispatcher passes its own clock so the bound moves with the occurrence it is
            deciding about rather than with wall time.
    """
    from mlflow.environment_variables import MLFLOW_SERVER_JOB_DEFAULT_TIMEOUT
    from mlflow.utils.mlflow_tags import MLFLOW_RUN_TYPE, MLFLOW_RUN_TYPE_ISSUE_DETECTION
    from mlflow.utils.time import get_current_time_millis

    now_ms = get_current_time_millis() if now_ms is None else now_ms
    oldest_live_start_ms = now_ms - int(MLFLOW_SERVER_JOB_DEFAULT_TIMEOUT.get() * 1000)
    runs = store.search_runs(
        experiment_ids=[experiment_id],
        filter_string=(
            f"tags.`{MLFLOW_RUN_TYPE}` = '{MLFLOW_RUN_TYPE_ISSUE_DETECTION}' "
            f"AND attributes.status = 'RUNNING' "
            f"AND attributes.start_time > {oldest_live_start_ms}"
        ),
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
    )
    return runs[0].info.run_id if runs else None


def safe_trace_id(trace: Trace, position: int) -> str:
    """
    A trace's ID, or a positional stand-in when even that cannot be read.

    Falling back rather than raising keeps the trace counted: dropping it would shrink the
    sample the run reports against.
    """
    try:
        return trace.info.trace_id
    except Exception:
        return f"(unreadable trace at position {position})"


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


# Cost anomaly detection. Spend is compared on a log scale against a robust centre and
# spread (median and median absolute deviation), because agent spend is right-skewed: a mean
# and standard deviation are pulled upward by the very traces being tested, so a sample's most
# expensive trace clears a 3-sigma bar in most healthy runs. A robust spread also lets each
# workload set its own bar -- a tight workload flags an 8x trace, one that routinely varies
# 30x does not -- which is what keeps a cost issue from being filed every run.
MIN_COST_SAMPLE = 10
# Robust deviations above typical at which spend is called anomalous. Chosen by simulation
# over lognormal samples: ~0.5% of healthy runs file a cost issue at any spread, while a
# genuine runaway (40x the typical trace) is caught in 99.9% of runs.
COST_ANOMALY_DEVIATIONS = 4.5
# Above this share of a group, "anomalies" are the distribution rather than outliers in it,
# so none are reported. Also guards a degenerate group whose spread is near zero.
MAX_ANOMALOUS_FRACTION = 0.10
# When the group agrees on one figure there is no spread to measure against, so a multiple of
# it stands in. Deliberately well above the ordinary bar: unanimous peers are usually a sign of
# synthetic or heavily cached work, where small differences mean little.
UNANIMOUS_PEER_RATIO = 3.0
# MAD to standard-deviation scale factor for a normal distribution.
_MAD_TO_SIGMA = 1.4826


def get_agent_label(trace: Trace) -> str:
    """
    The label whose peers a trace's spend is compared against.

    OSS traces carry no dedicated agent field, so the trace name is used: it is the root span's
    name, which is the entry point the application was called through, and is what distinguishes
    a retrieval agent from a summarizer in practice.
    """
    tags = trace.info.tags or {}
    if name := tags.get(TraceTagKey.TRACE_NAME):
        # Coerced because the label only has to group and be printable: a tag written with a
        # non-string value should bucket the trace, not raise while grouping the sample.
        return str(name)
    for span in trace.data.spans or []:
        if span.parent_id is None and span.name:
            return str(span.name)
    return "(unnamed)"


def _trace_cost(trace: Trace) -> float | None:
    total = (trace.info.cost or {}).get("total_cost")
    return float(total) if total is not None else None


def _anomalies_in_group(
    label: str, priced: list[tuple[str, float]], deviations: float
) -> list[CostAnomaly]:
    logs = [math.log(cost) for _, cost in priced]
    centre = statistics.median(logs)
    mad = statistics.median([abs(value - centre) for value in logs])
    typical = math.exp(centre)

    if mad > 0:
        spread = _MAD_TO_SIGMA * mad
        scored = [(trace_id, cost, (math.log(cost) - centre) / spread) for trace_id, cost in priced]
        found = [entry for entry in scored if entry[2] >= deviations]
    else:
        # More than half the group agrees on one figure, so there is no spread to scale by.
        # Estimating one from the whole group would let a runaway set its own threshold: the
        # standard deviation grows with it, and the score saturates no matter how extreme the
        # trace gets. Against unanimous peers, a material multiple is the evidence.
        found = [
            (trace_id, cost, math.inf)
            for trace_id, cost in priced
            if cost >= UNANIMOUS_PEER_RATIO * typical
        ]

    if len(found) > MAX_ANOMALOUS_FRACTION * len(priced):
        # The qualifying share is the distribution, not an outlier in it.
        return []
    return sorted(
        (
            CostAnomaly(
                trace_id=trace_id,
                label=label,
                cost_usd=cost,
                typical_usd=typical,
                peer_count=len(priced),
                deviations=score,
            )
            for trace_id, cost, score in found
        ),
        key=lambda anomaly: anomaly.cost_usd,
        reverse=True,
    )


def analyze_cost_anomalies(
    traces: list[Trace],
    *,
    deviations: float = COST_ANOMALY_DEVIATIONS,
    min_sample: int = MIN_COST_SAMPLE,
) -> list[CostAnomaly]:
    """
    Traces whose spend is anomalous for the sample, or an empty list.

    Cost is the one category the triage judge cannot reach: every other category is decided
    from one trace's content, but "expensive" only means anything relative to other traces,
    and a judge scoring one trace at a time cannot see them.

    Traces are grouped by agent label so a summarizer is not compared against a retriever,
    and a label without enough priced peers is not judged at all. Pooling such labels together
    was tried and removed: it reports whichever label happens to be the most expensive in the
    sample, so two ordinary traces of a costlier agent are flagged purely because the other
    agents in the sample are cheap, and adding same-agent peers makes the finding vanish
    without those traces changing. An unjudged label is the honest outcome -- "expensive" has
    no meaning without comparable peers. A trace with no recorded cost is excluded, never read
    as zero spend.

    Args:
        traces: The sampled traces. Only recorded cost is read.
        deviations: Robust deviations above typical required to call spend anomalous.
        min_sample: Priced traces a label needs before its spend is judged.

    Returns:
        The anomalies, most expensive first. Empty when no label has enough priced peers,
        when spend is uniformly spread, or when so much of it qualifies that the spread is
        the distribution rather than an outlier in it.
    """
    by_label: dict[str, list[tuple[str, float]]] = {}
    unpriced = 0
    for position, trace in enumerate(traces):
        trace_id = safe_trace_id(trace, position)
        try:
            cost = _trace_cost(trace)
            label = get_agent_label(trace)
        except Exception:
            # Cost findings are advisory: a trace whose usage cannot be read is left out
            # rather than taking the run down with it.
            _logger.debug("Could not read cost for trace %s", trace_id, exc_info=True)
            unpriced += 1
            continue
        if cost is None or cost <= 0:
            unpriced += 1
            continue
        by_label.setdefault(label, []).append((trace_id, cost))

    judged = {label: priced for label, priced in by_label.items() if len(priced) >= min_sample}

    anomalies: list[CostAnomaly] = []
    for label, priced in sorted(judged.items(), key=lambda item: str(item[0])):
        anomalies.extend(_anomalies_in_group(label, priced, deviations))

    if skipped := {label: len(priced) for label, priced in by_label.items() if label not in judged}:
        _logger.info(
            "Cost was not judged for %d agent label(s) with fewer than %d priced traces: %s",
            len(skipped),
            min_sample,
            ", ".join(f"{label!r} ({count})" for label, count in sorted(skipped.items())),
        )
    if unpriced:
        _logger.info(
            "%d traces were left out of cost analysis because they record no positive cost",
            unpriced,
        )
    return sorted(anomalies, key=lambda anomaly: anomaly.cost_usd, reverse=True)


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
