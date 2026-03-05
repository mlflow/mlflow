from __future__ import annotations

import logging
import random
from collections import defaultdict

import mlflow
from mlflow.entities.assessment import Feedback
from mlflow.entities.trace import Trace
from mlflow.genai.discovery.constants import (
    SAMPLE_POOL_MULTIPLIER,
    SAMPLE_RANDOM_SEED,
)
from mlflow.genai.scorers.base import Scorer
from mlflow.tracing.constant import TraceMetadataKey

_logger = logging.getLogger(__name__)


def get_session_id(trace: Trace) -> str | None:
    return (trace.info.trace_metadata or {}).get(TraceMetadataKey.TRACE_SESSION)


def sample_traces(
    sample_size: int,
    search_kwargs: dict[str, object],
) -> list[Trace]:
    """
    Randomly sample traces, grouping by session when session IDs exist.

    Fetches a pool of traces, groups them by session (or treats each trace
    as its own group when no sessions exist), then randomly selects
    ``sample_size`` groups and returns all traces from those groups.

    Args:
        sample_size: Number of groups (sessions or individual traces) to sample.
        search_kwargs: Keyword arguments passed to ``mlflow.search_traces``.

    Returns:
        List of sampled Trace objects.
    """
    pool_size = sample_size * SAMPLE_POOL_MULTIPLIER
    pool = mlflow.search_traces(max_results=pool_size, **search_kwargs)
    if not pool:
        return []

    # Group traces by session; traces without a session become their own group
    groups: dict[str, list[Trace]] = defaultdict(list)
    for trace in pool:
        key = get_session_id(trace) or trace.info.trace_id
        groups[key].append(trace)

    rng = random.Random(SAMPLE_RANDOM_SEED)
    group_keys = sorted(groups.keys())
    num_samples = min(sample_size, len(group_keys))
    selected = rng.sample(group_keys, num_samples)
    result = [trace for key in selected for trace in groups[key]]
    _logger.info(
        "Sampled %d groups (%d traces) from pool of %d groups",
        num_samples,
        len(result),
        len(group_keys),
    )
    return result


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


def verify_scorer(scorer: Scorer, trace: Trace) -> None:
    """
    Verify a scorer works on a single trace before running the full pipeline.

    Calls the scorer on the trace, fetches the updated trace, and checks
    that a Feedback assessment with a non-null value was produced.

    Args:
        scorer: The scorer to test.
        trace: A trace to run the scorer on.

    Raises:
        MlflowException: If the scorer produces no feedback or returns a null value.
    """
    try:
        scorer(trace=trace)
        result_trace = mlflow.get_trace(trace.info.trace_id)
        if result_trace is None:
            raise mlflow.exceptions.MlflowException(
                f"Scorer '{scorer.name}' produced no feedback on test trace"
            )
        feedback = next(
            (
                assessment
                for assessment in result_trace.info.assessments
                if isinstance(assessment, Feedback) and assessment.name == scorer.name
            ),
            None,
        )
        if feedback is None:
            raise mlflow.exceptions.MlflowException(
                f"Scorer '{scorer.name}' produced no feedback on test trace"
            )
        if feedback.value is None:
            error = feedback.error_message or "unknown error (check model API logs)"
            raise mlflow.exceptions.MlflowException(
                f"Scorer '{scorer.name}' returned null value: {error}"
            )
    except Exception as exc:
        raise mlflow.exceptions.MlflowException(
            f"Scorer '{scorer.name}' failed verification on trace {trace.info.trace_id}: {exc}"
        ) from exc
