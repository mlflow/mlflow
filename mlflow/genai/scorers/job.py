"""
Huey job function for async scorer invocation.

This module provides the job function for invoking scorers on traces asynchronously.
It reuses the core scoring and logging logic from the evaluation harness for consistency.
"""

import functools
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from mlflow.entities import Trace
from mlflow.environment_variables import (
    MLFLOW_SERVER_JUDGE_INVOKE_MAX_WORKERS,
    MLFLOW_SERVER_SCORER_INVOKE_BATCH_SIZE,
)
from mlflow.exceptions import MlflowException
from mlflow.genai.evaluation.entities import EvalItem
from mlflow.genai.evaluation.harness import _compute_eval_scores, _log_assessments
from mlflow.genai.evaluation.session_utils import (
    evaluate_session_level_scorers,
    get_first_trace_in_session,
)
from mlflow.genai.scorers.base import Scorer
from mlflow.server.jobs import job
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.tracing.constant import TraceMetadataKey


@dataclass
class ScorerFailure:
    trace_id: str | None
    error_code: str
    error_message: str


@dataclass
class InvokeScorerResult:
    trace_ids: list[str]
    assessments: dict[str, list[Any]] = field(default_factory=dict)
    failures: list[ScorerFailure] = field(default_factory=list)


def _make_failure(trace_id: str | None, error: Exception, prefix: str = "") -> ScorerFailure:
    msg = f"{prefix}{error}" if prefix else str(error)
    return ScorerFailure(trace_id=trace_id, error_code=type(error).__name__, error_message=msg)


def _translate_scorer_job_failure(
    func: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    """
    Decorator that catches uncaught exceptions and converts them to failure responses.

    Acts as a safety net for job-level failures (e.g., deserialization errors,
    unexpected crashes). Per-trace error handling is still done within the
    batch processing functions.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> dict[str, Any]:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Extract trace_ids from args/kwargs for the failure response
            # trace_ids is the 3rd positional arg or a kwarg
            trace_ids = kwargs.get("trace_ids") or (args[2] if len(args) > 2 else [])
            return asdict(
                InvokeScorerResult(
                    trace_ids=trace_ids,
                    failures=[_make_failure(None, e)],
                )
            )

    return wrapper


@job(name="invoke_scorer", max_workers=MLFLOW_SERVER_JUDGE_INVOKE_MAX_WORKERS.get())
@_translate_scorer_job_failure
def invoke_scorer_job(
    experiment_id: str,
    serialized_scorer: str,
    trace_ids: list[str],
    log_assessments: bool = True,
) -> dict[str, Any]:
    """
    Huey job function for async scorer invocation.

    Reuses the core scoring logic from the evaluation harness
    for consistency and feature parity.

    Args:
        experiment_id: The experiment ID for the traces.
        serialized_scorer: JSON string of the serialized scorer.
        trace_ids: List of trace IDs to evaluate.
        log_assessments: Whether to log assessments to the traces.

    Returns:
        A dictionary containing:
        - trace_ids: The trace IDs that were requested
        - assessments: Dict mapping trace_id to list of assessment dictionaries
        - failures: List of failure dictionaries with trace_id, error_code, error_message
    """
    from mlflow.server.handlers import _get_tracking_store

    # Deserialize scorer
    scorer_dict = json.loads(serialized_scorer)
    scorer = Scorer.model_validate(scorer_dict)

    tracking_store = _get_tracking_store()

    if scorer.is_session_level_scorer:
        result = _run_session_scorer(scorer, trace_ids, tracking_store, log_assessments)
    else:
        result = _run_single_turn_scorer_batch(scorer, trace_ids, tracking_store, log_assessments)

    return asdict(result)


def _fetch_traces_batch(
    trace_ids: list[str],
    tracking_store: AbstractStore,
) -> tuple[dict[str, Trace], list[ScorerFailure]]:
    """
    Fetch traces in batch and return a mapping plus failures for missing traces.

    Args:
        trace_ids: List of trace IDs to fetch.
        tracking_store: The tracking store instance.

    Returns:
        Tuple of (trace_id -> trace mapping, list of failures for missing traces).

    Raises:
        Exception: If batch_get_traces fails (e.g., DB error).
    """
    traces = tracking_store.batch_get_traces(trace_ids)
    trace_map = {t.info.trace_id: t for t in traces}

    # Detect missing traces
    failures = [
        ScorerFailure(
            trace_id=trace_id,
            error_code="TRACE_NOT_FOUND",
            error_message=f"Trace not found: {trace_id}",
        )
        for trace_id in trace_ids
        if trace_id not in trace_map
    ]

    return trace_map, failures


def _run_session_scorer(
    scorer: Any,
    trace_ids: list[str],
    tracking_store: AbstractStore,
    log_assessments: bool,
) -> InvokeScorerResult:
    """
    Run a session-level scorer on all traces as a conversation.

    Reuses evaluate_session_level_scorers from the evaluation harness.

    Args:
        scorer: The scorer instance.
        trace_ids: List of trace IDs (representing a session).
        tracking_store: The tracking store instance.
        log_assessments: Whether to log assessments to the traces.

    Returns:
        Result dictionary with assessments and failures.
    """
    trace_map, failures = _fetch_traces_batch(trace_ids, tracking_store)

    # For session scorers, we need all traces - fail if any are missing
    if failures:
        return InvokeScorerResult(
            trace_ids=trace_ids,
            failures=failures,
        )

    # Preserve order of traces as requested
    traces = [trace_map[tid] for tid in trace_ids]

    session_items = [EvalItem.from_trace(t) for t in traces]

    # Get session_id from the first trace's metadata
    first_session_item = get_first_trace_in_session(session_items)
    trace_metadata = first_session_item.trace.info.trace_metadata or {}
    session_id = trace_metadata.get(TraceMetadataKey.TRACE_SESSION)

    if not session_id:
        raise MlflowException(
            "Session-level scorer requires traces with session metadata. "
            f"Trace {first_session_item.trace.info.trace_id} is missing "
            f"'{TraceMetadataKey.TRACE_SESSION}' in its metadata."
        )

    first_trace = first_session_item.trace
    first_trace_id = first_trace.info.trace_id

    try:
        result = evaluate_session_level_scorers(
            session_id=session_id,
            session_items=session_items,
            multi_turn_scorers=[scorer],
        )

        # result is {first_trace_id: [feedbacks]}
        feedbacks = result[first_trace_id]

        # Log assessments if requested
        if log_assessments and feedbacks:
            _log_assessments(
                run_id=None,  # No MLflow run context in API path
                trace=first_trace,
                assessments=feedbacks,
            )

        return InvokeScorerResult(
            trace_ids=trace_ids,
            assessments={first_trace_id: [f.to_dictionary() for f in feedbacks]},
        )
    except Exception as e:
        return InvokeScorerResult(
            trace_ids=trace_ids,
            failures=[_make_failure(first_trace_id, e)],
        )


def _run_single_turn_scorer_batch(
    scorer: Any,
    trace_ids: list[str],
    tracking_store: AbstractStore,
    log_assessments: bool,
) -> InvokeScorerResult:
    """
    Run a single-turn scorer on each trace individually (batch processing).

    Reuses _compute_eval_scores from the evaluation harness.

    Args:
        scorer: The scorer instance.
        trace_ids: List of trace IDs to evaluate.
        tracking_store: The tracking store instance.
        log_assessments: Whether to log assessments to the traces.

    Returns:
        Result dictionary with assessments and failures.
    """
    trace_map, failures = _fetch_traces_batch(trace_ids, tracking_store)

    assessments_by_trace: dict[str, list[dict[str, Any]]] = {}

    for trace_id, trace in trace_map.items():
        eval_item = EvalItem.from_trace(trace)

        try:
            # Use _compute_eval_scores from harness - supports scorer tracing,
            # captures stack traces on errors
            feedbacks = _compute_eval_scores(
                eval_item=eval_item,
                scorers=[scorer],
            )

            # Log assessments if requested
            if log_assessments and feedbacks:
                _log_assessments(
                    run_id=None,  # No MLflow run context in API path
                    trace=trace,
                    assessments=feedbacks,
                )

            assessments_by_trace[trace_id] = [f.to_dictionary() for f in feedbacks]
        except Exception as e:
            failures.append(_make_failure(trace_id, e))

    return InvokeScorerResult(
        trace_ids=trace_ids,
        assessments=assessments_by_trace,
        failures=failures,
    )


def _group_traces_by_session_id(
    trace_ids: list[str],
    tracking_store: AbstractStore,
) -> dict[str, list[str]]:
    """
    Group trace_ids by their session_id metadata.

    Fetches trace info from the tracking store and groups them by session_id.
    Traces without a session_id are skipped.

    Args:
        trace_ids: List of trace IDs to group.
        tracking_store: The tracking store instance.

    Returns:
        Dictionary mapping session_id to list of trace_ids, sorted by timestamp.
    """
    session_groups: dict[str, list[str]] = {}
    # trace_id -> (session_id, timestamp_ms)
    trace_info_cache: dict[str, tuple[str, int | None]] = {}

    for trace_id in trace_ids:
        try:
            if trace_info := tracking_store.get_trace_info(trace_id):
                trace_metadata = trace_info.trace_metadata or {}
                if session_id := trace_metadata.get(TraceMetadataKey.TRACE_SESSION):
                    if session_id not in session_groups:
                        session_groups[session_id] = []
                    session_groups[session_id].append(trace_id)
                    trace_info_cache[trace_id] = (session_id, trace_info.timestamp_ms)
        except Exception:
            # Skip traces that can't be fetched
            pass

    # Sort trace_ids within each session by trace timestamp (None timestamps sort last)
    for session_id in session_groups:
        session_groups[session_id] = sorted(
            session_groups[session_id],
            key=lambda tid: trace_info_cache.get(tid, ("", None))[1] or float("inf"),
        )

    return session_groups


def get_trace_batches_for_scorer(
    trace_ids: list[str],
    scorer: Scorer,
    tracking_store: AbstractStore,
) -> list[list[str]]:
    """
    Get trace ID batches for scorer invocation.

    Handles batching logic for both session-level and single-turn scorers:
    - Session-level scorers: Groups traces by session_id, returns one batch per session.
    - Single-turn scorers: Batches traces based on MLFLOW_SERVER_SCORER_INVOKE_BATCH_SIZE.

    Args:
        trace_ids: List of trace IDs to evaluate.
        scorer: The validated Scorer instance.
        tracking_store: The tracking store instance.

    Returns:
        List of trace ID batches, where each batch should be submitted as a separate job.
    """
    if scorer.is_session_level_scorer:
        # For conversation judges, group traces by session_id
        session_groups = _group_traces_by_session_id(trace_ids, tracking_store)
        return list(session_groups.values())
    else:
        # For single-turn judges, batch traces into fixed-size batches
        batch_size = MLFLOW_SERVER_SCORER_INVOKE_BATCH_SIZE.get()
        return [trace_ids[i : i + batch_size] for i in range(0, len(trace_ids), batch_size)]
