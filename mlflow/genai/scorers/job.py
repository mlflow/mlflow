"""Huey job functions for async scorer invocation."""

import logging
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any

from mlflow.entities import Trace
from mlflow.environment_variables import (
    MLFLOW_GENAI_EVAL_MAX_WORKERS,
    MLFLOW_SERVER_JUDGE_INVOKE_MAX_WORKERS,
    MLFLOW_SERVER_ONLINE_SCORING_MAX_WORKERS,
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
from mlflow.genai.scorers.online import (
    OnlineScorer,
    OnlineScoringConfig,
    OnlineSessionScoringProcessor,
    OnlineTraceScoringProcessor,
)
from mlflow.server.handlers import _get_tracking_store
from mlflow.server.jobs import job, submit_job
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.tracing.constant import TraceMetadataKey

_logger = logging.getLogger(__name__)

# Constants for job names that are referenced in multiple locations
ONLINE_TRACE_SCORER_JOB_NAME = "run_online_trace_scorer"
ONLINE_SESSION_SCORER_JOB_NAME = "run_online_session_scorer"


@dataclass
class ScorerFailure:
    error_code: str
    error_message: str


@dataclass
class TraceResult:
    assessments: list[Any] = field(default_factory=list)
    failures: list[ScorerFailure] = field(default_factory=list)


def _extract_failures_from_feedbacks(feedbacks: list[Any]) -> list[ScorerFailure]:
    return [
        ScorerFailure(
            error_code=feedback.error.error_code,
            error_message=feedback.error.error_message,
        )
        for feedback in feedbacks
        if feedback.error
    ]


@job(
    name=ONLINE_TRACE_SCORER_JOB_NAME,
    max_workers=MLFLOW_SERVER_ONLINE_SCORING_MAX_WORKERS.get(),
    exclusive=["experiment_id"],
)
def run_online_trace_scorer_job(
    experiment_id: str,
    online_scorers: list[dict[str, Any]],
) -> None:
    """
    Job that fetches samples of individual traces and runs scorers on them.

    This job is exclusive per experiment_id to prevent duplicate scoring of the same
    experiment. Multiple jobs with different scorers for the same experiment will not
    run simultaneously, ensuring consistent checkpoint management.

    Args:
        experiment_id: The experiment ID to fetch traces from.
        online_scorers: List of OnlineScorer dicts specifying which scorers to run.
    """
    scorer_objects = [
        OnlineScorer(
            name=scorer_dict["name"],
            serialized_scorer=scorer_dict["serialized_scorer"],
            online_config=OnlineScoringConfig(**scorer_dict["online_config"]),
        )
        for scorer_dict in online_scorers
    ]

    tracking_store = _get_tracking_store()
    processor = OnlineTraceScoringProcessor.create(experiment_id, scorer_objects, tracking_store)
    processor.process_traces()


@job(
    name=ONLINE_SESSION_SCORER_JOB_NAME,
    max_workers=MLFLOW_SERVER_ONLINE_SCORING_MAX_WORKERS.get(),
    exclusive=["experiment_id"],
)
def run_online_session_scorer_job(
    experiment_id: str,
    online_scorers: list[dict[str, Any]],
) -> None:
    """
    Job that finds completed sessions and runs session-level scorers on them.

    This job is exclusive per experiment_id to prevent duplicate scoring of the same
    experiment. Multiple jobs with different scorers for the same experiment will not
    run simultaneously, ensuring consistent checkpoint management.

    Args:
        experiment_id: The experiment ID to fetch sessions from.
        online_scorers: List of OnlineScorer dicts specifying which scorers to run.
    """
    scorer_objects = [
        OnlineScorer(
            name=scorer_dict["name"],
            serialized_scorer=scorer_dict["serialized_scorer"],
            online_config=OnlineScoringConfig(**scorer_dict["online_config"]),
        )
        for scorer_dict in online_scorers
    ]

    tracking_store = _get_tracking_store()
    processor = OnlineSessionScoringProcessor.create(experiment_id, scorer_objects, tracking_store)
    processor.process_sessions()


@job(name="invoke_scorer", max_workers=MLFLOW_SERVER_JUDGE_INVOKE_MAX_WORKERS.get())
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
        Dict mapping trace_id to TraceResult (assessments and failures).
    """
    # Deserialize scorer
    scorer = Scorer.model_validate_json(serialized_scorer)

    tracking_store = _get_tracking_store()

    if scorer.is_session_level_scorer:
        result = _run_session_scorer(scorer, trace_ids, tracking_store, log_assessments)
    else:
        result = _run_single_turn_scorer_batch(scorer, trace_ids, tracking_store, log_assessments)

    return {trace_id: asdict(trace_result) for trace_id, trace_result in result.items()}


def _fetch_traces_batch(
    trace_ids: list[str],
    tracking_store: AbstractStore,
) -> dict[str, Trace]:
    """
    Fetch traces in batch and return a mapping.

    Args:
        trace_ids: List of trace IDs to fetch.
        tracking_store: The tracking store instance.

    Returns:
        Dict mapping trace_id to Trace.

    Raises:
        MlflowException: If any trace IDs are not found.
    """
    traces = tracking_store.batch_get_traces(trace_ids)
    trace_map = {t.info.trace_id: t for t in traces}

    if missing_ids := [tid for tid in trace_ids if tid not in trace_map]:
        raise MlflowException(f"Traces not found: {missing_ids}")

    return trace_map


def _run_session_scorer(
    scorer: Any,
    trace_ids: list[str],
    tracking_store: AbstractStore,
    log_assessments: bool,
) -> dict[str, TraceResult]:
    """
    Run a session-level scorer on all traces as a conversation.

    Reuses evaluate_session_level_scorers from the evaluation harness.

    Args:
        scorer: The scorer instance.
        trace_ids: List of trace IDs (representing a session).
        tracking_store: The tracking store instance.
        log_assessments: Whether to log assessments to the traces.

    Returns:
        Dict mapping trace_id to TraceResult.
    """
    trace_map = _fetch_traces_batch(trace_ids, tracking_store)

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

        failures = _extract_failures_from_feedbacks(feedbacks)

        if log_assessments and feedbacks:
            _log_assessments(
                run_id=None,  # No MLflow run context in API path
                trace=first_trace,
                assessments=feedbacks,
            )

        return {
            first_trace_id: TraceResult(
                assessments=[f.to_dictionary() for f in feedbacks],
                failures=failures,
            )
        }
    except Exception as e:
        return {
            first_trace_id: TraceResult(
                failures=[ScorerFailure(error_code=type(e).__name__, error_message=str(e))]
            )
        }


def _run_single_turn_scorer_batch(
    scorer: Any,
    trace_ids: list[str],
    tracking_store: AbstractStore,
    log_assessments: bool,
) -> dict[str, TraceResult]:
    """
    Run a single-turn scorer on each trace in parallel.

    Args:
        scorer: The scorer instance.
        trace_ids: List of trace IDs to evaluate.
        tracking_store: The tracking store instance.
        log_assessments: Whether to log assessments to the traces.

    Returns:
        Dict mapping trace_id to TraceResult.
    """
    trace_map = _fetch_traces_batch(trace_ids, tracking_store)

    def process_trace(trace_id: str, trace: Trace) -> tuple[str, TraceResult]:
        eval_item = EvalItem.from_trace(trace)

        try:
            # Use _compute_eval_scores from harness - supports scorer tracing,
            # captures stack traces on errors
            feedbacks = _compute_eval_scores(
                eval_item=eval_item,
                scorers=[scorer],
            )

            failures = _extract_failures_from_feedbacks(feedbacks)

            if log_assessments and feedbacks:
                _log_assessments(
                    run_id=None,  # No MLflow run context in API path
                    trace=trace,
                    assessments=feedbacks,
                )

            return trace_id, TraceResult(
                assessments=[f.to_dictionary() for f in feedbacks],
                failures=failures,
            )
        except Exception as e:
            return trace_id, TraceResult(
                failures=[ScorerFailure(error_code=type(e).__name__, error_message=str(e))]
            )

    max_workers = min(len(trace_map), MLFLOW_GENAI_EVAL_MAX_WORKERS.get())
    results: dict[str, TraceResult] = {}

    with ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="MlflowScorerInvoke",
    ) as executor:
        futures = {
            executor.submit(process_trace, tid, trace): tid for tid, trace in trace_map.items()
        }
        for future in as_completed(futures):
            trace_id, result = future.result()
            results[trace_id] = result

    return results


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

    trace_infos = tracking_store.batch_get_trace_infos(trace_ids)

    for trace_info in trace_infos:
        trace_metadata = trace_info.trace_metadata or {}
        if session_id := trace_metadata.get(TraceMetadataKey.TRACE_SESSION):
            if session_id not in session_groups:
                session_groups[session_id] = []
            session_groups[session_id].append(trace_info.trace_id)
            trace_info_cache[trace_info.trace_id] = (session_id, trace_info.timestamp_ms)

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


def run_online_scoring_scheduler() -> None:
    """
    Periodic task that fetches active online scorers and submits scoring jobs.

    Groups scorers by experiment_id and submits two jobs per experiment:
    1. Trace-level scoring job for single-turn scorers
    2. Session-level scoring job for session scorers

    Groups are shuffled to prevent starvation when there are limited job runners available.
    """
    tracking_store = _get_tracking_store()
    online_scorers = tracking_store.get_active_online_scorers()
    _logger.debug(f"Online scoring scheduler found {len(online_scorers)} active scorers")

    scorers_by_experiment: dict[str, list[OnlineScorer]] = defaultdict(list)
    for scorer in online_scorers:
        scorers_by_experiment[scorer.online_config.experiment_id].append(scorer)

    # Shuffle configs randomly to prevent scorer starvation when there are
    # limited job runners available
    experiment_groups = list(scorers_by_experiment.items())
    random.shuffle(experiment_groups)
    _logger.debug(
        f"Grouped into {len(experiment_groups)} experiments, submitting jobs per experiment"
    )

    for experiment_id, scorers in experiment_groups:
        # Separate scorers by type
        session_level_scorers = []
        trace_level_scorers = []

        for scorer in scorers:
            try:
                scorer_obj = Scorer.model_validate_json(scorer.serialized_scorer)
                if scorer_obj.is_session_level_scorer:
                    session_level_scorers.append(scorer)
                else:
                    trace_level_scorers.append(scorer)
            except Exception as e:
                _logger.warning(
                    f"Failed to load scorer '{scorer.name}'; scorer will be skipped: {e}"
                )

        # Only submit jobs for scorer types that exist
        if trace_level_scorers:
            _logger.debug(
                f"Submitting trace scoring job for experiment {experiment_id} "
                f"with {len(trace_level_scorers)} scorers"
            )
            trace_scorer_dicts = [asdict(scorer) for scorer in trace_level_scorers]
            submit_job(
                run_online_trace_scorer_job,
                {"experiment_id": experiment_id, "online_scorers": trace_scorer_dicts},
            )

        if session_level_scorers:
            _logger.debug(
                f"Submitting session scoring job for experiment {experiment_id} "
                f"with {len(session_level_scorers)} scorers"
            )
            session_scorer_dicts = [asdict(scorer) for scorer in session_level_scorers]
            submit_job(
                run_online_session_scorer_job,
                {"experiment_id": experiment_id, "online_scorers": session_scorer_dicts},
            )
