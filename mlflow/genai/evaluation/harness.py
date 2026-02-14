"""Entry point to the evaluation harness"""

from __future__ import annotations

import logging
import queue
import threading
import time
import traceback
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from typing import Any, Callable

import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:
    # If tqdm is not installed, we don't show a progress bar
    tqdm = None

# Optional dependencies — imported eagerly in the main thread so that worker
# threads never trigger first-time imports (which can deadlock under Python's
# per-module import lock when many threads import simultaneously).
try:
    import litellm  # noqa: F401
except ImportError:
    pass
try:
    from databricks.sdk import WorkspaceClient  # noqa: F401
except ImportError:
    pass

import mlflow
from mlflow.entities import SpanType
from mlflow.entities.assessment import Assessment, Expectation, Feedback
from mlflow.entities.assessment_error import AssessmentError
from mlflow.entities.trace import Trace
from mlflow.environment_variables import (
    MLFLOW_GENAI_EVAL_ENABLE_SCORER_TRACING,
    MLFLOW_GENAI_EVAL_MAX_RETRIES,
    MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS,
    MLFLOW_GENAI_EVAL_MAX_WORKERS,
    MLFLOW_GENAI_EVAL_PREDICT_RATE_LIMIT,
    MLFLOW_GENAI_EVAL_RATE_LIMIT_UPPER_MULTIPLIER,
    MLFLOW_GENAI_EVAL_SCORER_RATE_LIMIT,
)
from mlflow.genai.evaluation import context
from mlflow.genai.evaluation.entities import EvalItem, EvalResult, EvaluationResult
from mlflow.genai.evaluation.rate_limiter import (
    NoOpRateLimiter,
    RateLimiter,
    RPSRateLimiter,
    call_with_retry,
    eval_retry_context,
)
from mlflow.genai.evaluation.session_utils import (
    classify_scorers,
    evaluate_session_level_scorers,
    group_traces_by_session,
)
from mlflow.genai.evaluation.telemetry import emit_metric_usage_event
from mlflow.genai.evaluation.utils import (
    PGBAR_FORMAT,
    is_none_or_nan,
    make_code_type_assessment_source,
    standardize_scorer_value,
    validate_tags,
)
from mlflow.genai.scorers.aggregation import compute_aggregated_metrics
from mlflow.genai.scorers.base import Scorer
from mlflow.genai.utils.trace_utils import (
    _does_store_support_trace_linking,
    batch_link_traces_to_run,
    clean_up_extra_traces,
    construct_eval_result_df,
    create_minimal_trace,
)
from mlflow.pyfunc.context import Context, set_prediction_context
from mlflow.tracing.constant import AssessmentMetadataKey, TraceTagKey
from mlflow.tracing.utils.copy import copy_trace_to_experiment
from mlflow.tracking.client import MlflowClient
from mlflow.utils.mlflow_tags import IMMUTABLE_TAGS

_logger = logging.getLogger(__name__)


def _log_multi_turn_assessments_to_traces(
    multi_turn_assessments: dict[str, list[Feedback]],
    eval_results: list[EvalResult],
    run_id: str,
) -> None:
    """
    Log multi-turn assessments to the first trace of each session.

    Args:
        multi_turn_assessments: Dictionary mapping trace_id to list of assessments
        eval_results: List of EvalResult objects to update with multi-turn assessments
        run_id: MLflow run ID for logging
    """
    for eval_result in eval_results:
        if eval_result.eval_item.trace is None:
            continue

        trace_id = eval_result.eval_item.trace.info.trace_id
        if trace_id not in multi_turn_assessments:
            continue

        assessments_list = multi_turn_assessments[trace_id]
        try:
            _log_assessments(
                run_id=run_id,
                trace=eval_result.eval_item.trace,
                assessments=assessments_list,
            )
            eval_result.assessments.extend(assessments_list)
        except Exception as e:
            _logger.warning(f"Failed to log multi-turn assessments for trace {trace_id}: {e}")


def _parse_rate_limit(raw: str | None) -> tuple[float | None, bool]:
    """Parse a rate-limit env var into (rps_or_none, adaptive).

    Returns:
        (None, False)          when rate limiting is disabled ("0" or None)
        (rps, True)            when "auto"
        (rps, False)           when a fixed numeric value
    """
    auto_initial_rps = 10.0
    if raw is None:
        return None, False
    if raw.strip().lower() == "auto":
        return auto_initial_rps, True
    rate = float(raw)
    if rate <= 0:
        return None, False
    return rate, False


def _make_rate_limiter(
    rps: float | None, adaptive: bool = False, max_rps_multiplier: float = 5.0
) -> RateLimiter:
    if rps is None or rps <= 0:
        return NoOpRateLimiter()
    return RPSRateLimiter(rps, adaptive=adaptive, max_rps_multiplier=max_rps_multiplier)


def _pool_size(rps: float | None) -> int:
    """Derive thread count from rate limit, capped at [10, 500].

    Assumes each LLM call takes about ``_AVG_LLM_LATENCY_SECS`` seconds on
    average, so we need ``rps * latency`` threads to keep the pipeline busy.
    The rate limiter handles queueing — threads that can't get a token just
    block in acquire(). The HTTP connection pool is auto-sized to match.
    """
    avg_llm_latency_secs = 2
    if not rps:
        return 10
    return min(500, max(10, int(rps * avg_llm_latency_secs)))


def backpressure_buffer(score_workers: int) -> int:
    """Max items that may be predicted but not yet scored, bounding memory usage."""
    backpressure_multiplier = 2
    return backpressure_multiplier * score_workers


@context.eval_context
def run(
    *,
    eval_df: pd.DataFrame,
    predict_fn=None,
    scorers=None,
    run_id: str | None = None,
) -> EvaluationResult:
    """
    Runs GenAI evaluation harness to the given dataset.

    The overall flow is:

    1. Convert the dataset to a list of EvalItem objects
    2. Classify scorers into single-turn and multi-turn
    3. Run prediction and scoring in a pipelined fashion:
        a. Submit all predict tasks to a predict thread pool
        b. As each predict completes, submit its score task to a score thread pool
        c. As each score completes, collect the result
    4. If multi-turn scorers exist, evaluate them on session groups
    5. Compute the aggregated metrics from the assessments.

    Rate limiting is controlled via environment variables:
    - MLFLOW_GENAI_EVAL_PREDICT_RATE_LIMIT: max predict_fn calls/second
    - MLFLOW_GENAI_EVAL_SCORER_RATE_LIMIT: max scorer calls/second
    """
    eval_items = [EvalItem.from_dataset_row(row) for row in eval_df.to_dict(orient="records")]
    eval_start_time = int(time.time() * 1000)

    run_id = context.get_context().get_mlflow_run_id() if run_id is None else run_id

    # Classify scorers into single-turn and multi-turn
    single_turn_scorers, multi_turn_scorers = classify_scorers(scorers)

    session_groups = group_traces_by_session(eval_items) if multi_turn_scorers else {}

    total_tasks = len(eval_items) + len(session_groups)

    # Create rate limiters from environment variables.
    # Scorer rate auto-derives from predict rate * num_scorers when not set explicitly.
    predict_rps, predict_adaptive = _parse_rate_limit(MLFLOW_GENAI_EVAL_PREDICT_RATE_LIMIT.get())
    if MLFLOW_GENAI_EVAL_SCORER_RATE_LIMIT.is_set():
        scorer_rps, scorer_adaptive = _parse_rate_limit(MLFLOW_GENAI_EVAL_SCORER_RATE_LIMIT.get())
    else:
        num_scorers = len(single_turn_scorers) + len(multi_turn_scorers)
        scorer_rps = (predict_rps * num_scorers) if predict_rps and num_scorers else predict_rps
        scorer_adaptive = predict_adaptive
    upper_multiplier = MLFLOW_GENAI_EVAL_RATE_LIMIT_UPPER_MULTIPLIER.get()
    predict_limiter = _make_rate_limiter(
        predict_rps, adaptive=predict_adaptive, max_rps_multiplier=upper_multiplier
    )
    scorer_limiter = _make_rate_limiter(
        scorer_rps, adaptive=scorer_adaptive, max_rps_multiplier=upper_multiplier
    )
    max_retries = MLFLOW_GENAI_EVAL_MAX_RETRIES.get()

    if MLFLOW_GENAI_EVAL_MAX_WORKERS.is_set():
        predict_workers = score_workers = MLFLOW_GENAI_EVAL_MAX_WORKERS.get()
    else:
        predict_workers = _pool_size(predict_rps)
        score_workers = _pool_size(scorer_rps)

    predict_pool = ThreadPoolExecutor(
        max_workers=predict_workers,
        thread_name_prefix="MlflowGenAIEvalPredict",
    )
    score_pool = ThreadPoolExecutor(
        max_workers=score_workers,
        thread_name_prefix="MlflowGenAIEvalScore",
    )
    # Backpressure: cap predicted-but-not-yet-scored items to bound memory usage.
    in_flight = threading.Semaphore(backpressure_buffer(score_workers))

    progress_bar = (
        tqdm(
            total=total_tasks,
            desc="Evaluating",
            smoothing=0,
            bar_format=PGBAR_FORMAT,
        )
        if tqdm is not None
        else None
    )

    eval_results = [None] * len(eval_items)
    multi_turn_assessments = {}

    # Thread-time accumulators for predict vs score split.
    # Lists are used instead of nonlocal floats for safe mutation from threads.
    predict_times = []
    score_times = []
    _time_lock = threading.Lock()

    def _timed_predict(*args):
        start = time.monotonic()
        result = _run_predict(*args)
        with _time_lock:
            predict_times.append(time.monotonic() - start)
        return result

    def _timed_score(*args):
        start = time.monotonic()
        result = _run_score(*args)
        with _time_lock:
            score_times.append(time.monotonic() - start)
        return result

    # Sentinel signaling that the submit thread is done (or failed).
    _SUBMIT_DONE = None

    try:
        # Queue for the submit thread to pass (future, index) pairs to the main loop.
        predict_queue: queue.Queue[tuple[Future, int] | None] = queue.Queue()
        score_futures = {}  # future → item_index

        # Submit predict tasks with backpressure: the semaphore blocks when
        # too many items are predicted but not yet scored, bounding memory.
        submit_error = None

        def _submit_predicts():
            nonlocal submit_error
            try:
                for i, eval_item in enumerate(eval_items):
                    _logger.debug(f"Submit thread: waiting for backpressure slot (item {i})")
                    in_flight.acquire()
                    _logger.debug(f"Submit thread: submitting predict for item {i}")
                    future = predict_pool.submit(
                        _timed_predict,
                        eval_item,
                        predict_fn,
                        run_id,
                        predict_limiter,
                        max_retries,
                    )
                    predict_queue.put((future, i))
            except Exception as e:
                submit_error = e
            finally:
                predict_queue.put(_SUBMIT_DONE)

        submit_thread = threading.Thread(
            target=_submit_predicts, daemon=True, name="MlflowGenAIEvalSubmit"
        )
        submit_thread.start()

        # Pipeline loop: handles both predict and score completions.
        predict_futures = {}  # future → item_index
        items_scored = 0
        items_predicted = 0
        pending = set()
        last_heartbeat = time.monotonic()

        while items_scored < len(eval_items):
            # Drain newly submitted predict futures from the queue.
            # When pending is empty, block on the queue to avoid busy-waiting.
            while True:
                try:
                    if pending:
                        item = predict_queue.get_nowait()
                    else:
                        item = predict_queue.get(timeout=0.01)
                except queue.Empty:
                    break
                if item is _SUBMIT_DONE:
                    break
                future, idx = item
                predict_futures[future] = idx
                pending.add(future)

            if submit_error:
                raise submit_error

            if not pending:
                continue

            now = time.monotonic()
            if now - last_heartbeat >= 15:
                last_heartbeat = now
                predict_rps_str = (
                    f"{predict_limiter.current_rps:.1f}"
                    if predict_limiter.current_rps is not None
                    else "off"
                )
                scorer_rps_str = (
                    f"{scorer_limiter.current_rps:.1f}"
                    if scorer_limiter.current_rps is not None
                    else "off"
                )
                msg = (
                    f"[heartbeat] predicted={items_predicted}/{len(eval_items)}, "
                    f"scored={items_scored}/{len(eval_items)}, "
                    f"pending={len(pending)} "
                    f"({len(predict_futures)} predict, {len(score_futures)} score), "
                    f"rate: predict={predict_rps_str} rps, score={scorer_rps_str} rps"
                )
                _logger.info(msg)
                if progress_bar:
                    progress_bar.write(msg)

            _logger.debug(
                f"Pipeline loop: waiting on {len(pending)} pending futures "
                f"({len(predict_futures)} predict, {len(score_futures)} score), "
                f"scored={items_scored}/{len(eval_items)}"
            )
            done, pending = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
            if not done:
                continue
            for future in done:
                if future in predict_futures:
                    # Predict completed → submit score task
                    idx = predict_futures.pop(future)
                    items_predicted += 1
                    _logger.debug(f"Predict completed for item {idx}, submitting score")
                    future.result()  # propagate exceptions
                    score_future = score_pool.submit(
                        _timed_score,
                        eval_items[idx],
                        single_turn_scorers,
                        run_id,
                        scorer_limiter,
                        max_retries,
                    )
                    score_futures[score_future] = idx
                    pending.add(score_future)
                else:
                    # Score completed → item is fully done
                    idx = score_futures.pop(future)
                    _logger.debug(f"Score completed for item {idx}")
                    eval_results[idx] = future.result()
                    in_flight.release()
                    items_scored += 1
                    if progress_bar:
                        progress_bar.update(1)

        submit_thread.join()

        # Phase 2: Submit and complete multi-turn tasks (after single-turn)
        # We run multi-turn scorers after single-turn, since single-turn scorers may create new
        # traces that are needed by multi-turn scorers.
        if multi_turn_scorers and session_groups:
            multi_turn_futures = [
                score_pool.submit(
                    evaluate_session_level_scorers,
                    session_id=session_id,
                    session_items=session_items,
                    multi_turn_scorers=multi_turn_scorers,
                    scorer_rate_limiter=scorer_limiter,
                    max_retries=max_retries,
                )
                for session_id, session_items in session_groups.items()
            ]

            for future in as_completed(multi_turn_futures):
                session_result = future.result()
                multi_turn_assessments.update(session_result)
                if progress_bar:
                    progress_bar.update(1)
    finally:
        predict_pool.shutdown(wait=False, cancel_futures=True)
        score_pool.shutdown(wait=False, cancel_futures=True)

        predict_total = sum(predict_times)
        score_total = sum(score_times)
        total_thread_time = predict_total + score_total
        if progress_bar:
            if total_thread_time > 0:
                predict_pct = predict_total / total_thread_time * 100
                score_pct = score_total / total_thread_time * 100
                split = f" [predict_fn: {predict_pct:.0f}%, scorers: {score_pct:.0f}%]"
                progress_bar.bar_format = PGBAR_FORMAT + split
                progress_bar.refresh()
            progress_bar.close()

    if multi_turn_assessments:
        _log_multi_turn_assessments_to_traces(
            multi_turn_assessments=multi_turn_assessments,
            eval_results=eval_results,
            run_id=run_id,
        )

    # Link traces to the run if the backend support it
    batch_link_traces_to_run(run_id=run_id, eval_results=eval_results)

    # Refresh traces on eval_results to include all logged assessments.
    # This is done once after all assessments (single-turn and multi-turn) are logged to the traces.
    _refresh_eval_result_traces(eval_results)

    # Aggregate metrics and log to MLflow run
    aggregated_metrics = compute_aggregated_metrics(eval_results, scorers=scorers)
    mlflow.log_metrics(aggregated_metrics)

    try:
        emit_metric_usage_event(scorers, len(eval_items), len(session_groups), aggregated_metrics)
    except Exception as e:
        _logger.debug(f"Failed to emit metric usage event: {e}", exc_info=True)

    # Search for all traces in the run. We need to fetch the traces from backend here to include
    # all traces in the result.
    traces = mlflow.search_traces(run_id=run_id, include_spans=False, return_type="list")

    # Collect trace IDs from eval results to preserve them during cleanup.
    input_trace_ids = {
        result.eval_item.trace.info.trace_id
        for result in eval_results
        if result.eval_item.trace is not None
    }

    # Clean up noisy traces generated during evaluation
    clean_up_extra_traces(traces, eval_start_time, input_trace_ids)

    return EvaluationResult(
        run_id=run_id,
        result_df=construct_eval_result_df(run_id, traces, eval_results),
        metrics=aggregated_metrics,
    )


def _run_predict(
    eval_item: EvalItem,
    predict_fn: Callable[..., Any] | None,
    run_id: str | None,
    rate_limiter: RateLimiter,
    max_retries: int = 0,
) -> None:
    if run_id:
        ctx = context.get_context()
        ctx.set_mlflow_run_id(run_id)

    if predict_fn:
        # NB: Setting prediction context let us retrieve the trace by a custom ID. Setting
        # is_evaluate=True disables async trace logging to make sure the trace is available.
        eval_request_id = str(uuid.uuid4())
        with (
            set_prediction_context(Context(request_id=eval_request_id, is_evaluate=True)),
            eval_retry_context(),
        ):
            try:
                eval_item.outputs = call_with_retry(
                    lambda: predict_fn(eval_item.inputs),
                    rate_limiter,
                    max_retries,
                )
            except Exception as e:
                eval_item.error_message = (
                    f"Failed to invoke the predict_fn with {eval_item.inputs}: {e}"
                )

        eval_item.trace = mlflow.get_trace(eval_request_id, silent=True)
    elif eval_item.trace is not None:
        if _should_clone_trace(eval_item.trace, run_id):
            try:
                trace_id = copy_trace_to_experiment(eval_item.trace.to_dict())
                eval_item.trace = mlflow.get_trace(trace_id)
            except Exception as e:
                eval_item.error_message = f"Failed to clone trace to the current experiment: {e}"
        else:
            MlflowClient().link_traces_to_run([eval_item.trace.info.trace_id], run_id=run_id)
    else:
        # When static dataset (a pair of inputs and outputs) is given, we create a minimal
        # trace with root span only, to log the assessments on it.
        eval_item.trace = create_minimal_trace(eval_item)


def _run_score(
    eval_item: EvalItem,
    scorers: list[Scorer],
    run_id: str | None,
    scorer_rate_limiter: RateLimiter,
    max_retries: int = 0,
) -> EvalResult:
    if run_id:
        ctx = context.get_context()
        ctx.set_mlflow_run_id(run_id)

    with eval_retry_context():
        assessments = _compute_eval_scores(
            eval_item=eval_item,
            scorers=scorers,
            rate_limiter=scorer_rate_limiter,
            max_retries=max_retries,
        )
    assessments.extend(_get_new_expectations(eval_item))
    eval_result = EvalResult(eval_item=eval_item, assessments=assessments)

    tags = eval_item.tags if not is_none_or_nan(eval_item.tags) else {}
    validate_tags(tags)

    for key in tags.keys() - IMMUTABLE_TAGS:
        try:
            mlflow.set_trace_tag(trace_id=eval_item.trace.info.trace_id, key=key, value=tags[key])
        except Exception as e:
            _logger.warning(f"Failed to log tag {key} to MLflow: {e}")

    try:
        _log_assessments(
            run_id=run_id,
            trace=eval_item.trace,
            assessments=eval_result.assessments,
        )
    except Exception as e:
        _logger.warning(f"Failed to log trace and assessments to MLflow: {e}")

    return eval_result


def _compute_eval_scores(
    *,
    eval_item: EvalItem,
    scorers: list[Scorer],
    rate_limiter: RateLimiter = NoOpRateLimiter(),
    max_retries: int = 0,
) -> list[Feedback]:
    if not scorers:
        return []

    should_trace = MLFLOW_GENAI_EVAL_ENABLE_SCORER_TRACING.get()

    def run_scorer(scorer):
        try:
            scorer_func = scorer.run

            if should_trace:
                scorer_func = mlflow.trace(name=scorer.name, span_type=SpanType.EVALUATOR)(
                    scorer_func
                )

            def _invoke():
                return scorer_func(
                    inputs=eval_item.inputs,
                    outputs=eval_item.outputs,
                    expectations=eval_item.expectations,
                    trace=eval_item.trace,
                )

            value = call_with_retry(_invoke, rate_limiter, max_retries)
            feedbacks = standardize_scorer_value(scorer.name, value)

        except Exception as e:
            feedbacks = [
                Feedback(
                    name=scorer.name,
                    source=make_code_type_assessment_source(scorer.name),
                    error=AssessmentError(
                        error_code="SCORER_ERROR",
                        error_message=str(e),
                        stack_trace=traceback.format_exc(),
                    ),
                )
            ]

        # Record the trace ID for the scorer function call.
        if should_trace and (trace_id := mlflow.get_last_active_trace_id(thread_local=True)):
            for feedback in feedbacks:
                feedback.metadata = {
                    **(feedback.metadata or {}),
                    AssessmentMetadataKey.SCORER_TRACE_ID: trace_id,
                }
            # Set the scorer name tag to the trace to identify the trace is generated by a scorer.
            mlflow.set_trace_tag(
                trace_id=trace_id,
                key=TraceTagKey.SOURCE_SCORER_NAME,
                value=scorer.name,
            )
        return feedbacks

    # Use a thread pool to run scorers in parallel
    # Limit concurrent scorers to prevent rate limiting errors with external LLM APIs
    max_scorer_workers = min(len(scorers), MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS.get())
    with ThreadPoolExecutor(
        max_workers=max_scorer_workers,
        thread_name_prefix="MlflowGenAIEvalScorer",
    ) as executor:
        futures = [executor.submit(run_scorer, scorer) for scorer in scorers]

        try:
            results = [future.result() for future in as_completed(futures)]
        except KeyboardInterrupt:
            # Cancel pending futures
            executor.shutdown(cancel_futures=True)
            raise

    # Flatten list[list[Assessment]] into a single list[Assessment]
    return [assessment for sublist in results for assessment in sublist]


def _get_new_expectations(eval_item: EvalItem) -> list[Expectation]:
    existing_expectations = {
        a.name for a in eval_item.trace.info.assessments if a.expectation is not None
    }
    return [
        exp
        for exp in eval_item.get_expectation_assessments()
        if exp.name not in existing_expectations
    ]


def _log_assessments(
    run_id: str | None,
    trace: Trace,
    assessments: list[Assessment],
) -> None:
    """
    Log assessments to a trace.
    """
    for assessment in assessments:
        # Ensure that if we created a new trace, that the updated trace_id is reflected in
        # the assessments.
        assessment.trace_id = trace.info.trace_id
        if run_id is not None:
            assessment.metadata = {
                **(assessment.metadata or {}),
                AssessmentMetadataKey.SOURCE_RUN_ID: run_id,
            }

        # NB: Root span ID is necessarily to show assessment results in DBX eval UI.
        if root_span := trace.data._get_root_span():
            assessment.span_id = root_span.span_id
        else:
            _logger.debug(f"No root span found for trace {trace.info.trace_id}")

        mlflow.log_assessment(trace_id=assessment.trace_id, assessment=assessment)


def _refresh_eval_result_traces(eval_results: list[EvalResult]) -> None:
    """
    Refresh traces on eval_results to include logged assessments.

    This function fetches the updated traces from the backend after all assessments
    (both single-turn and multi-turn) have been logged.
    """

    def _fetch_trace(eval_result: EvalResult):
        if eval_result.eval_item.trace is None:
            return None
        trace_id = eval_result.eval_item.trace.info.trace_id
        try:
            return eval_result, mlflow.get_trace(trace_id)
        except Exception as e:
            _logger.warning(f"Failed to refresh trace {trace_id}: {e}")
            return None

    with ThreadPoolExecutor(
        max_workers=MLFLOW_GENAI_EVAL_MAX_WORKERS.get(),
        thread_name_prefix="GenAIEvaluationTraceRefresh",
    ) as executor:
        futures = [executor.submit(_fetch_trace, er) for er in eval_results]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                eval_result, refreshed_trace = result
                eval_result.eval_item.trace = refreshed_trace


def _should_clone_trace(trace: Trace | None, run_id: str | None) -> bool:
    from mlflow.tracking.fluent import _get_experiment_id

    if trace is None:
        return False

    # If the trace is stored in UC table, we don't clone the trace
    if trace.info.trace_location.uc_schema is not None:
        return False

    # Check if the trace is from the same experiment. If it isn't, we need to clone the trace
    trace_experiment = trace.info.trace_location.mlflow_experiment
    current_experiment = _get_experiment_id()
    if trace_experiment is not None and trace_experiment.experiment_id != current_experiment:
        return True

    # If the backend doesn't support trace<->run linking, need to clone the trace to the new run.
    return not _does_store_support_trace_linking(
        tracking_uri=mlflow.get_tracking_uri(),
        trace=trace,
        run_id=run_id,
    )
