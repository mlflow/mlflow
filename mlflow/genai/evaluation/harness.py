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


def _warmup_databricks_sdk() -> None:
    """Import databricks.sdk in the main thread to avoid import-lock deadlocks in workers."""
    try:
        import databricks.sdk  # noqa: F401
    except ImportError:
        pass


import mlflow
from mlflow.entities import SpanType
from mlflow.entities.assessment import Assessment, Expectation, Feedback
from mlflow.entities.assessment_error import AssessmentError
from mlflow.entities.trace import Trace
from mlflow.environment_variables import (
    MLFLOW_GENAI_EVAL_ENABLE_HEARTBEAT,
    MLFLOW_GENAI_EVAL_ENABLE_SCORER_TRACING,
    MLFLOW_GENAI_EVAL_MAX_RETRIES,
    MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS,
    MLFLOW_GENAI_EVAL_MAX_WORKERS,
    MLFLOW_GENAI_EVAL_PREDICT_RATE_LIMIT,
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
    """Log multi-turn assessments to traces in parallel."""

    def _log_for_result(eval_result: EvalResult) -> None:
        if eval_result.eval_item.trace is None:
            return
        trace_id = eval_result.eval_item.trace.info.trace_id
        if trace_id not in multi_turn_assessments:
            return
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

    with ThreadPoolExecutor(
        max_workers=MLFLOW_GENAI_EVAL_MAX_WORKERS.get(),
    ) as executor:
        futures = [executor.submit(_log_for_result, er) for er in eval_results]
        for future in as_completed(futures):
            future.result()


AUTO_INITIAL_RPS = 10.0


def _parse_rate_limit(raw: str | None) -> tuple[float | None, bool]:
    """Parse a rate-limit env var into (rps_or_none, adaptive).

    Returns:
        (None, False)          when rate limiting is disabled ("0" or None)
        (rps, True)            when "auto"
        (rps, False)           when a fixed numeric value
    """
    if raw is None:
        return None, False
    if raw.strip().lower() == "auto":
        return AUTO_INITIAL_RPS, True
    rate = float(raw)
    if rate <= 0:
        return None, False
    return rate, False


_AIMD_UPPER_MULTIPLIER = 2.0


def _make_rate_limiter(
    rps: float | None, adaptive: bool = False, max_rps_multiplier: float = _AIMD_UPPER_MULTIPLIER
) -> RateLimiter:
    if rps is None or rps <= 0:
        return NoOpRateLimiter()
    return RPSRateLimiter(rps, adaptive=adaptive, max_rps_multiplier=max_rps_multiplier)


def _pool_size(rps: float | None, max_rps_multiplier: float = 1.0) -> int:
    """Derive thread count from rate limit, capped at [10, 500].

    Assumes each LLM call takes about ``avg_llm_latency_secs`` seconds on
    average, so we need ``peak_rps * latency`` threads to keep the pipeline
    busy at the AIMD ceiling. The rate limiter handles queueing — threads
    that can't get a token just block in acquire().
    """
    avg_llm_latency_secs = 2
    if not rps:
        return 10
    peak_rps = rps * max_rps_multiplier
    return min(500, max(10, int(peak_rps * avg_llm_latency_secs)))


def backpressure_buffer(score_workers: int) -> int:
    """Max items that may be predicted but not yet scored, bounding memory usage."""
    backpressure_multiplier = 2
    return backpressure_multiplier * score_workers


def _get_scorer_rate_config(
    predict_rps: float | None,
    predict_adaptive: bool,
    num_scorers: int,
) -> tuple[float | None, bool]:
    """Derive scorer rate limit from env var or predict rate.

    When MLFLOW_GENAI_EVAL_SCORER_RATE_LIMIT is explicitly set, parse it.
    Otherwise auto-derive as predict_rps * num_scorers.
    """
    if MLFLOW_GENAI_EVAL_SCORER_RATE_LIMIT.is_set():
        return _parse_rate_limit(MLFLOW_GENAI_EVAL_SCORER_RATE_LIMIT.get())
    scorer_rps = (predict_rps * num_scorers) if predict_rps and num_scorers else predict_rps
    return scorer_rps, predict_adaptive


def _get_pool_sizes(
    predict_rps: float | None,
    scorer_rps: float | None,
    max_rps_multiplier: float = 1.0,
) -> tuple[int, int]:
    """Determine predict and score thread pool sizes.

    Uses MLFLOW_GENAI_EVAL_MAX_WORKERS as an override when set, otherwise
    derives independently from each rate limit.
    """
    if MLFLOW_GENAI_EVAL_MAX_WORKERS.is_set():
        size = MLFLOW_GENAI_EVAL_MAX_WORKERS.get()
        return size, size
    return (
        _pool_size(predict_rps, max_rps_multiplier),
        _pool_size(scorer_rps, max_rps_multiplier),
    )


class _Heartbeat:
    """Periodic debug-level heartbeat for the pipeline loop.

    Only active when ``MLFLOW_GENAI_EVAL_ENABLE_HEARTBEAT`` is set to True.
    """

    def __init__(
        self,
        predictor: "_PredictSubmitter",
        scorer: "_ScoreSubmitter",
        total_items: int,
        interval_secs: float = 15,
    ):
        self._enabled = MLFLOW_GENAI_EVAL_ENABLE_HEARTBEAT.get()
        self._predictor = predictor
        self._scorer = scorer
        self._total = total_items
        self._interval = interval_secs
        self._last_time = time.monotonic()

    @property
    def interval(self) -> float:
        return self._interval

    @staticmethod
    def _format_rps(limiter: RateLimiter) -> str:
        rps = limiter.current_rps
        return f"{rps:.1f}" if rps is not None else "off"

    def tick(self, items_predicted: int, items_scored: int) -> None:
        if not self._enabled:
            return
        now = time.monotonic()
        if now - self._last_time < self._interval:
            return
        self._last_time = now
        _logger.debug(
            "[heartbeat] predicted=%d/%d, scored=%d/%d, "
            "pending: %d predict, %d score, rate: predict=%s rps, score=%s rps",
            items_predicted,
            self._total,
            items_scored,
            self._total,
            self._predictor.pending_count,
            self._scorer.pending_count,
            self._format_rps(self._predictor.limiter),
            self._format_rps(self._scorer.limiter),
        )


class _PredictSubmitter:
    """Owns the submit thread, predict pool, backpressure semaphore, and predict timing."""

    def __init__(
        self,
        eval_items: list[EvalItem],
        predict_fn: Callable[..., Any] | None,
        run_id: str | None,
        max_retries: int,
        rps: float | None,
        adaptive: bool,
        max_rps_multiplier: float,
        pool_workers: int,
        score_workers: int,
    ):
        """
        Args:
            eval_items: Items to evaluate — each will be submitted for prediction.
            predict_fn: User-provided function that produces outputs from inputs.
                When None, predictions are skipped and existing traces are used.
            run_id: MLflow run ID for trace/assessment logging.
            max_retries: Max 429-retry attempts per predict call.
            rps: Requests-per-second for the predict rate limiter, or None to disable.
            adaptive: Whether the rate limiter uses AIMD to adapt to 429 signals.
            max_rps_multiplier: AIMD ceiling as a multiple of the initial rps.
            pool_workers: Number of threads in the predict pool.
            score_workers: Number of score-pool threads, used to size the
                backpressure buffer that bounds predicted-but-not-yet-scored items.
        """
        self._eval_items = eval_items
        self._predict_fn = predict_fn
        self._run_id = run_id
        self._max_retries = max_retries

        self._limiter = _make_rate_limiter(
            rps, adaptive=adaptive, max_rps_multiplier=max_rps_multiplier
        )
        self._pool = ThreadPoolExecutor(
            max_workers=pool_workers, thread_name_prefix="MlflowGenAIEvalPredict"
        )
        self._in_flight = threading.Semaphore(backpressure_buffer(score_workers))
        self._queue: queue.Queue[tuple[Future, int] | None] = queue.Queue()
        self._predict_futures_to_eval_id: dict[Future, int] = {}
        self._times: list[float] = []
        self._time_lock = threading.Lock()
        self._submit_error: Exception | None = None
        self._thread: threading.Thread | None = None

    @property
    def limiter(self) -> RateLimiter:
        return self._limiter

    @property
    def predict_times(self) -> list[float]:
        return self._times

    @property
    def pending_count(self) -> int:
        return len(self._predict_futures_to_eval_id)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def start(self) -> None:
        """Submit all eval items to the predict pool from a background thread.

        Each submission blocks on the backpressure semaphore to bound the number
        of predicted-but-not-yet-scored items. Completed futures are placed on
        ``self._queue`` for the main loop to drain. A None sentinel signals
        that all items have been submitted (or an error occurred).
        """

        self._thread = threading.Thread(
            target=self._submit_all, daemon=True, name="MlflowGenAIEvalSubmit"
        )
        self._thread.start()

    def join(self) -> None:
        if self._thread is not None:
            self._thread.join()

    def _submit_all(self) -> None:
        try:
            for i, eval_item in enumerate(self._eval_items):
                _logger.debug(f"Submit thread: waiting for backpressure slot (item {i})")
                self._in_flight.acquire()
                _logger.debug(f"Submit thread: submitting predict for item {i}")
                future = self._pool.submit(
                    self._timed_predict,
                    eval_item,
                    self._predict_fn,
                    self._run_id,
                    self._limiter,
                    self._max_retries,
                )
                self._queue.put((future, i))
        except Exception as e:
            self._submit_error = e
        finally:
            self._queue.put(None)  # sentinel

    def _timed_predict(self, *args) -> None:
        start = time.monotonic()
        _run_predict(*args)
        with self._time_lock:
            self._times.append(time.monotonic() - start)

    def drain(self, *, block: bool = False) -> list[Future]:
        """Return newly submitted predict futures from the submit thread.

        When *block* is True, waits briefly (10 ms) for new items so the main
        loop doesn't busy-wait when there is no other pending work.
        """
        drained: list[Future] = []
        while True:
            try:
                item = self._queue.get_nowait() if not block else self._queue.get(timeout=0.01)
            except queue.Empty:
                break
            if item is None:
                break
            block = False  # only block on the first get
            future, idx = item
            self._predict_futures_to_eval_id[future] = idx
            drained.append(future)
        return drained

    def check_error(self) -> None:
        if self._submit_error:
            raise self._submit_error

    def owns(self, future: Future) -> bool:
        return future in self._predict_futures_to_eval_id

    def on_complete(self, future: Future) -> int:
        """Finalize a completed predict future: propagate exceptions and return its item index."""
        idx = self._predict_futures_to_eval_id.pop(future)
        future.result()  # propagate exceptions
        return idx

    def release_slot(self) -> None:
        """Release one backpressure slot, allowing the submit thread to enqueue another predict."""
        self._in_flight.release()


class _ScoreSubmitter:
    """Owns the score pool, score futures, multi-turn scoring, and score timing."""

    def __init__(
        self,
        eval_items: list[EvalItem],
        single_turn_scorers: list[Scorer],
        multi_turn_scorers: list[Scorer],
        session_groups: dict[str, list[EvalItem]],
        run_id: str | None,
        max_retries: int,
        rps: float | None,
        adaptive: bool,
        max_rps_multiplier: float,
        pool_workers: int,
    ):
        """
        Args:
            eval_items: Items to evaluate — indexed by position for score dispatch.
            single_turn_scorers: Scorers applied to each item individually.
            multi_turn_scorers: Scorers applied to session groups after the
                single-turn pipeline completes.
            session_groups: Mapping of session_id to ordered list of eval items
                for multi-turn scoring.
            run_id: MLflow run ID for trace/assessment logging.
            max_retries: Max 429-retry attempts per scorer call.
            rps: Requests-per-second for the scorer rate limiter, or None to disable.
            adaptive: Whether the rate limiter uses AIMD to adapt to 429 signals.
            max_rps_multiplier: AIMD ceiling as a multiple of the initial rps.
            pool_workers: Number of threads in the score pool.
        """
        self._eval_items = eval_items
        self._single_turn_scorers = single_turn_scorers
        self._multi_turn_scorers = multi_turn_scorers
        self._session_groups = session_groups
        self._run_id = run_id
        self._max_retries = max_retries

        self._limiter = _make_rate_limiter(
            rps, adaptive=adaptive, max_rps_multiplier=max_rps_multiplier
        )
        self._pool = ThreadPoolExecutor(
            max_workers=pool_workers, thread_name_prefix="MlflowGenAIEvalScore"
        )
        self._score_futures_to_eval_id: dict[Future, int] = {}
        self._times: list[float] = []
        self._time_lock = threading.Lock()

    @property
    def limiter(self) -> RateLimiter:
        return self._limiter

    @property
    def score_times(self) -> list[float]:
        return self._times

    @property
    def pending_count(self) -> int:
        return len(self._score_futures_to_eval_id)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def submit(self, idx: int) -> Future:
        """Submit a score task for eval item *idx* and return the future."""
        _logger.debug(f"Predict completed for item {idx}, submitting score")
        future = self._pool.submit(
            self._timed_score,
            self._eval_items[idx],
            self._single_turn_scorers,
            self._run_id,
            self._limiter,
            self._max_retries,
        )
        self._score_futures_to_eval_id[future] = idx
        return future

    def _timed_score(self, *args) -> EvalResult:
        start = time.monotonic()
        result = _run_score(*args)
        with self._time_lock:
            self._times.append(time.monotonic() - start)
        return result

    def on_complete(self, future: Future) -> tuple[int, EvalResult]:
        idx = self._score_futures_to_eval_id.pop(future)
        return idx, future.result()

    def run_multi_turn(
        self, multi_turn_assessments: dict[str, list[Feedback]], progress_bar
    ) -> None:
        if not self._multi_turn_scorers or not self._session_groups:
            return
        futures = [
            self._pool.submit(
                evaluate_session_level_scorers,
                session_id=session_id,
                session_items=session_items,
                multi_turn_scorers=self._multi_turn_scorers,
                scorer_rate_limiter=self._limiter,
                max_retries=self._max_retries,
            )
            for session_id, session_items in self._session_groups.items()
        ]
        for future in as_completed(futures):
            multi_turn_assessments.update(future.result())
            if progress_bar:
                progress_bar.update(1)


def _run_pipeline(
    eval_items: list[EvalItem],
    eval_results: list[EvalResult | None],
    predict_fn: Callable[..., Any] | None,
    single_turn_scorers: list[Scorer],
    multi_turn_scorers: list[Scorer],
    session_groups: dict[str, list[EvalItem]],
    run_id: str | None,
    progress_bar,
    multi_turn_assessments: dict[str, list[Feedback]],
) -> tuple[list[float], list[float]]:
    """Run the predict→score pipeline and multi-turn scoring.

    Creates rate limiters and thread pools from environment variables,
    runs the pipelined predict→score loop, then multi-turn scoring.
    Returns (predict_times, score_times) for reporting.
    """
    _warmup_databricks_sdk()

    predict_rps, predict_adaptive = _parse_rate_limit(MLFLOW_GENAI_EVAL_PREDICT_RATE_LIMIT.get())
    num_scorers = len(single_turn_scorers) + len(multi_turn_scorers)
    scorer_rps, scorer_adaptive = _get_scorer_rate_config(
        predict_rps, predict_adaptive, num_scorers
    )
    max_retries = MLFLOW_GENAI_EVAL_MAX_RETRIES.get()
    pool_multiplier = _AIMD_UPPER_MULTIPLIER if predict_adaptive else 1.0
    predict_workers, score_workers = _get_pool_sizes(predict_rps, scorer_rps, pool_multiplier)

    predictor = _PredictSubmitter(
        eval_items,
        predict_fn,
        run_id,
        max_retries,
        rps=predict_rps,
        adaptive=predict_adaptive,
        max_rps_multiplier=_AIMD_UPPER_MULTIPLIER,
        pool_workers=predict_workers,
        score_workers=score_workers,
    )
    scorer = _ScoreSubmitter(
        eval_items,
        single_turn_scorers,
        multi_turn_scorers,
        session_groups,
        run_id,
        max_retries,
        rps=scorer_rps,
        adaptive=scorer_adaptive,
        max_rps_multiplier=_AIMD_UPPER_MULTIPLIER,
        pool_workers=score_workers,
    )

    try:
        if single_turn_scorers:
            heartbeat = _Heartbeat(predictor, scorer, len(eval_items))

            predictor.start()
            pending: set[Future] = set()
            items_predicted = 0
            items_scored = 0

            while items_scored < len(eval_items):
                pending.update(predictor.drain(block=not pending))
                predictor.check_error()
                if not pending:
                    continue

                heartbeat.tick(items_predicted, items_scored)

                # Timeout matches the heartbeat interval so the loop wakes periodically
                # even if no futures complete, allowing the heartbeat to fire.
                done, pending = wait(
                    pending, timeout=heartbeat.interval, return_when=FIRST_COMPLETED
                )
                for future in done:
                    if predictor.owns(future):
                        idx = predictor.on_complete(future)
                        items_predicted += 1
                        pending.add(scorer.submit(idx))
                    else:
                        idx, result = scorer.on_complete(future)
                        _logger.debug(f"Score completed for item {idx}")
                        eval_results[idx] = result
                        predictor.release_slot()
                        items_scored += 1
                        if progress_bar:
                            progress_bar.update(1)

            predictor.join()
        else:
            # No single-turn scorers — run predictions to set up traces
            # but skip the scoring pipeline and its progress ticks.
            predictor.start()
            pending: set[Future] = set()
            items_done = 0
            while items_done < len(eval_items):
                pending.update(predictor.drain(block=not pending))
                predictor.check_error()
                if not pending:
                    continue
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    idx = predictor.on_complete(future)
                    predictor.release_slot()
                    eval_results[idx] = EvalResult(eval_item=eval_items[idx], assessments=[])
                    items_done += 1
            predictor.join()

        # Multi-turn scorers run after single-turn scoring completes because they
        # operate on session groups and need fully scored traces.
        scorer.run_multi_turn(multi_turn_assessments, progress_bar)

        return predictor.predict_times, scorer.score_times
    finally:
        predictor.shutdown()
        scorer.shutdown()


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

    Rate limiting is controlled via environment variables:
    - MLFLOW_GENAI_EVAL_PREDICT_RATE_LIMIT: max predict_fn calls/second
    - MLFLOW_GENAI_EVAL_SCORER_RATE_LIMIT: max scorer calls/second
    """
    eval_items = [EvalItem.from_dataset_row(row) for row in eval_df.to_dict(orient="records")]
    eval_start_time = int(time.time() * 1000)

    run_id = context.get_context().get_mlflow_run_id() if run_id is None else run_id

    single_turn_scorers, multi_turn_scorers = classify_scorers(scorers)
    session_groups = group_traces_by_session(eval_items) if multi_turn_scorers else {}
    total_tasks = (len(eval_items) if single_turn_scorers else 0) + len(session_groups)

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
    predict_times: list[float] = []
    score_times: list[float] = []

    try:
        predict_times, score_times = _run_pipeline(
            eval_items=eval_items,
            eval_results=eval_results,
            predict_fn=predict_fn,
            single_turn_scorers=single_turn_scorers,
            multi_turn_scorers=multi_turn_scorers,
            session_groups=session_groups,
            run_id=run_id,
            progress_bar=progress_bar,
            multi_turn_assessments=multi_turn_assessments,
        )
    finally:
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


def _invoke_scorer(scorer_func: Callable, eval_item: EvalItem):
    return scorer_func(
        inputs=eval_item.inputs,
        outputs=eval_item.outputs,
        expectations=eval_item.expectations,
        trace=eval_item.trace,
    )


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

            value = call_with_retry(
                lambda: _invoke_scorer(scorer_func, eval_item), rate_limiter, max_retries
            )
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
