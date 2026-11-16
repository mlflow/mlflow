"""Online scoring processor for executing scorers on traces."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import mlflow
from mlflow.entities import AssessmentError, Trace, TraceData, TraceInfo
from mlflow.entities.trace_state import TraceState
from mlflow.environment_variables import MLFLOW_ONLINE_SCORING_MAX_WORKER_THREADS
from mlflow.genai.scorers.base import Scorer
from mlflow.genai.scorers.online.constants import (
    EXCLUDE_EVAL_RUN_TRACES_FILTER,
    MAX_JOB_RUNTIME_MS,
    MAX_SCORER_FAILURE_RETRY_MS,
    MAX_TRACE_COMPLETION_WAIT_MS,
    MAX_TRACES_PER_JOB,
)
from mlflow.genai.scorers.online.entities import OnlineScorer
from mlflow.genai.scorers.online.sampler import OnlineScorerSampler
from mlflow.genai.scorers.online.trace_checkpointer import (
    OnlineTraceCheckpointManager,
    OnlineTraceScoringCheckpoint,
    OnlineTraceScoringTimeWindow,
)
from mlflow.genai.scorers.online.trace_loader import OnlineTraceLoader, trace_end_time_ms
from mlflow.genai.scorers.online.watermark import (
    ConsideredEntity,
    EntityKey,
    compute_checkpoint,
    is_after,
    safe_watermark,
)
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.tracing.constant import AssessmentMetadataKey

_logger = logging.getLogger(__name__)

# Error code of the assessment recorded when online scoring gives up waiting for a trace.
TRACE_INCOMPLETE_ERROR_CODE = "TRACE_INCOMPLETE"


@dataclass
class TraceScoringTask:
    """A task to score a single trace with multiple scorers."""

    trace: Trace | None
    scorers: list[Scorer]
    # The trace's end time (start + duration): the key the checkpoint is ordered on. For an
    # in-progress trace it is the end of its latest exported span.
    timestamp_ms: int
    trace_info: TraceInfo | None = None


@dataclass
class _Discovery:
    """Everything the discovery phase learned about the current window."""

    # Every trace fetched in this run, keyed by trace ID, with whether it is durably done.
    # A trace that was sampled out, or whose selected scorers all already ran, is done.
    considered: dict[str, ConsideredEntity] = field(default_factory=dict)
    # Traces that still have scorers to run, keyed by trace ID.
    tasks: dict[str, TraceScoringTask] = field(default_factory=dict)
    # Last (timestamp_ms, trace_id) of every filter group whose page was full.
    full_page_last_keys: list[EntityKey] = field(default_factory=list)


class OnlineTraceScoringProcessor:
    """
    Orchestrates online scoring of individual traces.

    Traces are discovered and checkpointed by their *end* time (start + duration), not their start
    time. Traces are written to the store when they finish, so a start-time watermark would
    permanently skip any trace still running when the watermark passed its start; by end time a
    trace is always ahead of the watermark at the moment it lands, however long it ran.

    The checkpoint persisted at the end of each run is a durable watermark: it never moves past a
    trace that matched a scorer's filter unless that trace was durably assessed, was
    deterministically sampled out, or was still incomplete after ``MAX_TRACE_COMPLETION_WAIT_MS``
    without progress (in which case a ``TRACE_INCOMPLETE`` error assessment is recorded, and
    replaced by a real score if the trace completes later).
    Scorer failures are retried on later runs rather than recorded, until the trace is older
    than ``MAX_SCORER_FAILURE_RETRY_MS``; only then is an error feedback persisted. Re-running a
    window is idempotent: assessments written by online scoring carry the scorer name in their
    metadata, and scorers that already produced an assessment for a trace are not run again.
    """

    def __init__(
        self,
        trace_loader: OnlineTraceLoader,
        checkpoint_manager: OnlineTraceCheckpointManager,
        sampler: OnlineScorerSampler,
        experiment_id: str,
    ):
        self._trace_loader = trace_loader
        self._checkpoint_manager = checkpoint_manager
        self._sampler = sampler
        self._experiment_id = experiment_id

    @classmethod
    def create(
        cls,
        experiment_id: str,
        online_scorers: list[OnlineScorer],
        tracking_store: AbstractStore,
    ) -> "OnlineTraceScoringProcessor":
        """
        Factory method to create an OnlineTraceScoringProcessor with dependencies.

        Args:
            experiment_id: The experiment ID to process traces from.
            online_scorers: List of OnlineScorer instances.
            tracking_store: The tracking store instance.

        Returns:
            Configured OnlineTraceScoringProcessor instance.
        """
        return cls(
            trace_loader=OnlineTraceLoader(tracking_store),
            checkpoint_manager=OnlineTraceCheckpointManager(tracking_store, experiment_id),
            sampler=OnlineScorerSampler(online_scorers),
            experiment_id=experiment_id,
        )

    def process_traces(self) -> None:
        """
        Execute online scoring for the experiment.

        Runs scoring batches until the backlog is drained: each batch fetches traces since the
        last checkpoint, applies sampling to select scorers, runs scoring in parallel, and
        updates the checkpoint. Another batch follows while the previous batch's discovery page
        was full (more eligible traces exist), the checkpoint moved, and the job has been running
        for less than ``MAX_JOB_RUNTIME_MS``.
        """
        job_started_ms = int(time.time() * 1000)
        last_persisted: EntityKey | None = None
        while True:
            more_pending, persisted = self._process_batch()
            # Continue only if the checkpoint moved strictly beyond what this job last wrote;
            # this also guards against looping if the store does not reflect the write.
            advanced = persisted is not None and (
                last_persisted is None or is_after(persisted, last_persisted)
            )
            if not (more_pending and advanced):
                break
            last_persisted = persisted
            if int(time.time() * 1000) - job_started_ms >= MAX_JOB_RUNTIME_MS:
                _logger.info(
                    f"Online trace scoring for experiment {self._experiment_id} still has a "
                    "backlog after reaching the job runtime limit; the next job continues"
                )
                break

    def _process_batch(self) -> tuple[bool, EntityKey | None]:
        """
        Run one scoring batch.

        Returns:
            ``(more_pending, persisted)``: whether at least one discovery page was full (so more
            eligible traces may exist beyond this batch), and the checkpoint persisted by this batch
            (``None`` if it did not move).
        """
        time_window = self._checkpoint_manager.calculate_time_window()
        checkpoint = self._checkpoint_manager.get_checkpoint()
        now_ms = int(time.time() * 1000)

        _logger.debug(
            f"Online scoring for experiment {self._experiment_id}: "
            f"time window [{time_window.min_trace_timestamp_ms}, "
            f"{time_window.max_trace_timestamp_ms}]"
        )

        discovery = self._discover(time_window, checkpoint)
        watermark = safe_watermark(discovery.full_page_last_keys)
        tasks = self._select_tasks(discovery, watermark, now_ms)

        if tasks:
            _logger.debug(f"Running scoring: {len(tasks)} trace tasks")
            self._load_full_traces(tasks, discovery, now_ms)
            failed_trace_ids = self._execute_scoring(tasks, now_ms)
            for trace_id in failed_trace_ids:
                discovery.considered[trace_id].done = False
        else:
            _logger.debug("No traces selected after sampling, skipping")

        new_checkpoint = compute_checkpoint(
            previous=(checkpoint.timestamp_ms, checkpoint.trace_id) if checkpoint else None,
            window_max_ms=time_window.max_trace_timestamp_ms,
            considered=discovery.considered.values(),
            watermark=watermark,
        )
        if new_checkpoint is None:
            _logger.debug("Checkpoint not advanced: earliest pending trace is still outstanding")
        else:
            self._checkpoint_manager.persist_checkpoint(
                OnlineTraceScoringCheckpoint(
                    timestamp_ms=new_checkpoint[0], trace_id=new_checkpoint[1]
                )
            )

        _logger.debug(f"Online trace scoring batch completed for experiment {self._experiment_id}")
        return watermark is not None, new_checkpoint

    def _discover(
        self,
        time_window: OnlineTraceScoringTimeWindow,
        checkpoint: OnlineTraceScoringCheckpoint | None,
    ) -> _Discovery:
        """
        Fetch trace infos for every filter group and decide which scorers each trace needs.

        Args:
            time_window: OnlineTraceScoringTimeWindow with timestamp bounds.
            checkpoint: OnlineTraceScoringCheckpoint with last processed trace info.

        Returns:
            The discovery result. ``tasks`` contains only traces that still have scorers to run
            (sampled in and not already assessed by that scorer); every fetched trace appears in
            ``considered``.
        """
        discovery = _Discovery()

        # The checkpoint trace ID is only a valid tie-breaker if the window still starts at the
        # checkpoint timestamp (the lookback limit may have moved the window start forward).
        after_trace_id = None
        if (
            checkpoint is not None
            and checkpoint.trace_id is not None
            and checkpoint.timestamp_ms == time_window.min_trace_timestamp_ms
        ):
            after_trace_id = checkpoint.trace_id

        # Group scorers by filter string to fetch matching traces in a single query per filter
        for filter_string, scorers in self._sampler.group_scorers_by_filter(
            session_level=False
        ).items():
            combined_filter = (
                f"{EXCLUDE_EVAL_RUN_TRACES_FILTER} AND {filter_string}"
                if filter_string
                else EXCLUDE_EVAL_RUN_TRACES_FILTER
            )
            trace_infos = self._trace_loader.fetch_trace_infos_in_range(
                self._experiment_id,
                time_window.min_trace_timestamp_ms,
                time_window.max_trace_timestamp_ms,
                combined_filter,
                MAX_TRACES_PER_JOB,
                after_trace_id=after_trace_id,
                time_key="end_time_ms",
            )

            if not trace_infos:
                _logger.debug(f"No trace infos found for filter: {filter_string}")
                continue

            _logger.debug(f"Found {len(trace_infos)} trace infos for filter: {filter_string}")

            if len(trace_infos) >= MAX_TRACES_PER_JOB:
                # The page was full, so there may be more matching traces beyond the last one
                # returned. Nothing after it can be considered covered by this run.
                last = trace_infos[-1]
                discovery.full_page_last_keys.append((trace_end_time_ms(last), last.trace_id))

            for trace_info in trace_infos:
                trace_id = trace_info.trace_id
                if trace_id not in discovery.considered:
                    discovery.considered[trace_id] = ConsideredEntity(
                        timestamp_ms=trace_end_time_ms(trace_info),
                        entity_id=trace_id,
                        done=True,
                    )

                selected = self._sampler.sample(trace_id, scorers)
                # Skip scorers that already produced an assessment on this trace (e.g. this
                # window is being re-run after a crash or a held checkpoint) so retries never
                # duplicate visible assessments.
                selected = _scorers_not_yet_applied(trace_info, selected)
                if not selected:
                    continue

                if trace_id not in discovery.tasks:
                    discovery.tasks[trace_id] = TraceScoringTask(
                        trace=None,
                        scorers=[],
                        timestamp_ms=trace_end_time_ms(trace_info),
                        trace_info=trace_info,
                    )
                # Add scorers, avoiding duplicates (same scorer from different filters)
                task = discovery.tasks[trace_id]
                existing_scorer_names = {s.name for s in task.scorers}
                task.scorers.extend(s for s in selected if s.name not in existing_scorer_names)

        return discovery

    def _select_tasks(
        self, discovery: _Discovery, watermark: EntityKey | None, now_ms: int
    ) -> dict[str, TraceScoringTask]:
        """
        Decide which discovered tasks run in this job and mark the rest as pending.

        Tasks beyond the safe watermark or the per-job cap are deferred to the next run. Traces
        that are still in progress are held (the checkpoint stops before them) until they either
        complete or exceed ``MAX_TRACE_COMPLETION_WAIT_MS``, at which point they are given up on.

        Returns:
            Tasks to execute, ordered by ``(timestamp_ms, trace_id)``.
        """
        # Sort tasks by timestamp (ascending) to ensure chronological processing
        ordered = sorted(discovery.tasks.items(), key=lambda kv: (kv[1].timestamp_ms, kv[0]))

        runnable: dict[str, TraceScoringTask] = {}
        for trace_id, task in ordered:
            key = (task.timestamp_ms, trace_id)
            if watermark is not None and is_after(key, watermark):
                # Not covered by every filter group's page; the next run re-discovers it.
                discovery.considered[trace_id].done = False
                continue

            if task.trace_info is not None and task.trace_info.state == TraceState.IN_PROGRESS:
                discovery.considered[trace_id].done = self._give_up_on_incomplete_trace(
                    task, now_ms, reason="is still in progress"
                )
                continue

            if len(runnable) >= MAX_TRACES_PER_JOB:
                discovery.considered[trace_id].done = False
                continue

            runnable[trace_id] = task

        return runnable

    def _load_full_traces(
        self, tasks: dict[str, TraceScoringTask], discovery: _Discovery, now_ms: int
    ) -> None:
        """Attach full traces to tasks; tasks whose trace cannot be loaded are held or dropped."""
        full_traces = self._trace_loader.fetch_traces(list(tasks.keys()))
        trace_map = {t.info.trace_id: t for t in full_traces}

        for trace_id, task in list(tasks.items()):
            task.trace = trace_map.get(trace_id)
            if task.trace is None:
                discovery.considered[trace_id].done = self._give_up_on_incomplete_trace(
                    task, now_ms, reason="could not be loaded"
                )
                del tasks[trace_id]

    def _give_up_on_incomplete_trace(
        self, task: TraceScoringTask, now_ms: int, reason: str
    ) -> bool:
        """
        Decide whether to stop waiting for a trace that cannot be scored yet.

        While the trace is younger than ``MAX_TRACE_COMPLETION_WAIT_MS`` the checkpoint is held
        before it so the next run retries. Once it is older, an error assessment is recorded for
        each selected scorer so the trace is never silently dropped, and it counts as done.

        Returns:
            True if the trace is done (given up on, with the error durably recorded), False if
            the checkpoint should be held before it.
        """
        trace_id = task.trace_info.trace_id
        if now_ms - task.timestamp_ms <= MAX_TRACE_COMPLETION_WAIT_MS:
            _logger.debug(f"Trace {trace_id} {reason}; holding checkpoint to retry on the next run")
            return False

        message = (
            f"Trace {reason} more than {MAX_TRACE_COMPLETION_WAIT_MS // 1000}s after it last "
            "made progress; online scoring gave up waiting for it"
        )
        _logger.warning(f"Trace {trace_id}: {message}")
        # The trace has no usable span data, so attach the error to its metadata only.
        placeholder = Trace(info=task.trace_info, data=TraceData(spans=[]))
        # A previous run may already have recorded this (the window is being re-run).
        scorers = [
            scorer
            for scorer in task.scorers
            if not _incomplete_placeholders(placeholder, scorer.name)
        ]
        if not scorers:
            return True
        error = AssessmentError(error_code=TRACE_INCOMPLETE_ERROR_CODE, error_message=message)
        return self._persist(
            lambda: self._log_error_assessments(error, scorers, placeholder), trace_id
        )

    def _log_error_assessments(
        self,
        error: Exception | AssessmentError,
        scorers: list[Scorer],
        trace: Trace,
    ) -> None:
        """
        Log error assessments for failed scoring operations.

        Creates and logs error Feedback objects for each scorer when scoring fails,
        making failures visible in the trace's assessment history.

        Args:
            error: The exception that occurred during scoring.
            scorers: List of scorers that were being executed.
            trace: The trace being scored.

        Raises:
            Exception: If the error assessments themselves could not be persisted.
        """
        from mlflow.entities import AssessmentSource, AssessmentSourceType, Feedback
        from mlflow.genai.evaluation.harness import _log_assessments

        error_feedbacks = [
            Feedback(
                name=scorer.name,
                error=error,
                source=AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE),
                trace_id=trace.info.trace_id,
                metadata={AssessmentMetadataKey.ONLINE_SCORING_SCORER_NAME: scorer.name},
            )
            for scorer in scorers
        ]
        _log_assessments(trace=trace, assessments=error_feedbacks, run_id=None)

    def _score_trace_with_scorer(self, task: TraceScoringTask, scorer: Scorer, now_ms: int) -> bool:
        """
        Run one scorer on one trace and persist the result. Runs in a worker thread.

        A scorer failure (an exception, or a feedback carrying an error) is treated as
        transient: nothing is persisted and the trace is retried on the next run, until the
        trace is older than ``MAX_SCORER_FAILURE_RETRY_MS``. After that the error is persisted
        as an error feedback so it is visible and the pipeline can move on.

        Returns:
            True if the outcome for this scorer was durably persisted, False if the trace must
            be retried (scorer failure within the retry budget, or the store rejected the write).
        """
        # Import evaluation modules lazily to avoid pulling in pandas at module load
        # time, which would break the skinny client.
        from mlflow.genai.evaluation.entities import EvalItem
        from mlflow.genai.evaluation.harness import _compute_eval_scores, _log_assessments

        trace_id = task.trace.info.trace_id
        retry_budget_exhausted = now_ms - task.timestamp_ms > MAX_SCORER_FAILURE_RETRY_MS

        try:
            eval_item = EvalItem.from_trace(task.trace)
            result = _compute_eval_scores(eval_item=eval_item, scorers=[scorer])
            feedbacks = result.assessments or []
        except Exception as e:
            if not retry_budget_exhausted:
                _logger.info(
                    f"Scorer '{scorer.name}' failed on trace {trace_id}; holding checkpoint to "
                    f"retry on the next run: {e}"
                )
                return False
            _logger.warning(
                f"Scorer '{scorer.name}' still failing on trace {trace_id} after "
                f"{MAX_SCORER_FAILURE_RETRY_MS // 1000}s; recording the error: {e}"
            )
            error = e
            return self._persist(
                lambda: self._log_error_assessments(error, [scorer], task.trace), trace_id
            )

        if failures := [f for f in feedbacks if f.error is not None]:
            if not retry_budget_exhausted:
                _logger.info(
                    f"Scorer '{scorer.name}' returned an error for trace {trace_id}; holding "
                    f"checkpoint to retry on the next run: {failures[0].error}"
                )
                return False
            _logger.warning(
                f"Scorer '{scorer.name}' still failing on trace {trace_id} after "
                f"{MAX_SCORER_FAILURE_RETRY_MS // 1000}s; recording the error: {failures[0].error}"
            )

        if not feedbacks:
            return True

        for feedback in feedbacks:
            feedback.metadata = {
                **(feedback.metadata or {}),
                AssessmentMetadataKey.ONLINE_SCORING_SCORER_NAME: scorer.name,
            }
        placeholders = _incomplete_placeholders(task.trace, scorer.name)

        def write():
            _log_assessments(trace=task.trace, assessments=feedbacks, run_id=None)
            # The trace completed after we had given up on it: replace the placeholder error.
            for assessment_id in placeholders:
                mlflow.delete_assessment(trace_id=trace_id, assessment_id=assessment_id)

        return self._persist(write, trace_id)

    @staticmethod
    def _persist(write, trace_id: str) -> bool:
        """Run a store write; a failure means the trace must be retried."""
        try:
            write()
            return True
        except Exception as e:
            _logger.warning(
                f"Failed to log assessments for trace {trace_id}: {e}",
                exc_info=_logger.isEnabledFor(logging.DEBUG),
            )
            return False

    def _execute_scoring(
        self,
        tasks: dict[str, TraceScoringTask],
        now_ms: int,
    ) -> set[str]:
        """
        Execute trace scoring tasks in parallel, one unit of work per (trace, scorer) pair.

        Args:
            tasks: Trace-level scoring tasks.

        Returns:
            IDs of traces for which at least one scorer did not reach a durable outcome (the
            scorer failed within its retry budget, or the store rejected the write). The
            checkpoint must not move past these traces.
        """
        failed_trace_ids: set[str] = set()
        with ThreadPoolExecutor(
            max_workers=MLFLOW_ONLINE_SCORING_MAX_WORKER_THREADS.get(),
            thread_name_prefix="OnlineScoring",
        ) as executor:
            futures = {}
            for trace_id, task in tasks.items():
                if task.trace is None:
                    _logger.warning(f"Skipping task with no trace for trace_id: {trace_id}")
                    continue
                for scorer in task.scorers:
                    future = executor.submit(self._score_trace_with_scorer, task, scorer, now_ms)
                    futures[future] = trace_id

            for future in as_completed(futures):
                trace_id = futures[future]
                try:
                    persisted = future.result()
                except Exception as e:
                    # _score_trace_with_scorer handles its own errors; this is a safety net.
                    _logger.warning(
                        f"Unexpected error scoring trace {trace_id}: {e}", exc_info=True
                    )
                    persisted = False
                if not persisted:
                    failed_trace_ids.add(trace_id)
        return failed_trace_ids


def _scorers_not_yet_applied(trace_info: TraceInfo, scorers: list[Scorer]) -> list[Scorer]:
    """
    Drop scorers that already have an online-scoring assessment on this trace.

    A ``TRACE_INCOMPLETE`` error recorded when online scoring gave up waiting for the trace does
    not count: if the trace has since completed it should be scored for real (and the placeholder
    removed, see ``_incomplete_placeholders``).
    """
    if not scorers:
        return []
    applied = {
        (assessment.metadata or {}).get(AssessmentMetadataKey.ONLINE_SCORING_SCORER_NAME)
        for assessment in (trace_info.assessments or [])
        if not _is_incomplete_placeholder(assessment)
    }
    return [s for s in scorers if s.name not in applied]


def _is_incomplete_placeholder(assessment) -> bool:
    error = getattr(assessment, "error", None)
    return error is not None and getattr(error, "error_code", None) == TRACE_INCOMPLETE_ERROR_CODE


def _incomplete_placeholders(trace: Trace, scorer_name: str) -> list[str]:
    """IDs of ``TRACE_INCOMPLETE`` assessments this scorer previously left on the trace."""
    return [
        assessment.assessment_id
        for assessment in (trace.info.assessments or [])
        if _is_incomplete_placeholder(assessment)
        and (assessment.metadata or {}).get(AssessmentMetadataKey.ONLINE_SCORING_SCORER_NAME)
        == scorer_name
        and assessment.assessment_id
    ]
