"""Session-level online scoring processor for executing scorers on completed sessions."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from mlflow.entities import AssessmentError, Trace, TraceData
from mlflow.entities.assessment import Assessment
from mlflow.entities.trace_state import TraceState
from mlflow.environment_variables import MLFLOW_ONLINE_SCORING_MAX_WORKER_THREADS
from mlflow.genai.scorers.base import Scorer
from mlflow.genai.scorers.online.constants import (
    EXCLUDE_EVAL_RUN_TRACES_FILTER,
    MAX_JOB_RUNTIME_MS,
    MAX_SCORER_FAILURE_RETRY_MS,
    MAX_SESSIONS_PER_JOB,
    MAX_TRACE_COMPLETION_WAIT_MS,
)
from mlflow.genai.scorers.online.entities import CompletedSession, OnlineScorer
from mlflow.genai.scorers.online.sampler import OnlineScorerSampler
from mlflow.genai.scorers.online.session_checkpointer import (
    OnlineSessionCheckpointManager,
    OnlineSessionScoringCheckpoint,
    OnlineSessionScoringTimeWindow,
)
from mlflow.genai.scorers.online.trace_loader import OnlineTraceLoader
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


@dataclass
class SessionScoringTask:
    session: CompletedSession
    scorers: list[Scorer] = field(default_factory=list)


class _SessionNotReady(Exception):
    """Raised by ``_score_session`` when a session must be retried on a later run."""


class OnlineSessionScoringProcessor:
    """
    Orchestrates online scoring of completed sessions.

    This processor identifies sessions that have been inactive for a completion buffer
    period (no new traces added), applies session-level scorers to them, and maintains
    a checkpoint to avoid reprocessing. Sessions are processed in parallel with one
    thread per session.

    The processor:
    - Fetches completed sessions within a time window based on checkpoint state
    - Applies sampling to determine which scorers should run on each session
    - Loads all traces for each session and evaluates session-level scorers
    - Logs assessments with session metadata for cleanup tracking
    - Removes old assessments when a session is re-scored (e.g., if new traces are
      added since the last time it was scored)
    - Updates the checkpoint to the last processed session

    Sessions are processed in chronological order (sorted by last_trace_timestamp_ms
    and session_id) to ensure deterministic, resumable processing. The checkpoint is a
    durable watermark: it never moves past a session unless that session was durably
    assessed, was deterministically sampled out, or had no scorable traces. Scorer failures
    are retried on later runs rather than recorded, until the session is older than
    ``MAX_SCORER_FAILURE_RETRY_MS``; only then is an error feedback persisted.
    """

    def __init__(
        self,
        trace_loader: "OnlineTraceLoader",
        checkpoint_manager: OnlineSessionCheckpointManager,
        sampler: OnlineScorerSampler,
        experiment_id: str,
        tracking_store: AbstractStore,
    ):
        self._trace_loader = trace_loader
        self._checkpoint_manager = checkpoint_manager
        self._sampler = sampler
        self._experiment_id = experiment_id
        self._tracking_store = tracking_store

    @classmethod
    def create(
        cls,
        experiment_id: str,
        online_scorers: list[OnlineScorer],
        tracking_store: AbstractStore,
    ) -> "OnlineSessionScoringProcessor":
        """
        Factory method to create an OnlineSessionScoringProcessor with dependencies.

        Args:
            experiment_id: The experiment ID to process sessions from.
            online_scorers: List of OnlineScorer instances.
            tracking_store: The tracking store instance.

        Returns:
            Configured OnlineSessionScoringProcessor instance.
        """
        return cls(
            trace_loader=OnlineTraceLoader(tracking_store),
            checkpoint_manager=OnlineSessionCheckpointManager(tracking_store, experiment_id),
            sampler=OnlineScorerSampler(online_scorers),
            experiment_id=experiment_id,
            tracking_store=tracking_store,
        )

    def process_sessions(self) -> None:
        """
        Execute online scoring for completed sessions in the experiment.

        Finds sessions that have been inactive for the completion buffer duration,
        applies sampling to select scorers, runs scoring in parallel (one thread per
        session), and updates the checkpoint.
        """
        if not self._sampler._online_scorers:
            _logger.debug("No scorer configs provided, skipping")
            return

        # Run batches until the backlog is drained (see OnlineTraceScoringProcessor).
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
                    f"Online session scoring for experiment {self._experiment_id} still has a "
                    "backlog after reaching the job runtime limit; the next job continues"
                )
                break

    def _process_batch(self) -> tuple[bool, EntityKey | None]:
        """
        Run one session scoring batch.

        Returns:
            ``(more_pending, persisted)``: whether at least one discovery page was full (so more
            completed sessions may exist beyond this batch), and the checkpoint persisted by
            this batch (``None`` if it did not move).
        """
        time_window = self._checkpoint_manager.calculate_time_window()
        checkpoint = self._checkpoint_manager.get_checkpoint()

        _logger.debug(
            f"Session scoring for experiment {self._experiment_id}: "
            f"looking for sessions in "
            f"[{time_window.min_last_trace_timestamp_ms}, "
            f"{time_window.max_last_trace_timestamp_ms}]"
        )

        considered, session_tasks, full_page_last_keys = self._fetch_and_filter_completed_sessions(
            time_window, checkpoint
        )
        watermark = safe_watermark(full_page_last_keys)
        session_tasks = self._select_tasks(session_tasks, considered, watermark)

        if session_tasks:
            _logger.debug(f"Found {len(session_tasks)} completed sessions for scoring")
            for session_id in self._execute_session_scoring(session_tasks):
                considered[session_id].done = False
        else:
            _logger.debug("No completed sessions found, skipping")

        new_checkpoint = compute_checkpoint(
            previous=(checkpoint.timestamp_ms, checkpoint.session_id) if checkpoint else None,
            window_max_ms=time_window.max_last_trace_timestamp_ms,
            considered=considered.values(),
            watermark=watermark,
        )
        if new_checkpoint is None:
            _logger.debug("Checkpoint not advanced: earliest pending session is still outstanding")
        else:
            self._checkpoint_manager.persist_checkpoint(
                OnlineSessionScoringCheckpoint(
                    timestamp_ms=new_checkpoint[0], session_id=new_checkpoint[1]
                )
            )

        _logger.debug(
            f"Online session scoring batch completed for experiment {self._experiment_id}"
        )
        return watermark is not None, new_checkpoint

    def _fetch_and_filter_completed_sessions(
        self,
        time_window: OnlineSessionScoringTimeWindow,
        checkpoint: OnlineSessionScoringCheckpoint | None,
    ) -> tuple[dict[str, ConsideredEntity], list[SessionScoringTask], list[EntityKey]]:
        """
        Fetch completed sessions and create scoring tasks with applicable scorers.

        Fetches sessions separately for each unique filter_string used by session-level scorers,
        creating tasks that track which scorers should run on each session based on filter match.

        Sessions at the checkpoint boundary that were already scored in a previous run are
        excluded by the store (``after_session_id``), so the tie-break comparison uses the same
        collation as the store's ordering.

        Args:
            time_window: Time window with min/max last trace timestamps
            checkpoint: Current checkpoint with timestamp and session_id

        Returns:
            A tuple of: every fetched session keyed by session ID with its done status (sessions
            with no scorers to run are done); the list of SessionScoringTask objects for sessions
            with scorers to run, sorted by (last_trace_timestamp_ms ASC, session_id ASC); and the
            last (timestamp, session_id) of every filter group whose page was full.
        """
        # The checkpoint session ID is only a valid tie-breaker if the window still starts at the
        # checkpoint timestamp (the lookback limit may have moved the window start forward).
        after_session_id = None
        if (
            checkpoint is not None
            and checkpoint.session_id is not None
            and checkpoint.timestamp_ms == time_window.min_last_trace_timestamp_ms
        ):
            after_session_id = checkpoint.session_id

        # Group session-level scorers by their filter_string
        session_scorers_by_filter = self._sampler.group_scorers_by_filter(session_level=True)

        # Fetch completed sessions for each filter group and build tasks
        considered: dict[str, ConsideredEntity] = {}
        tasks: dict[str, SessionScoringTask] = {}
        full_page_last_keys: list[EntityKey] = []
        for filter_string, scorers in session_scorers_by_filter.items():
            sessions = self._tracking_store.find_completed_sessions(
                experiment_id=self._experiment_id,
                min_last_trace_timestamp_ms=time_window.min_last_trace_timestamp_ms,
                max_last_trace_timestamp_ms=time_window.max_last_trace_timestamp_ms,
                max_results=MAX_SESSIONS_PER_JOB,
                filter_string=filter_string,
                after_session_id=after_session_id,
            )
            if len(sessions) >= MAX_SESSIONS_PER_JOB:
                # The page was full, so there may be more matching sessions beyond the last
                # one returned. Nothing after it can be considered covered by this run.
                last = sessions[-1]
                full_page_last_keys.append((last.last_trace_timestamp_ms, last.session_id))

            # For each session that matches this filter, add applicable scorers
            for session in sessions:
                if session.session_id not in considered:
                    considered[session.session_id] = ConsideredEntity(
                        timestamp_ms=session.last_trace_timestamp_ms,
                        entity_id=session.session_id,
                        done=True,
                    )
                # Apply sampling to select which scorers from this filter group should run
                if selected := self._sampler.sample(session.session_id, scorers):
                    if session.session_id not in tasks:
                        tasks[session.session_id] = SessionScoringTask(session=session, scorers=[])
                    # Add scorers, avoiding duplicates (same scorer from different filters)
                    existing_scorer_names = {s.name for s in tasks[session.session_id].scorers}
                    tasks[session.session_id].scorers.extend(
                        s for s in selected if s.name not in existing_scorer_names
                    )

        # Sort tasks by (last_trace_timestamp_ms ASC, session_id ASC) for deterministic ordering
        sorted_tasks = sorted(
            tasks.values(),
            key=lambda t: (t.session.last_trace_timestamp_ms, t.session.session_id),
        )
        return considered, sorted_tasks, full_page_last_keys

    @staticmethod
    def _select_tasks(
        sorted_tasks: list[SessionScoringTask],
        considered: dict[str, ConsideredEntity],
        watermark: EntityKey | None,
    ) -> list[SessionScoringTask]:
        """
        Keep the tasks this job will run; sessions beyond the safe watermark or the per-job cap
        are marked pending so the checkpoint stops before them and the next run picks them up.
        """
        runnable = []
        for task in sorted_tasks:
            key = (task.session.last_trace_timestamp_ms, task.session.session_id)
            if (watermark is not None and is_after(key, watermark)) or len(
                runnable
            ) >= MAX_SESSIONS_PER_JOB:
                considered[task.session.session_id].done = False
                continue
            runnable.append(task)
        return runnable

    def _clean_up_old_assessments(
        self, trace, session_id: str, new_assessments: list[Assessment]
    ) -> None:
        """
        Remove online scoring assessments left by a previous scoring of this session.

        Finds and deletes previous assessments from the same session/scorers to avoid
        duplicates when a session is re-scored (e.g., when new traces are added, or when a
        run is retried). Called before the new assessments are logged.

        Args:
            trace: The Trace object containing all assessments.
            session_id: The session ID to match in assessment metadata.
            new_assessments: List of new assessments about to be logged.
        """
        if not trace or not trace.info.assessments:
            return

        new_assessment_names = {a.name for a in new_assessments}
        new_assessment_ids = {a.assessment_id for a in new_assessments}

        deleted_count = 0
        for assessment in trace.info.assessments:
            metadata = assessment.metadata or {}
            online_session_id = metadata.get(AssessmentMetadataKey.ONLINE_SCORING_SESSION_ID)

            if (
                online_session_id == session_id
                and assessment.name in new_assessment_names
                and assessment.assessment_id not in new_assessment_ids
            ):
                self._tracking_store.delete_assessment(
                    trace_id=trace.info.trace_id, assessment_id=assessment.assessment_id
                )
                deleted_count += 1

        if deleted_count > 0:
            _logger.debug(f"Deleted {deleted_count} old assessments for session {session_id}")

    def _execute_session_scoring(self, tasks: list[SessionScoringTask]) -> set[str]:
        """
        Execute session-level scoring tasks in parallel.

        Each thread loads traces for its session independently and runs all applicable
        scorers on that session.

        Args:
            tasks: List of SessionScoringTask objects containing sessions and their scorers.

        Returns:
            IDs of sessions that were not durably processed and must be retried on a later run.
            The checkpoint must not move past these sessions.
        """
        pending_session_ids: set[str] = set()
        with ThreadPoolExecutor(
            max_workers=MLFLOW_ONLINE_SCORING_MAX_WORKER_THREADS.get(),
            thread_name_prefix="SessionScoring",
        ) as executor:
            futures = {}
            for task in tasks:
                future = executor.submit(self._score_session, task)
                futures[future] = task

            for future in as_completed(futures):
                task = futures[future]
                session_id = task.session.session_id
                try:
                    future.result()
                except _SessionNotReady as e:
                    _logger.info(f"Session {session_id} deferred to a later run: {e}")
                    pending_session_ids.add(session_id)
                except Exception as e:
                    _logger.warning(
                        f"Failed to score session {session_id}: {e}",
                        exc_info=True,
                    )
                    pending_session_ids.add(session_id)
        return pending_session_ids

    def _score_session(self, task: SessionScoringTask) -> None:
        """
        Score a single session by loading its traces and applying pre-selected scorers.

        This method runs in a worker thread. It fetches all traces for the session
        and runs the scorers that were already selected during task creation.

        Args:
            task: The SessionScoringTask containing the session and applicable scorers.

        Raises:
            _SessionNotReady: If the session cannot be scored yet (a trace is still in
                progress) or a scorer failed within its retry budget; it is retried later.
            Exception: If the session's results (or error feedbacks) could not be persisted.
        """
        # Import evaluation modules lazily to avoid pulling in pandas at module load
        # time, which would break the skinny client.
        from mlflow.genai.evaluation.entities import EvalItem
        from mlflow.genai.evaluation.harness import _log_assessments
        from mlflow.genai.evaluation.session_utils import evaluate_session_level_scorers

        if not task.scorers:
            return

        session = task.session
        session_filter = f"metadata.`mlflow.trace.session` = '{session.session_id}'"
        combined_filter = f"{EXCLUDE_EVAL_RUN_TRACES_FILTER} AND {session_filter}"
        # A session is scored as a whole, so load every one of its traces: a cap here would
        # silently score a long conversation on its first N turns only.
        trace_infos = self._trace_loader.fetch_trace_infos_in_range(
            experiment_id=self._experiment_id,
            start_time_ms=session.first_trace_timestamp_ms,
            end_time_ms=session.last_trace_timestamp_ms,
            filter_string=combined_filter,
            max_traces=None,
        )

        if not trace_infos:
            _logger.warning(f"No traces found for session {session.session_id}")
            return

        # A trace that is still running cannot be scored (its spans are incomplete). Wait for it
        # while it is young enough to plausibly finish; otherwise record why it was excluded on
        # the trace itself and score the session without it.
        now_ms = int(time.time() * 1000)
        complete_infos = []
        for trace_info in trace_infos:
            if trace_info.state != TraceState.IN_PROGRESS:
                complete_infos.append(trace_info)
            elif now_ms - trace_info.timestamp_ms <= MAX_TRACE_COMPLETION_WAIT_MS:
                raise _SessionNotReady(f"trace {trace_info.trace_id} is still in progress")
            else:
                self._record_incomplete_trace(
                    task,
                    trace_info,
                    reason=(
                        f"Trace is still in progress more than "
                        f"{MAX_TRACE_COMPLETION_WAIT_MS // 1000}s after it started; excluded from "
                        f"session-level scoring of session {session.session_id}"
                    ),
                )
        if not complete_infos:
            _logger.warning(f"No complete traces found for session {session.session_id}")
            return

        trace_ids = [t.trace_id for t in complete_infos]
        full_traces = self._trace_loader.fetch_traces(trace_ids)
        if not full_traces:
            # Trace infos exist but none of their span data could be loaded (still being
            # exported, or archived/deleted). Wait while the session is recent; then record the
            # failure on the session's first trace so it is never silently dropped.
            if now_ms - session.last_trace_timestamp_ms <= MAX_TRACE_COMPLETION_WAIT_MS:
                raise _SessionNotReady("no trace data could be loaded yet")
            self._record_incomplete_trace(
                task,
                min(complete_infos, key=lambda t: t.timestamp_ms),
                reason=(
                    f"Trace data for session {session.session_id} could not be loaded more than "
                    f"{MAX_TRACE_COMPLETION_WAIT_MS // 1000}s after its last trace started; online "
                    "scoring gave up waiting for it"
                ),
            )
            return
        if len(full_traces) < len(trace_ids):
            _logger.warning(
                f"Only {len(full_traces)} of {len(trace_ids)} traces in session "
                f"{session.session_id} could be loaded; scoring with the available traces"
            )
        full_traces.sort(key=lambda t: t.info.timestamp_ms)
        session_items = [EvalItem.from_trace(t) for t in full_traces]

        retry_budget_exhausted = (
            now_ms - session.last_trace_timestamp_ms > MAX_SCORER_FAILURE_RETRY_MS
        )

        try:
            result = evaluate_session_level_scorers(
                session_id=session.session_id,
                session_items=session_items,
                multi_turn_scorers=task.scorers,
            )
            assessments = result.assessments
            trace = result.eval_item.trace
        except Exception as e:
            # evaluate_session_level_scorers turns scorer errors into error feedbacks, so this is
            # an unexpected failure. Retry it like a scorer failure; once the retry budget is
            # exhausted surface it on the first trace so the session is durably marked as
            # attempted rather than silently skipped.
            if not retry_budget_exhausted:
                raise _SessionNotReady(f"session-level evaluation failed: {e}") from e
            _logger.warning(
                f"Session-level evaluation still failing for session {session.session_id} after "
                f"{MAX_SCORER_FAILURE_RETRY_MS // 1000}s; recording the error: {e}",
                exc_info=True,
            )
            trace = full_traces[0]
            assessments = self._build_error_feedbacks(e, task.scorers, trace)

        if trace is None:
            return

        # Scorer failures are usually transient. While the session is within its retry budget,
        # persist only the successful results and defer the session so the failed scorers run
        # again next time (the cleanup below replaces the successes on that re-run). Once the
        # budget is exhausted the errors are persisted too, so they are visible.
        failed = [a for a in assessments if a.error is not None]
        if failed and not retry_budget_exhausted:
            assessments = [a for a in assessments if a.error is None]

        if assessments:
            try:
                # Add session ID metadata to identify these as online scoring assessments
                for feedback in assessments:
                    feedback.metadata = {
                        **(feedback.metadata or {}),
                        AssessmentMetadataKey.ONLINE_SCORING_SESSION_ID: session.session_id,
                    }
                # Remove results from a previous scoring of this session *before* logging the
                # new ones. If anything fails in between, the session is retried and re-logged,
                # so a retry can never leave two visible assessments for the same scorer.
                self._clean_up_old_assessments(trace, session.session_id, assessments)
                _log_assessments(run_id=None, trace=trace, assessments=assessments)
            except Exception as e:
                _logger.warning(
                    f"Failed to log assessments for trace {trace.info.trace_id} "
                    f"in session {session.session_id}: {e}",
                    exc_info=_logger.isEnabledFor(logging.DEBUG),
                )
                raise

        if failed and not retry_budget_exhausted:
            names = ", ".join(sorted({a.name for a in failed}))
            raise _SessionNotReady(f"scorer(s) {names} failed: {failed[0].error}")
        if failed:
            _logger.warning(
                f"Scorer(s) still failing for session {session.session_id} after "
                f"{MAX_SCORER_FAILURE_RETRY_MS // 1000}s; recorded the error: {failed[0].error}"
            )

    def _record_incomplete_trace(self, task: SessionScoringTask, trace_info, reason: str) -> None:
        """
        Persist an error assessment per scorer on a trace that online scoring gave up on, so the
        trace is never dropped with nothing recorded. Raises if the write fails, so the session
        is retried rather than dropped.
        """
        from mlflow.genai.evaluation.harness import _log_assessments

        _logger.warning(f"Trace {trace_info.trace_id}: {reason}")
        # The trace has no usable span data, so attach the error to its metadata only.
        placeholder = Trace(info=trace_info, data=TraceData(spans=[]))
        error = AssessmentError(error_code="TRACE_INCOMPLETE", error_message=reason)
        feedbacks = self._build_error_feedbacks(error, task.scorers, placeholder)
        for feedback in feedbacks:
            feedback.metadata = {
                **(feedback.metadata or {}),
                AssessmentMetadataKey.ONLINE_SCORING_SESSION_ID: task.session.session_id,
            }
        _log_assessments(run_id=None, trace=placeholder, assessments=feedbacks)

    @staticmethod
    def _build_error_feedbacks(
        error: Exception | AssessmentError, scorers: list[Scorer], trace: Trace
    ) -> list:
        from mlflow.entities import AssessmentSource, AssessmentSourceType, Feedback

        return [
            Feedback(
                name=scorer.name,
                error=error,
                source=AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE),
                trace_id=trace.info.trace_id,
            )
            for scorer in scorers
        ]
