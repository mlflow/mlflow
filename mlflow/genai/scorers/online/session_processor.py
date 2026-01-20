"""Session-level online scoring processor for executing scorers on completed sessions."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from mlflow.entities.assessment import Assessment
from mlflow.environment_variables import MLFLOW_ONLINE_SCORING_MAX_WORKER_THREADS
from mlflow.genai.scorers.base import Scorer
from mlflow.genai.scorers.online.constants import (
    EXCLUDE_EVAL_RUN_TRACES_FILTER,
    MAX_SESSIONS_PER_JOB,
)
from mlflow.genai.scorers.online.entities import CompletedSession, OnlineScorer
from mlflow.genai.scorers.online.sampler import OnlineScorerSampler
from mlflow.genai.scorers.online.session_checkpointer import (
    OnlineSessionCheckpointManager,
    OnlineSessionScoringCheckpoint,
    OnlineSessionScoringTimeWindow,
)
from mlflow.genai.scorers.online.trace_loader import OnlineTraceLoader
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.tracing.constant import AssessmentMetadataKey

_logger = logging.getLogger(__name__)


@dataclass
class SessionScoringTask:
    session: CompletedSession
    scorers: list[Scorer] = field(default_factory=list)


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
    and session_id) to ensure deterministic, resumable processing.
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

        time_window = self._checkpoint_manager.calculate_time_window()
        checkpoint = self._checkpoint_manager.get_checkpoint()

        _logger.debug(
            f"Session scoring for experiment {self._experiment_id}: "
            f"looking for sessions in "
            f"[{time_window.min_last_trace_timestamp_ms}, "
            f"{time_window.max_last_trace_timestamp_ms}]"
        )

        session_tasks = self._fetch_and_filter_completed_sessions(time_window, checkpoint)

        if not session_tasks:
            _logger.debug("No completed sessions found, skipping")
            # Still need to advance checkpoint to avoid reprocessing the same time window
            checkpoint = OnlineSessionScoringCheckpoint(
                timestamp_ms=time_window.max_last_trace_timestamp_ms,
                session_id=None,
            )
            self._checkpoint_manager.persist_checkpoint(checkpoint)
            return

        _logger.debug(f"Found {len(session_tasks)} completed sessions for scoring")

        self._execute_session_scoring(session_tasks)

        # Update checkpoint to last processed session
        latest_task = session_tasks[-1]
        checkpoint = OnlineSessionScoringCheckpoint(
            timestamp_ms=latest_task.session.last_trace_timestamp_ms,
            session_id=latest_task.session.session_id,
        )
        self._checkpoint_manager.persist_checkpoint(checkpoint)

        _logger.debug(f"Online session scoring completed for experiment {self._experiment_id}")

    def _fetch_and_filter_completed_sessions(
        self,
        time_window: OnlineSessionScoringTimeWindow,
        checkpoint: OnlineSessionScoringCheckpoint | None,
    ) -> list[SessionScoringTask]:
        """
        Fetch completed sessions and create scoring tasks with applicable scorers.

        Fetches sessions separately for each unique filter_string used by session-level scorers,
        creating tasks that track which scorers should run on each session based on filter match.

        Sessions at the checkpoint boundary are filtered based on session_id to avoid
        reprocessing sessions that were already scored in a previous run.

        Args:
            time_window: Time window with min/max last trace timestamps
            checkpoint: Current checkpoint with timestamp and session_id

        Returns:
            List of SessionScoringTask objects, each containing a session and applicable scorers,
            sorted by (session.last_trace_timestamp_ms ASC, session.session_id ASC)
        """
        # Group session-level scorers by their filter_string
        session_scorers_by_filter = self._sampler.group_scorers_by_filter(session_level=True)

        # Fetch completed sessions for each filter group and build tasks
        tasks = {}
        for filter_string, scorers in session_scorers_by_filter.items():
            sessions = self._tracking_store.find_completed_sessions(
                experiment_id=self._experiment_id,
                min_last_trace_timestamp_ms=time_window.min_last_trace_timestamp_ms,
                max_last_trace_timestamp_ms=time_window.max_last_trace_timestamp_ms,
                max_results=MAX_SESSIONS_PER_JOB,
                filter_string=filter_string,
            )
            # For each session that matches this filter, add applicable scorers
            for session in sessions:
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

        # Filter out sessions at checkpoint boundary that have already been processed
        if checkpoint is not None and checkpoint.session_id is not None:
            sorted_tasks = [
                task
                for task in sorted_tasks
                if not (
                    task.session.last_trace_timestamp_ms == checkpoint.timestamp_ms
                    and task.session.session_id <= checkpoint.session_id
                )
            ]

        # Respect max_results limit
        return sorted_tasks[:MAX_SESSIONS_PER_JOB]

    def _clean_up_old_assessments(
        self, trace, session_id: str, new_assessments: list[Assessment]
    ) -> None:
        """
        Remove old online scoring assessments after successfully logging new ones.

        Finds and deletes previous assessments from the same session/scorers to avoid
        duplicates when a session is re-scored (e.g., when new traces are added).

        Args:
            trace: The Trace object containing all assessments.
            session_id: The session ID to match in assessment metadata.
            new_assessments: List of new assessments that were just logged.
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

    def _execute_session_scoring(self, tasks: list[SessionScoringTask]) -> None:
        """
        Execute session-level scoring tasks in parallel.

        Each thread loads traces for its session independently and runs all applicable
        scorers on that session.

        Args:
            tasks: List of SessionScoringTask objects containing sessions and their scorers.
        """
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
                try:
                    future.result()
                except Exception as e:
                    _logger.warning(
                        f"Failed to score session {task.session.session_id}: {e}",
                        exc_info=True,
                    )

    def _score_session(self, task: SessionScoringTask) -> None:
        """
        Score a single session by loading its traces and applying pre-selected scorers.

        This method runs in a worker thread. It fetches all traces for the session
        and runs the scorers that were already selected during task creation.

        Args:
            task: The SessionScoringTask containing the session and applicable scorers.
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
        trace_infos = self._trace_loader.fetch_trace_infos_in_range(
            experiment_id=self._experiment_id,
            start_time_ms=session.first_trace_timestamp_ms,
            end_time_ms=session.last_trace_timestamp_ms,
            filter_string=combined_filter,
        )

        if not trace_infos:
            _logger.warning(f"No traces found for session {session.session_id}")
            return

        trace_ids = [t.trace_id for t in trace_infos]
        full_traces = self._trace_loader.fetch_traces(trace_ids)
        if not full_traces:
            _logger.warning(f"Failed to fetch full traces for session {session.session_id}")
            return
        full_traces.sort(key=lambda t: t.info.timestamp_ms)
        trace_map = {t.info.trace_id: t for t in full_traces}
        session_items = [EvalItem.from_trace(t) for t in full_traces]

        result = evaluate_session_level_scorers(
            session_id=session.session_id,
            session_items=session_items,
            multi_turn_scorers=task.scorers,
        )

        for trace_id, feedbacks in result.items():
            if feedbacks and (trace := trace_map.get(trace_id)):
                try:
                    # Add session ID metadata to identify these as online scoring assessments
                    for feedback in feedbacks:
                        feedback.metadata = {
                            **(feedback.metadata or {}),
                            AssessmentMetadataKey.ONLINE_SCORING_SESSION_ID: session.session_id,
                        }
                    _log_assessments(run_id=None, trace=trace, assessments=feedbacks)

                    # Clean up old assessments after successfully logging new ones
                    self._clean_up_old_assessments(trace, session.session_id, feedbacks)
                except Exception as e:
                    _logger.warning(
                        f"Failed to log assessments for trace {trace_id} "
                        f"in session {session.session_id}: {e}",
                        exc_info=_logger.isEnabledFor(logging.DEBUG),
                    )
