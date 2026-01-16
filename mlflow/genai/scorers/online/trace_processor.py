"""Online scoring processor for executing scorers on traces."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from mlflow.entities import Trace
from mlflow.environment_variables import MLFLOW_ONLINE_SCORING_MAX_WORKER_THREADS
from mlflow.genai.scorers.base import Scorer
from mlflow.genai.scorers.online.constants import EXCLUDE_EVAL_RUN_TRACES_FILTER, MAX_TRACES_PER_JOB
from mlflow.genai.scorers.online.entities import OnlineScorer
from mlflow.genai.scorers.online.sampler import OnlineScorerSampler
from mlflow.genai.scorers.online.trace_checkpointer import (
    OnlineTraceCheckpointManager,
    OnlineTraceScoringCheckpoint,
)
from mlflow.genai.scorers.online.trace_loader import OnlineTraceLoader
from mlflow.store.tracking.abstract_store import AbstractStore

_logger = logging.getLogger(__name__)


@dataclass
class TraceScoringTask:
    """A task to score a single trace with multiple scorers."""

    trace: Trace
    scorers: list[Scorer]
    timestamp_ms: int


class OnlineTraceScoringProcessor:
    """Orchestrates online scoring of individual traces."""

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

        Fetches traces since the last checkpoint, applies sampling to select
        scorers, runs scoring in parallel, and updates the checkpoint.
        """
        time_window = self._checkpoint_manager.calculate_time_window()
        checkpoint = self._checkpoint_manager.get_checkpoint()

        _logger.debug(
            f"Online scoring for experiment {self._experiment_id}: "
            f"time window [{time_window.min_trace_timestamp_ms}, "
            f"{time_window.max_trace_timestamp_ms}]"
        )

        tasks = self._build_scoring_tasks(time_window, checkpoint)

        if not tasks:
            _logger.debug("No traces selected after sampling, skipping")
            # Still need to advance checkpoint to avoid reprocessing the same time window
            checkpoint = OnlineTraceScoringCheckpoint(
                timestamp_ms=time_window.max_trace_timestamp_ms,
                trace_id=None,
            )
            self._checkpoint_manager.persist_checkpoint(checkpoint)
            return

        _logger.debug(f"Running scoring: {len(tasks)} trace tasks")

        sampled_trace_ids = list(tasks.keys())
        full_traces = self._trace_loader.fetch_traces(sampled_trace_ids)
        trace_map = {t.info.trace_id: t for t in full_traces}

        for trace_id, task in tasks.items():
            task.trace = trace_map.get(trace_id)

        self._execute_scoring(tasks)

        # Find the trace with the latest timestamp to use for checkpoint
        if full_traces:
            latest_trace = max(full_traces, key=lambda t: (t.info.timestamp_ms, t.info.trace_id))
            checkpoint = OnlineTraceScoringCheckpoint(
                timestamp_ms=latest_trace.info.timestamp_ms,
                trace_id=latest_trace.info.trace_id,
            )
        else:
            # If no traces were fetched, use the end of the time window as checkpoint
            checkpoint = OnlineTraceScoringCheckpoint(
                timestamp_ms=time_window.max_trace_timestamp_ms,
                trace_id=None,
            )
        self._checkpoint_manager.persist_checkpoint(checkpoint)

        _logger.debug(f"Online trace scoring completed for experiment {self._experiment_id}")

    def _build_scoring_tasks(
        self,
        time_window,
        checkpoint,
    ) -> dict[str, TraceScoringTask]:
        """
        Build scoring tasks by fetching trace infos and applying sampling.

        Args:
            time_window: OnlineTraceScoringTimeWindow with timestamp bounds.
            checkpoint: OnlineTraceScoringCheckpoint with last processed trace info.

        Returns:
            Dictionary mapping trace_id to TraceScoringTask.
        """
        tasks: dict[str, TraceScoringTask] = {}

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
            )

            if not trace_infos:
                _logger.debug(f"No trace infos found for filter: {filter_string}")
                continue

            # Filter out traces at checkpoint boundary that have already been processed.
            # Traces are ordered by (timestamp_ms ASC, trace_id ASC), so we filter out
            # any traces with the checkpoint timestamp and trace_id <= checkpoint.trace_id.
            if checkpoint is not None and checkpoint.trace_id is not None:
                trace_infos = [
                    t
                    for t in trace_infos
                    if not (
                        t.timestamp_ms == checkpoint.timestamp_ms
                        and t.trace_id <= checkpoint.trace_id
                    )
                ]

            _logger.debug(f"Found {len(trace_infos)} trace infos for filter: {filter_string}")

            for trace_info in trace_infos:
                trace_id = trace_info.trace_id
                if selected := self._sampler.sample(trace_id, scorers):
                    # Store just the trace_id and scorers - we'll fetch full traces later
                    if trace_id not in tasks:
                        tasks[trace_id] = TraceScoringTask(
                            trace=None, scorers=[], timestamp_ms=trace_info.timestamp_ms
                        )
                    # Add scorers, avoiding duplicates (same scorer from different filters)
                    existing_scorer_names = {s.name for s in tasks[trace_id].scorers}
                    tasks[trace_id].scorers.extend(
                        s for s in selected if s.name not in existing_scorer_names
                    )

        # Sort tasks by timestamp (ascending) to ensure chronological processing
        # and truncate to MAX_TRACES_PER_JOB (list slicing handles len < MAX).
        sorted_trace_ids = sorted(tasks.keys(), key=lambda tid: (tasks[tid].timestamp_ms, tid))
        return {tid: tasks[tid] for tid in sorted_trace_ids[:MAX_TRACES_PER_JOB]}

    def _log_error_assessments(
        self,
        error: Exception,
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
        """
        from mlflow.entities import AssessmentSource, AssessmentSourceType, Feedback
        from mlflow.genai.evaluation.harness import _log_assessments

        error_feedbacks = [
            Feedback(
                name=scorer.name,
                error=error,
                source=AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE),
                trace_id=trace.info.trace_id,
            )
            for scorer in scorers
        ]
        try:
            _log_assessments(trace=trace, assessments=error_feedbacks, run_id=None)
        except Exception as log_error:
            _logger.warning(
                f"Failed to log error assessments for trace {trace.info.trace_id}: {log_error}",
                exc_info=_logger.isEnabledFor(logging.DEBUG),
            )

    def _execute_scoring(
        self,
        tasks: dict[str, TraceScoringTask],
    ) -> None:
        """
        Execute trace scoring tasks in parallel.

        Args:
            tasks: Trace-level scoring tasks.
        """
        # Import evaluation modules lazily to avoid pulling in pandas at module load
        # time, which would break the skinny client.
        from mlflow.genai.evaluation.entities import EvalItem
        from mlflow.genai.evaluation.harness import _compute_eval_scores, _log_assessments

        with ThreadPoolExecutor(
            max_workers=MLFLOW_ONLINE_SCORING_MAX_WORKER_THREADS.get(),
            thread_name_prefix="OnlineScoring",
        ) as executor:
            futures = {}
            for trace_id, task in tasks.items():
                if task.trace is None:
                    _logger.warning(f"Skipping task with no trace for trace_id: {trace_id}")
                    continue
                eval_item = EvalItem.from_trace(task.trace)
                future = executor.submit(
                    _compute_eval_scores, eval_item=eval_item, scorers=task.scorers
                )
                futures[future] = task

            for future in as_completed(futures):
                task = futures[future]
                try:
                    if feedbacks := future.result():
                        _log_assessments(trace=task.trace, assessments=feedbacks, run_id=None)
                except Exception as e:
                    _logger.warning(
                        f"Failed to score trace {task.trace.info.trace_id}: {e}",
                        exc_info=_logger.isEnabledFor(logging.DEBUG),
                    )
                    self._log_error_assessments(e, task.scorers, task.trace)
