"""Checkpoint management for trace-level online scoring."""

import json
import logging
import time
from dataclasses import asdict, dataclass

from mlflow.entities.experiment_tag import ExperimentTag
from mlflow.genai.scorers.online.constants import MAX_LOOKBACK_MS
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.utils.mlflow_tags import MLFLOW_LATEST_ONLINE_SCORING_TRACE_CHECKPOINT

_logger = logging.getLogger(__name__)


@dataclass
class OnlineTraceScoringCheckpoint:
    timestamp_ms: int  # Timestamp of the last processed trace in milliseconds
    trace_id: str | None = None  # Trace ID used as tie breaker when traces have same timestamp

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> "OnlineTraceScoringCheckpoint":
        data = json.loads(json_str)
        return cls(**data)


@dataclass
class OnlineTraceScoringTimeWindow:
    min_trace_timestamp_ms: int
    max_trace_timestamp_ms: int


class OnlineTraceCheckpointManager:
    def __init__(self, tracking_store: AbstractStore, experiment_id: str):
        self._tracking_store = tracking_store
        self._experiment_id = experiment_id

    def get_checkpoint(self) -> OnlineTraceScoringCheckpoint | None:
        """
        Get the last processed trace checkpoint from the experiment tag.

        Returns:
            OnlineTraceScoringCheckpoint, or None if no checkpoint exists.
        """
        try:
            experiment = self._tracking_store.get_experiment(self._experiment_id)
            if checkpoint_str := experiment.tags.get(MLFLOW_LATEST_ONLINE_SCORING_TRACE_CHECKPOINT):
                return OnlineTraceScoringCheckpoint.from_json(checkpoint_str)
        except (TypeError, ValueError, json.JSONDecodeError) as e:
            _logger.debug(
                f"Failed to parse checkpoint for experiment {self._experiment_id}: {e}",
                exc_info=True,
            )

    def persist_checkpoint(self, checkpoint: OnlineTraceScoringCheckpoint) -> None:
        """
        Persist the checkpoint tag with a new checkpoint.

        Args:
            checkpoint: The checkpoint to store.
        """
        self._tracking_store.set_experiment_tag(
            self._experiment_id,
            ExperimentTag(MLFLOW_LATEST_ONLINE_SCORING_TRACE_CHECKPOINT, checkpoint.to_json()),
        )

    def calculate_time_window(self) -> OnlineTraceScoringTimeWindow:
        """
        Calculate the time window for trace scoring.

        Enforces a maximum lookback period to prevent getting stuck on persistently
        failing traces. If the checkpoint is older than MAX_LOOKBACK_MS, uses
        current_time - MAX_LOOKBACK_MS instead to skip over old problematic traces.

        Returns:
            OnlineTraceScoringTimeWindow with min and max trace timestamps.
            min_trace_timestamp_ms is the checkpoint if it exists and is within the
            lookback period, otherwise now - MAX_LOOKBACK_MS.
            max_trace_timestamp_ms is the current time.
        """
        current_time_ms = int(time.time() * 1000)
        checkpoint = self.get_checkpoint()

        # Start from checkpoint, but never look back more than MAX_LOOKBACK_MS
        min_lookback_time_ms = current_time_ms - MAX_LOOKBACK_MS

        if checkpoint is not None:
            min_trace_timestamp_ms = max(checkpoint.timestamp_ms, min_lookback_time_ms)
        else:
            min_trace_timestamp_ms = min_lookback_time_ms

        return OnlineTraceScoringTimeWindow(
            min_trace_timestamp_ms=min_trace_timestamp_ms,
            max_trace_timestamp_ms=current_time_ms,
        )
