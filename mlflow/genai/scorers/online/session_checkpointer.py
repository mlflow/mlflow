"""Checkpoint management for session-level online scoring."""

import json
import logging
import time
from dataclasses import asdict, dataclass

from mlflow.entities.experiment_tag import ExperimentTag
from mlflow.environment_variables import (
    MLFLOW_ONLINE_SCORING_DEFAULT_SESSION_COMPLETION_BUFFER_SECONDS,
)
from mlflow.genai.scorers.online.constants import MAX_LOOKBACK_MS
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.utils.mlflow_tags import MLFLOW_LATEST_ONLINE_SCORING_SESSION_CHECKPOINT

_logger = logging.getLogger(__name__)


@dataclass
class OnlineSessionScoringCheckpoint:
    timestamp_ms: int  # Last trace timestamp of the last processed session
    session_id: str | None = None  # Session ID for tiebreaking when timestamps match

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> "OnlineSessionScoringCheckpoint":
        data = json.loads(json_str)
        return cls(**data)


@dataclass
class OnlineSessionScoringTimeWindow:
    min_last_trace_timestamp_ms: int
    max_last_trace_timestamp_ms: int


class OnlineSessionCheckpointManager:
    def __init__(self, tracking_store: AbstractStore, experiment_id: str):
        self._tracking_store = tracking_store
        self._experiment_id = experiment_id

    def get_checkpoint(self) -> OnlineSessionScoringCheckpoint | None:
        """
        Get the last processed session checkpoint from the experiment tag.

        Returns:
            OnlineSessionScoringCheckpoint, or None if no checkpoint exists.
        """
        experiment = self._tracking_store.get_experiment(self._experiment_id)
        if checkpoint_str := experiment.tags.get(MLFLOW_LATEST_ONLINE_SCORING_SESSION_CHECKPOINT):
            try:
                return OnlineSessionScoringCheckpoint.from_json(checkpoint_str)
            except (TypeError, ValueError, json.JSONDecodeError) as e:
                _logger.debug(
                    f"Failed to parse checkpoint for experiment {self._experiment_id}: {e}",
                    exc_info=True,
                )
        return None

    def persist_checkpoint(self, checkpoint: OnlineSessionScoringCheckpoint) -> None:
        """
        Persist the checkpoint tag with a new checkpoint.

        Args:
            checkpoint: The checkpoint to store.
        """
        self._tracking_store.set_experiment_tag(
            self._experiment_id,
            ExperimentTag(MLFLOW_LATEST_ONLINE_SCORING_SESSION_CHECKPOINT, checkpoint.to_json()),
        )

    def calculate_time_window(self) -> OnlineSessionScoringTimeWindow:
        """
        Calculate the time window for session scoring.

        Enforces a maximum lookback period to prevent getting stuck on persistently
        failing sessions. If the checkpoint is older than MAX_LOOKBACK_MS, uses
        current_time - MAX_LOOKBACK_MS instead to skip over old problematic sessions.

        Returns:
            OnlineSessionScoringTimeWindow with min and max last trace timestamps.
            min_last_trace_timestamp_ms is the checkpoint if it exists and is within
            the lookback period, otherwise now - MAX_LOOKBACK_MS.
            max_last_trace_timestamp_ms is current time - session completion buffer.
        """
        current_time_ms = int(time.time() * 1000)
        checkpoint = self.get_checkpoint()

        # Start from checkpoint, but never look back more than MAX_LOOKBACK_MS
        min_lookback_time_ms = current_time_ms - MAX_LOOKBACK_MS
        min_last_trace_timestamp_ms = max(
            checkpoint.timestamp_ms if checkpoint else 0, min_lookback_time_ms
        )

        buffer_seconds = max(
            0, MLFLOW_ONLINE_SCORING_DEFAULT_SESSION_COMPLETION_BUFFER_SECONDS.get()
        )
        max_last_trace_timestamp_ms = current_time_ms - buffer_seconds * 1000

        return OnlineSessionScoringTimeWindow(
            min_last_trace_timestamp_ms=min_last_trace_timestamp_ms,
            max_last_trace_timestamp_ms=max_last_trace_timestamp_ms,
        )
