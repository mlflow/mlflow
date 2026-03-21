import time
from unittest.mock import MagicMock

import pytest

from mlflow.genai.scorers.online.constants import MAX_LOOKBACK_MS
from mlflow.genai.scorers.online.trace_checkpointer import (
    OnlineTraceCheckpointManager,
    OnlineTraceScoringCheckpoint,
)
from mlflow.utils.mlflow_tags import MLFLOW_LATEST_ONLINE_SCORING_TRACE_CHECKPOINT


@pytest.fixture
def mock_store():
    return MagicMock()


@pytest.fixture
def checkpoint_manager(mock_store):
    return OnlineTraceCheckpointManager(mock_store, "exp1")


def test_checkpoint_json_roundtrip():
    original = OnlineTraceScoringCheckpoint(timestamp_ms=5000, trace_id="tr-abc")

    restored = OnlineTraceScoringCheckpoint.from_json(original.to_json())

    assert restored.timestamp_ms == original.timestamp_ms
    assert restored.trace_id == original.trace_id


def test_get_checkpoint_returns_none_when_no_tag(checkpoint_manager, mock_store):
    experiment = MagicMock()
    experiment.tags = {}
    mock_store.get_experiment.return_value = experiment

    result = checkpoint_manager.get_checkpoint()

    assert result is None


def test_get_checkpoint_deserializes_correctly(checkpoint_manager, mock_store):
    experiment = MagicMock()
    experiment.tags = {
        MLFLOW_LATEST_ONLINE_SCORING_TRACE_CHECKPOINT: '{"timestamp_ms": 1000, "trace_id": "tr-1"}'
    }
    mock_store.get_experiment.return_value = experiment

    result = checkpoint_manager.get_checkpoint()

    assert result.timestamp_ms == 1000
    assert result.trace_id == "tr-1"


def test_get_checkpoint_handles_invalid_json(checkpoint_manager, mock_store):
    experiment = MagicMock()
    experiment.tags = {MLFLOW_LATEST_ONLINE_SCORING_TRACE_CHECKPOINT: "invalid json"}
    mock_store.get_experiment.return_value = experiment

    result = checkpoint_manager.get_checkpoint()

    assert result is None


def test_persist_checkpoint_sets_experiment_tag(checkpoint_manager, mock_store):
    checkpoint = OnlineTraceScoringCheckpoint(timestamp_ms=2000, trace_id="tr-2")

    checkpoint_manager.persist_checkpoint(checkpoint)

    mock_store.set_experiment_tag.assert_called_once()
    call_args = mock_store.set_experiment_tag.call_args
    assert call_args[0][0] == "exp1"
    assert call_args[0][1].key == MLFLOW_LATEST_ONLINE_SCORING_TRACE_CHECKPOINT


def test_calculate_time_window_no_checkpoint(checkpoint_manager, mock_store, monkeypatch):
    experiment = MagicMock()
    experiment.tags = {}
    mock_store.get_experiment.return_value = experiment
    fixed_time = 1000000
    monkeypatch.setattr(time, "time", lambda: fixed_time)

    result = checkpoint_manager.calculate_time_window()

    expected_min = (fixed_time * 1000) - MAX_LOOKBACK_MS
    assert result.min_trace_timestamp_ms == expected_min
    assert result.max_trace_timestamp_ms == fixed_time * 1000


def test_calculate_time_window_recent_checkpoint(checkpoint_manager, mock_store, monkeypatch):
    fixed_time = 1000000
    recent_checkpoint_time = (fixed_time * 1000) - 60000  # 1 minute ago
    experiment = MagicMock()
    checkpoint_json = f'{{"timestamp_ms": {recent_checkpoint_time}}}'
    experiment.tags = {MLFLOW_LATEST_ONLINE_SCORING_TRACE_CHECKPOINT: checkpoint_json}
    mock_store.get_experiment.return_value = experiment
    monkeypatch.setattr(time, "time", lambda: fixed_time)

    result = checkpoint_manager.calculate_time_window()

    assert result.min_trace_timestamp_ms == recent_checkpoint_time
    assert result.max_trace_timestamp_ms == fixed_time * 1000


def test_calculate_time_window_old_checkpoint(checkpoint_manager, mock_store, monkeypatch):
    fixed_time = 1000000
    old_checkpoint_time = (
        (fixed_time * 1000) - MAX_LOOKBACK_MS - 1000000
    )  # Way older than max lookback
    experiment = MagicMock()
    experiment.tags = {
        MLFLOW_LATEST_ONLINE_SCORING_TRACE_CHECKPOINT: f'{{"timestamp_ms": {old_checkpoint_time}}}'
    }
    mock_store.get_experiment.return_value = experiment
    monkeypatch.setattr(time, "time", lambda: fixed_time)

    result = checkpoint_manager.calculate_time_window()

    expected_min = (fixed_time * 1000) - MAX_LOOKBACK_MS
    assert result.min_trace_timestamp_ms == expected_min
    assert result.max_trace_timestamp_ms == fixed_time * 1000
