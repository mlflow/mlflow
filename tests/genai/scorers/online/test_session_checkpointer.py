import time
from unittest.mock import MagicMock

import pytest

from mlflow.environment_variables import (
    MLFLOW_ONLINE_SCORING_DEFAULT_SESSION_COMPLETION_BUFFER_SECONDS,
)
from mlflow.genai.scorers.online.constants import MAX_LOOKBACK_MS
from mlflow.genai.scorers.online.session_checkpointer import (
    OnlineSessionCheckpointManager,
    OnlineSessionScoringCheckpoint,
)
from mlflow.utils.mlflow_tags import MLFLOW_LATEST_ONLINE_SCORING_SESSION_CHECKPOINT


@pytest.fixture
def mock_store():
    return MagicMock()


@pytest.fixture
def checkpoint_manager(mock_store):
    return OnlineSessionCheckpointManager(mock_store, "exp1")


def test_checkpoint_json_roundtrip():
    original = OnlineSessionScoringCheckpoint(timestamp_ms=5000, session_id="sess-abc")

    restored = OnlineSessionScoringCheckpoint.from_json(original.to_json())

    assert restored.timestamp_ms == original.timestamp_ms
    assert restored.session_id == original.session_id


def test_get_checkpoint_returns_none_when_no_tag(checkpoint_manager, mock_store):
    experiment = MagicMock()
    experiment.tags = {}
    mock_store.get_experiment.return_value = experiment

    result = checkpoint_manager.get_checkpoint()

    assert result is None


def test_get_checkpoint_deserializes_correctly(checkpoint_manager, mock_store):
    experiment = MagicMock()
    checkpoint_json = '{"timestamp_ms": 1000, "session_id": "sess-1"}'
    experiment.tags = {MLFLOW_LATEST_ONLINE_SCORING_SESSION_CHECKPOINT: checkpoint_json}
    mock_store.get_experiment.return_value = experiment

    result = checkpoint_manager.get_checkpoint()

    assert result.timestamp_ms == 1000
    assert result.session_id == "sess-1"


def test_get_checkpoint_handles_invalid_json(checkpoint_manager, mock_store):
    experiment = MagicMock()
    experiment.tags = {MLFLOW_LATEST_ONLINE_SCORING_SESSION_CHECKPOINT: "invalid json"}
    mock_store.get_experiment.return_value = experiment

    result = checkpoint_manager.get_checkpoint()

    assert result is None


def test_persist_checkpoint_sets_experiment_tag(checkpoint_manager, mock_store):
    checkpoint = OnlineSessionScoringCheckpoint(timestamp_ms=2000, session_id="sess-2")

    checkpoint_manager.persist_checkpoint(checkpoint)

    mock_store.set_experiment_tag.assert_called_once()
    call_args = mock_store.set_experiment_tag.call_args
    assert call_args[0][0] == "exp1"
    assert call_args[0][1].key == MLFLOW_LATEST_ONLINE_SCORING_SESSION_CHECKPOINT


def test_calculate_time_window_no_checkpoint(checkpoint_manager, mock_store, monkeypatch):
    experiment = MagicMock()
    experiment.tags = {}
    mock_store.get_experiment.return_value = experiment
    fixed_time = 1000000
    monkeypatch.setattr(time, "time", lambda: fixed_time)

    result = checkpoint_manager.calculate_time_window()

    expected_min = (fixed_time * 1000) - MAX_LOOKBACK_MS
    expected_max = (
        fixed_time * 1000
    ) - MLFLOW_ONLINE_SCORING_DEFAULT_SESSION_COMPLETION_BUFFER_SECONDS.get() * 1000
    assert result.min_last_trace_timestamp_ms == expected_min
    assert result.max_last_trace_timestamp_ms == expected_max


def test_calculate_time_window_recent_checkpoint(checkpoint_manager, mock_store, monkeypatch):
    fixed_time = 1000000
    recent_checkpoint_time = (fixed_time * 1000) - 60000
    experiment = MagicMock()
    checkpoint_json = f'{{"timestamp_ms": {recent_checkpoint_time}}}'
    experiment.tags = {MLFLOW_LATEST_ONLINE_SCORING_SESSION_CHECKPOINT: checkpoint_json}
    mock_store.get_experiment.return_value = experiment
    monkeypatch.setattr(time, "time", lambda: fixed_time)

    result = checkpoint_manager.calculate_time_window()

    expected_max = (
        fixed_time * 1000
    ) - MLFLOW_ONLINE_SCORING_DEFAULT_SESSION_COMPLETION_BUFFER_SECONDS.get() * 1000
    assert result.min_last_trace_timestamp_ms == recent_checkpoint_time
    assert result.max_last_trace_timestamp_ms == expected_max


def test_calculate_time_window_old_checkpoint(checkpoint_manager, mock_store, monkeypatch):
    fixed_time = 1000000
    old_checkpoint_time = (fixed_time * 1000) - MAX_LOOKBACK_MS - 1000000
    experiment = MagicMock()
    checkpoint_json = f'{{"timestamp_ms": {old_checkpoint_time}}}'
    experiment.tags = {MLFLOW_LATEST_ONLINE_SCORING_SESSION_CHECKPOINT: checkpoint_json}
    mock_store.get_experiment.return_value = experiment
    monkeypatch.setattr(time, "time", lambda: fixed_time)

    result = checkpoint_manager.calculate_time_window()

    expected_min = (fixed_time * 1000) - MAX_LOOKBACK_MS
    expected_max = (
        fixed_time * 1000
    ) - MLFLOW_ONLINE_SCORING_DEFAULT_SESSION_COMPLETION_BUFFER_SECONDS.get() * 1000
    assert result.min_last_trace_timestamp_ms == expected_min
    assert result.max_last_trace_timestamp_ms == expected_max


def test_calculate_time_window_with_custom_buffer(checkpoint_manager, mock_store, monkeypatch):
    experiment = MagicMock()
    experiment.tags = {}
    mock_store.get_experiment.return_value = experiment
    fixed_time = 1000000
    custom_buffer_seconds = 60  # 1 minute
    monkeypatch.setattr(time, "time", lambda: fixed_time)
    monkeypatch.setenv(
        "MLFLOW_ONLINE_SCORING_DEFAULT_SESSION_COMPLETION_BUFFER_SECONDS",
        str(custom_buffer_seconds),
    )

    result = checkpoint_manager.calculate_time_window()

    expected_min = (fixed_time * 1000) - MAX_LOOKBACK_MS
    expected_max = (fixed_time * 1000) - (custom_buffer_seconds * 1000)
    assert result.min_last_trace_timestamp_ms == expected_min
    assert result.max_last_trace_timestamp_ms == expected_max


def test_calculate_time_window_with_negative_buffer_defaults_to_zero(
    checkpoint_manager, mock_store, monkeypatch
):
    experiment = MagicMock()
    experiment.tags = {}
    mock_store.get_experiment.return_value = experiment
    fixed_time = 1000000
    monkeypatch.setattr(time, "time", lambda: fixed_time)
    monkeypatch.setenv("MLFLOW_ONLINE_SCORING_DEFAULT_SESSION_COMPLETION_BUFFER_SECONDS", "-100")

    result = checkpoint_manager.calculate_time_window()

    expected_min = (fixed_time * 1000) - MAX_LOOKBACK_MS
    expected_max = fixed_time * 1000  # buffer is 0, so max = current_time
    assert result.min_last_trace_timestamp_ms == expected_min
    assert result.max_last_trace_timestamp_ms == expected_max
