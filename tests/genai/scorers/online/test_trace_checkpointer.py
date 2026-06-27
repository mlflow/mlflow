import time
from unittest.mock import MagicMock

import pytest

from mlflow.environment_variables import (
    MLFLOW_ONLINE_SCORING_DEFAULT_TRACE_COMPLETION_BUFFER_SECONDS,
)
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
    monkeypatch.setenv(
        MLFLOW_ONLINE_SCORING_DEFAULT_TRACE_COMPLETION_BUFFER_SECONDS.name, "0"
    )

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
    monkeypatch.setenv(
        MLFLOW_ONLINE_SCORING_DEFAULT_TRACE_COMPLETION_BUFFER_SECONDS.name, "0"
    )

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
    monkeypatch.setenv(
        MLFLOW_ONLINE_SCORING_DEFAULT_TRACE_COMPLETION_BUFFER_SECONDS.name, "0"
    )

    result = checkpoint_manager.calculate_time_window()

    expected_min = (fixed_time * 1000) - MAX_LOOKBACK_MS
    assert result.min_trace_timestamp_ms == expected_min
    assert result.max_trace_timestamp_ms == fixed_time * 1000


def test_calculate_time_window_with_default_buffer(checkpoint_manager, mock_store, monkeypatch):
    """Test that the default buffer (300s) is subtracted from max_trace_timestamp_ms."""
    experiment = MagicMock()
    experiment.tags = {}
    mock_store.get_experiment.return_value = experiment
    fixed_time = 1000000
    monkeypatch.setattr(time, "time", lambda: fixed_time)
    # Don't override the env var — use the default buffer (300s)

    result = checkpoint_manager.calculate_time_window()

    expected_min = (fixed_time * 1000) - MAX_LOOKBACK_MS
    default_buffer_ms = 300 * 1000  # 5 minutes in ms
    expected_max = (fixed_time * 1000) - default_buffer_ms
    assert result.min_trace_timestamp_ms == expected_min
    assert result.max_trace_timestamp_ms == expected_max


def test_calculate_time_window_with_custom_buffer(checkpoint_manager, mock_store, monkeypatch):
    """Test that a custom buffer value is correctly applied."""
    experiment = MagicMock()
    experiment.tags = {}
    mock_store.get_experiment.return_value = experiment
    fixed_time = 1000000
    monkeypatch.setattr(time, "time", lambda: fixed_time)
    monkeypatch.setenv(
        MLFLOW_ONLINE_SCORING_DEFAULT_TRACE_COMPLETION_BUFFER_SECONDS.name, "120"
    )

    result = checkpoint_manager.calculate_time_window()

    expected_min = (fixed_time * 1000) - MAX_LOOKBACK_MS
    custom_buffer_ms = 120 * 1000  # 2 minutes in ms
    expected_max = (fixed_time * 1000) - custom_buffer_ms
    assert result.min_trace_timestamp_ms == expected_min
    assert result.max_trace_timestamp_ms == expected_max


def test_calculate_time_window_buffer_gives_long_traces_time_to_complete(
    checkpoint_manager, mock_store, monkeypatch
):
    """
    Regression test for https://github.com/mlflow/mlflow/issues/21870

    When the trace completion buffer is applied, traces that started within the
    buffer period before the current time are excluded from the scoring window.
    This prevents long-running traces from being permanently skipped.
    """
    fixed_time = 1000000
    # Simulate a checkpoint from a previous scan
    previous_checkpoint_time = (fixed_time * 1000) - 120_000  # 2 minutes ago
    experiment = MagicMock()
    experiment.tags = {
        MLFLOW_LATEST_ONLINE_SCORING_TRACE_CHECKPOINT: (
            f'{{"timestamp_ms": {previous_checkpoint_time}}}'
        )
    }
    mock_store.get_experiment.return_value = experiment
    monkeypatch.setattr(time, "time", lambda: fixed_time)
    # Use a 60-second buffer
    monkeypatch.setenv(
        MLFLOW_ONLINE_SCORING_DEFAULT_TRACE_COMPLETION_BUFFER_SECONDS.name, "60"
    )

    result = checkpoint_manager.calculate_time_window()

    # The window should start from the checkpoint
    assert result.min_trace_timestamp_ms == previous_checkpoint_time
    # The upper bound should be 60 seconds before current time, giving traces
    # that started in the last 60 seconds time to complete
    expected_max = (fixed_time * 1000) - 60_000
    assert result.max_trace_timestamp_ms == expected_max


def test_calculate_time_window_negative_buffer_treated_as_zero(
    checkpoint_manager, mock_store, monkeypatch
):
    """Test that a negative buffer value is clamped to zero."""
    experiment = MagicMock()
    experiment.tags = {}
    mock_store.get_experiment.return_value = experiment
    fixed_time = 1000000
    monkeypatch.setattr(time, "time", lambda: fixed_time)
    monkeypatch.setenv(
        MLFLOW_ONLINE_SCORING_DEFAULT_TRACE_COMPLETION_BUFFER_SECONDS.name, "-10"
    )

    result = checkpoint_manager.calculate_time_window()

    # Negative values should be clamped to 0, so max should be current time
    assert result.max_trace_timestamp_ms == fixed_time * 1000


def test_calculate_time_window_checkpoint_ahead_of_buffered_max_clamps_window(
    checkpoint_manager, mock_store, monkeypatch
):
    """
    Test that when an existing checkpoint is ahead of the buffered upper bound
    (e.g., after upgrading from pre-buffer logic), the window is clamped so that
    max >= min. This prevents inverted windows and checkpoint rewind.
    """
    fixed_time = 1000000
    # Checkpoint is very recent — just 10 seconds ago (well within the 300s buffer)
    recent_checkpoint_time = (fixed_time * 1000) - 10_000  # 10 seconds ago
    experiment = MagicMock()
    experiment.tags = {
        MLFLOW_LATEST_ONLINE_SCORING_TRACE_CHECKPOINT: (
            f'{{"timestamp_ms": {recent_checkpoint_time}}}'
        )
    }
    mock_store.get_experiment.return_value = experiment
    monkeypatch.setattr(time, "time", lambda: fixed_time)
    # Default buffer of 300s means max would be current_time - 300s, which is
    # before the checkpoint. Without clamping, min > max (inverted window).

    result = checkpoint_manager.calculate_time_window()

    # min should be the checkpoint
    assert result.min_trace_timestamp_ms == recent_checkpoint_time
    # max should be clamped to at least min (not rewound before checkpoint)
    assert result.max_trace_timestamp_ms >= result.min_trace_timestamp_ms
    assert result.max_trace_timestamp_ms == recent_checkpoint_time
