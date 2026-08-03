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

    result = checkpoint_manager.calculate_time_window()

    expected_min = (fixed_time * 1000) - MAX_LOOKBACK_MS
    default_buffer_ms = MLFLOW_ONLINE_SCORING_DEFAULT_TRACE_COMPLETION_BUFFER_SECONDS.get() * 1000
    expected_max = (fixed_time * 1000) - default_buffer_ms
    assert result.min_trace_timestamp_ms == expected_min
    assert result.max_trace_timestamp_ms == expected_max


def test_calculate_time_window_recent_checkpoint(checkpoint_manager, mock_store, monkeypatch):
    fixed_time = 1000000
    recent_checkpoint_time = (fixed_time * 1000) - 300000  # 5 minutes ago, older than the buffer
    experiment = MagicMock()
    checkpoint_json = f'{{"timestamp_ms": {recent_checkpoint_time}}}'
    experiment.tags = {MLFLOW_LATEST_ONLINE_SCORING_TRACE_CHECKPOINT: checkpoint_json}
    mock_store.get_experiment.return_value = experiment
    monkeypatch.setattr(time, "time", lambda: fixed_time)

    result = checkpoint_manager.calculate_time_window()

    default_buffer_ms = MLFLOW_ONLINE_SCORING_DEFAULT_TRACE_COMPLETION_BUFFER_SECONDS.get() * 1000
    expected_max = (fixed_time * 1000) - default_buffer_ms
    assert result.min_trace_timestamp_ms == recent_checkpoint_time
    assert result.max_trace_timestamp_ms == expected_max


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
    default_buffer_ms = MLFLOW_ONLINE_SCORING_DEFAULT_TRACE_COMPLETION_BUFFER_SECONDS.get() * 1000
    expected_max = (fixed_time * 1000) - default_buffer_ms
    assert result.min_trace_timestamp_ms == expected_min
    assert result.max_trace_timestamp_ms == expected_max


def test_calculate_time_window_with_custom_buffer(checkpoint_manager, mock_store, monkeypatch):
    # Empty tags simulates no existing checkpoint
    experiment = MagicMock()
    experiment.tags = {}
    mock_store.get_experiment.return_value = experiment
    fixed_time = 1000000
    custom_buffer_seconds = 30
    monkeypatch.setattr(time, "time", lambda: fixed_time)
    monkeypatch.setenv(
        "MLFLOW_ONLINE_SCORING_DEFAULT_TRACE_COMPLETION_BUFFER_SECONDS",
        str(custom_buffer_seconds),
    )

    result = checkpoint_manager.calculate_time_window()

    expected_min = (fixed_time * 1000) - MAX_LOOKBACK_MS
    expected_max = (fixed_time * 1000) - (custom_buffer_seconds * 1000)
    assert result.min_trace_timestamp_ms == expected_min
    assert result.max_trace_timestamp_ms == expected_max


def test_calculate_time_window_with_negative_buffer_defaults_to_zero(
    checkpoint_manager, mock_store, monkeypatch
):
    # Empty tags simulates no existing checkpoint
    experiment = MagicMock()
    experiment.tags = {}
    mock_store.get_experiment.return_value = experiment
    fixed_time = 1000000
    monkeypatch.setattr(time, "time", lambda: fixed_time)
    monkeypatch.setenv("MLFLOW_ONLINE_SCORING_DEFAULT_TRACE_COMPLETION_BUFFER_SECONDS", "-100")

    result = checkpoint_manager.calculate_time_window()

    expected_min = (fixed_time * 1000) - MAX_LOOKBACK_MS
    expected_max = fixed_time * 1000  # buffer is 0, so max = current_time
    assert result.min_trace_timestamp_ms == expected_min
    assert result.max_trace_timestamp_ms == expected_max


def test_calculate_time_window_trace_still_in_progress_is_excluded_until_buffer_elapses(
    checkpoint_manager, mock_store, monkeypatch
):
    """
    Regression test for the race condition in issue #21870: a trace that starts near
    the end of a scan window and is still IN_PROGRESS when that window is scanned
    must not fall outside the window on the very next scan, since the scheduler's
    checkpoint only ever moves forward and never revisits a skipped window.
    """
    experiment = MagicMock()
    experiment.tags = {}
    mock_store.get_experiment.return_value = experiment
    fixed_time = 1000000
    monkeypatch.setattr(time, "time", lambda: fixed_time)

    # A trace that started 10 seconds before "now" and has not finished yet.
    trace_start_ms = (fixed_time * 1000) - 10_000

    result = checkpoint_manager.calculate_time_window()

    # With the default 2-minute buffer, the window's upper bound sits well before the
    # trace's start time, so this still-running trace is correctly excluded from the
    # current scan rather than being silently skipped once it completes.
    assert trace_start_ms > result.max_trace_timestamp_ms


def test_calculate_time_window_checkpoint_within_buffer_does_not_invert_window(
    checkpoint_manager, mock_store, monkeypatch
):
    """
    A checkpoint more recent than (now - buffer), e.g. one written before the buffer
    was introduced, must not produce an inverted window. The upper bound is clamped
    to the checkpoint so persisting it never moves the checkpoint backward, which
    would re-score already-processed traces.
    """
    fixed_time = 1000000
    checkpoint_time = (fixed_time * 1000) - 60000  # 1 minute ago, within the 2-minute buffer
    experiment = MagicMock()
    checkpoint_json = f'{{"timestamp_ms": {checkpoint_time}}}'
    experiment.tags = {MLFLOW_LATEST_ONLINE_SCORING_TRACE_CHECKPOINT: checkpoint_json}
    mock_store.get_experiment.return_value = experiment
    monkeypatch.setattr(time, "time", lambda: fixed_time)

    result = checkpoint_manager.calculate_time_window()

    assert result.min_trace_timestamp_ms == checkpoint_time
    assert result.max_trace_timestamp_ms == checkpoint_time
    assert result.min_trace_timestamp_ms <= result.max_trace_timestamp_ms
