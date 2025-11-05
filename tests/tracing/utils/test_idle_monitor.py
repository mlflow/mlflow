import time
from unittest import mock

import pytest

from mlflow.tracing.utils.idle_monitor import (
    IdleTraceMonitor,
    get_idle_time,
    update_activity,
)


@pytest.fixture(autouse=True)
def reset_monitor():
    """Reset the singleton instance before each test."""
    IdleTraceMonitor.reset()
    yield
    IdleTraceMonitor.reset()


def test_update_activity_and_get_idle_time():
    """Test basic activity tracking."""
    update_activity()
    time.sleep(0.1)
    idle_time = get_idle_time()
    assert 0.09 < idle_time < 0.2  # Some margin for test execution


def test_idle_monitor_starts_and_stops():
    monitor = IdleTraceMonitor.get_instance()
    assert not monitor.is_active()

    monitor.start()
    assert monitor.is_active()

    monitor.stop()
    assert not monitor.is_active()


def test_idle_monitor_singleton():
    """Test that get_instance returns the same instance."""
    monitor1 = IdleTraceMonitor.get_instance()
    monitor2 = IdleTraceMonitor.get_instance()
    assert monitor1 is monitor2


def test_idle_monitor_flushes_completed_traces(monkeypatch):
    """Test that idle monitor flushes completed traces after threshold."""
    from mlflow.tracing.trace_manager import InMemoryTraceManager, _Trace

    # Create mock spans
    mock_root_span = mock.Mock()
    mock_root_span.end_time_ns = 1234567890  # Mark as ended (has end time)
    mock_root_span.span_id = "span_1"
    mock_root_span.parent_id = None

    # Set up trace manager with a completed trace
    manager = InMemoryTraceManager.get_instance()
    trace = _Trace(None, span_dict={"span_1": mock_root_span})
    trace._root_span_id = "span_1"
    manager._traces["trace_1"] = trace

    # Create monitor with short threshold for testing
    monitor = IdleTraceMonitor(idle_threshold=0.5, check_interval=0.2)

    try:
        # Set last activity to >threshold ago
        monkeypatch.setattr(
            "mlflow.tracing.utils.idle_monitor._last_activity_time",
            time.time() - 1.0,  # 1 second ago
        )

        monitor.start()

        # Wait for monitor to detect and flush
        time.sleep(0.8)

        # Verify trace was removed
        assert "trace_1" not in manager._traces

    finally:
        monitor.stop()


def test_idle_monitor_does_not_flush_in_progress_traces(monkeypatch):
    """Test that idle monitor does not flush traces that are still in progress."""
    from mlflow.tracing.trace_manager import InMemoryTraceManager, _Trace

    # Create mock span that's NOT ended
    mock_root_span = mock.Mock()
    mock_root_span.end_time_ns = None  # Still in progress (no end time)
    mock_root_span.span_id = "span_1"
    mock_root_span.parent_id = None

    # Set up trace manager with an in-progress trace
    manager = InMemoryTraceManager.get_instance()
    trace = _Trace(None, span_dict={"span_1": mock_root_span})
    trace._root_span_id = "span_1"
    manager._traces["trace_1"] = trace

    # Create monitor with short threshold
    monitor = IdleTraceMonitor(idle_threshold=0.5, check_interval=0.2)

    try:
        # Set last activity to >threshold ago
        monkeypatch.setattr(
            "mlflow.tracing.utils.idle_monitor._last_activity_time",
            time.time() - 1.0,
        )

        monitor.start()

        # Wait for monitor check
        time.sleep(0.8)

        # Verify trace was NOT removed (still in progress)
        assert "trace_1" in manager._traces

    finally:
        monitor.stop()


def test_start_idle_monitor_if_needed_in_serverless(monkeypatch):
    """Test that idle monitor starts automatically in serverless."""
    from mlflow.tracing.utils.idle_monitor import start_idle_monitor_if_needed

    monkeypatch.setenv("MLFLOW_ENABLE_SERVERLESS_IDLE_TRACE_CLEANUP", "true")

    with mock.patch(
        "mlflow.utils.databricks_utils.is_in_databricks_serverless_runtime",
        return_value=True,
    ):
        start_idle_monitor_if_needed()

        monitor = IdleTraceMonitor.get_instance()
        assert monitor.is_active()


def test_start_idle_monitor_if_needed_not_in_serverless(monkeypatch):
    """Test that idle monitor does not start outside serverless."""
    from mlflow.tracing.utils.idle_monitor import start_idle_monitor_if_needed

    monkeypatch.setenv("MLFLOW_ENABLE_SERVERLESS_IDLE_TRACE_CLEANUP", "true")

    with mock.patch(
        "mlflow.utils.databricks_utils.is_in_databricks_serverless_runtime",
        return_value=False,
    ):
        start_idle_monitor_if_needed()

        monitor = IdleTraceMonitor.get_instance()
        assert not monitor.is_active()


def test_start_idle_monitor_respects_disable_flag(monkeypatch):
    from mlflow.tracing.utils.idle_monitor import start_idle_monitor_if_needed

    monkeypatch.setenv("MLFLOW_ENABLE_SERVERLESS_IDLE_TRACE_CLEANUP", "false")

    with mock.patch(
        "mlflow.utils.databricks_utils.is_in_databricks_serverless_runtime",
        return_value=True,
    ):
        start_idle_monitor_if_needed()

        monitor = IdleTraceMonitor.get_instance()
        assert not monitor.is_active()


def test_idle_monitor_custom_threshold(monkeypatch):
    """Test that idle monitor respects custom threshold setting."""
    from mlflow.tracing.utils.idle_monitor import start_idle_monitor_if_needed

    monkeypatch.setenv("MLFLOW_ENABLE_SERVERLESS_IDLE_TRACE_CLEANUP", "true")
    monkeypatch.setenv("MLFLOW_SERVERLESS_IDLE_CLEANUP_THRESHOLD", "30")

    with mock.patch(
        "mlflow.utils.databricks_utils.is_in_databricks_serverless_runtime",
        return_value=True,
    ):
        start_idle_monitor_if_needed()

        monitor = IdleTraceMonitor.get_instance()
        assert monitor.is_active()
        assert monitor._idle_threshold == 30
