"""
Idle monitor for Databricks serverless environments to prevent Py4J connection issues.

In Databricks serverless, Py4J connections are closed after 60 seconds of inactivity.
This module monitors notebook activity and proactively cleans up trace references
before the timeout to allow proper connection cleanup.
"""

import atexit
import logging
import threading
import time
from typing import Optional

from mlflow.environment_variables import (
    MLFLOW_ENABLE_SERVERLESS_IDLE_TRACE_CLEANUP,
    MLFLOW_SERVERLESS_IDLE_CLEANUP_THRESHOLD,
)

_logger = logging.getLogger(__name__)

# Global activity tracker
_last_activity_time: float = time.time()
_activity_lock = threading.RLock()


def update_activity() -> None:
    """Record that MLflow activity just occurred. Thread-safe."""
    global _last_activity_time
    with _activity_lock:
        _last_activity_time = time.time()


def get_idle_time() -> float:
    """Get seconds since last recorded activity. Thread-safe."""
    with _activity_lock:
        return time.time() - _last_activity_time


class IdleTraceMonitor:
    """
    Monitors notebook idle time and cleans up traces before Py4J timeout.

    This monitor runs only in Databricks serverless environments and helps prevent
    Py4J connection issues by proactively flushing traces when the notebook is idle.
    """

    _instance: Optional["IdleTraceMonitor"] = None
    _instance_lock = threading.RLock()

    def __init__(
        self,
        idle_threshold: int = 50,  # Flush after 50s idle (before 60s Py4J timeout)
        check_interval: int = 10,  # Check every 10 seconds
    ):
        """
        Initialize idle monitor.

        Args:
            idle_threshold: Seconds of idle time before triggering cleanup (default: 50)
            check_interval: Seconds between idle checks (default: 10)
        """
        self._idle_threshold = idle_threshold
        self._check_interval = check_interval
        self._stop_event = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self._is_active = False
        self._lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> "IdleTraceMonitor":
        """Get or create singleton instance."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton instance. For testing only."""
        if cls._instance:
            cls._instance.stop()
        with cls._instance_lock:
            cls._instance = None

    def start(self) -> None:
        """Start the idle monitoring thread."""
        with self._lock:
            if self._is_active:
                _logger.debug("Idle monitor already active")
                return

            self._stop_event.clear()
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                name="MLflowIdleTraceMonitor",
                daemon=True,
            )
            self._monitor_thread.start()
            self._is_active = True

            # Register cleanup on exit
            atexit.register(self.stop)

            _logger.info(
                f"Started idle trace monitor (threshold: {self._idle_threshold}s, "
                f"check interval: {self._check_interval}s)"
            )

    def stop(self) -> None:
        """Stop the idle monitoring thread."""
        with self._lock:
            if not self._is_active:
                return

            self._stop_event.set()
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=5)
            self._is_active = False
            _logger.debug("Stopped idle trace monitor")

    def is_active(self) -> bool:
        """Check if the monitor is currently active."""
        with self._lock:
            return self._is_active

    def _monitor_loop(self) -> None:
        """Main monitoring loop that checks for idle time and flushes traces."""
        while not self._stop_event.is_set():
            try:
                time.sleep(self._check_interval)

                idle_time = get_idle_time()
                if idle_time >= self._idle_threshold:
                    self._flush_traces_on_idle(idle_time)

            except Exception as e:
                _logger.debug(f"Error in idle monitor loop: {e}", exc_info=True)

    def _flush_traces_on_idle(self, idle_time: float) -> None:
        """
        Flush completed traces when notebook is idle to release references.

        Args:
            idle_time: Current idle time in seconds
        """
        try:
            from mlflow.tracing.trace_manager import InMemoryTraceManager

            manager = InMemoryTraceManager.get_instance()

            # Get count before flush
            with manager._lock:
                trace_count = len(manager._traces)

            if trace_count > 0:
                # Clear completed traces from the cache
                # Note: This doesn't interrupt in-progress traces, just releases
                # references to completed ones that are sitting in the cache
                with manager._lock:
                    # Create a list of trace IDs to remove (completed traces)
                    traces_to_remove = []
                    for trace_id, trace in list(manager._traces.items()):
                        # Check if trace has a root span that's ended
                        root_span = trace.get_root_span()
                        if root_span and hasattr(root_span, "_ended") and root_span._ended:
                            traces_to_remove.append(trace_id)

                    # Remove completed traces
                    for trace_id in traces_to_remove:
                        manager._traces.pop(trace_id, None)

                    if traces_to_remove:
                        _logger.info(
                            f"Flushed {len(traces_to_remove)} completed trace(s) after "
                            f"{idle_time:.1f}s of idle time to prevent Py4J connection issues"
                        )

        except Exception as e:
            _logger.debug(f"Failed to flush traces on idle: {e}", exc_info=True)


def start_idle_monitor_if_needed() -> None:
    """
    Start idle monitor in Databricks serverless if enabled.

    This should be called once during tracing initialization.
    """
    if not MLFLOW_ENABLE_SERVERLESS_IDLE_TRACE_CLEANUP.get():
        return

    try:
        from mlflow.utils.databricks_utils import is_in_databricks_serverless_runtime

        if not is_in_databricks_serverless_runtime():
            return

        # Start the idle monitor
        threshold = MLFLOW_SERVERLESS_IDLE_CLEANUP_THRESHOLD.get()
        monitor = IdleTraceMonitor.get_instance()
        monitor._idle_threshold = threshold
        monitor.start()

    except Exception as e:
        _logger.debug(f"Failed to start idle monitor: {e}", exc_info=True)
