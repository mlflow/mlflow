"""System metrics logging module."""

from mlflow.environment_variables import (
    MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING,
    MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING,
    MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL,
)


def disable_system_metrics_logging():
    """Disable system metrics logging."""
    MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING.set(False)


def enable_system_metrics_logging():
    """Enable system metrics logging."""
    MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING.set(True)


def set_system_metrics_sampling_interval(interval):
    """Set the system metrics sampling interval.

    Every `interval` seconds, the system metrics will be collected. By default `interval=10`.
    """
    MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL.set(interval)


def set_system_metrics_samples_before_logging(samples):
    """Set the number of samples before logging system metrics.

    Every time `samples` samples have been collected, the system metrics will be logged to mlflow.
    By default `samples=1`.
    """
    MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING.set(samples)
