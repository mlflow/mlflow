"""System metrics logging module."""

import os


def disable_system_metrics_logging():
    """Disable system metrics logging."""
    os.environ["MLFLOW_DISABLE_SYSTEM_METRICS_LOGGING"] = "True"


def set_system_metrics_sampling_interval(interval):
    """Set the system metrics sampling interval.

    Every `interval` seconds, the system metrics will be collected. By default `interval=10`.
    """
    os.environ["MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL"] = str(interval)


def set_system_metrics_samples_before_logging(samples):
    """Set the number of samples before logging system metrics.

    Every time `samples` samples have been collected, the system metrics will be logged to mlflow.
    By default `samples=1`.
    """
    os.environ["MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING"] = str(samples)
