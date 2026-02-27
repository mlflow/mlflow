"""System metrics logging module."""

from mlflow.environment_variables import (
    MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING,
    MLFLOW_SYSTEM_METRICS_INCLUDE_CHILD_PROCESSES,
    MLFLOW_SYSTEM_METRICS_INCLUDE_PROCESS,
    MLFLOW_SYSTEM_METRICS_NODE_ID,
    MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING,
    MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL,
)


def disable_system_metrics_logging():
    """Disable system metrics logging globally.

    Calling this function will disable system metrics logging globally, but users can still opt in
    system metrics logging for individual runs by `mlflow.start_run(log_system_metrics=True)`.
    """
    MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING.set(False)


def enable_system_metrics_logging():
    """Enable system metrics logging globally.

    Calling this function will enable system metrics logging globally, but users can still opt out
    system metrics logging for individual runs by `mlflow.start_run(log_system_metrics=False)`.
    """
    MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING.set(True)


def set_system_metrics_sampling_interval(interval):
    """Set the system metrics sampling interval.

    Every `interval` seconds, the system metrics will be collected. By default `interval=10`.
    """
    if interval is None:
        MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL.unset()
    else:
        MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL.set(interval)


def set_system_metrics_samples_before_logging(samples):
    """Set the number of samples before logging system metrics.

    Every time `samples` samples have been collected, the system metrics will be logged to mlflow.
    By default `samples=1`.
    """
    if samples is None:
        MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING.unset()
    else:
        MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING.set(samples)


def set_system_metrics_node_id(node_id):
    """Set the system metrics node id.

    node_id is the identifier of the machine where the metrics are collected. This is useful in
    multi-node (distributed training) setup.
    """
    if node_id is None:
        MLFLOW_SYSTEM_METRICS_NODE_ID.unset()
    else:
        MLFLOW_SYSTEM_METRICS_NODE_ID.set(node_id)


def enable_process_metrics(include_children=True):
    """Enable process-level metrics logging.

    When enabled, MLflow will log CPU, memory, thread count, and file descriptor metrics
    specific to the current Python process (and optionally its child processes), rather than
    just system-wide metrics. This is particularly useful in shared environments where
    multiple processes run on the same machine.

    Args:
        include_children: If True (default), also aggregate metrics from child processes.
            This is useful for tracking resource usage of subprocesses spawned by your
            training script.

    Example:
        >>> import mlflow
        >>> mlflow.enable_system_metrics_logging()
        >>> mlflow.enable_process_metrics()  # Enable process-level tracking
        >>> with mlflow.start_run():
        ...     # Your training code here
        ...     pass
        >>> # Process metrics like system/process_cpu_percentage will be logged
    """
    MLFLOW_SYSTEM_METRICS_INCLUDE_PROCESS.set(True)
    MLFLOW_SYSTEM_METRICS_INCLUDE_CHILD_PROCESSES.set(include_children)


def disable_process_metrics():
    """Disable process-level metrics logging.

    After calling this, only system-wide metrics will be logged (the default behavior).
    """
    MLFLOW_SYSTEM_METRICS_INCLUDE_PROCESS.set(False)
