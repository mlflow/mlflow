"""Class for monitoring system stats."""

import logging
import threading

from mlflow.environment_variables import (
    MLFLOW_SYSTEM_METRICS_NODE_ID,
    MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING,
    MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL,
)
from mlflow.exceptions import MlflowException
from mlflow.system_metrics.metrics.cpu_monitor import CPUMonitor
from mlflow.system_metrics.metrics.disk_monitor import DiskMonitor
from mlflow.system_metrics.metrics.gpu_monitor import GPUMonitor
from mlflow.system_metrics.metrics.network_monitor import NetworkMonitor

_logger = logging.getLogger(__name__)


class SystemMetricsMonitor:
    """Class for monitoring system stats.

    This class is used for pulling system metrics and logging them to MLflow. Calling `start()` will
    spawn a thread that logs system metrics periodically. Calling `finish()` will stop the thread.
    Logging is done on a different frequency from pulling metrics, so that the metrics are
    aggregated over the period. Users can change the logging frequency by setting
    `MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL` and `MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING`
    environment variables, e.g., run `export MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL=10` in terminal
    will set the sampling interval to 10 seconds.

    System metrics are logged with a prefix "system/", e.g., "system/cpu_utilization_percentage".

    Args:
        run_id: string, the MLflow run ID.
        sampling_interval: float, default to 10. The interval (in seconds) at which to pull system
            metrics. Will be overridden by `MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL` environment
            variable.
        samples_before_logging: int, default to 1. The number of samples to aggregate before
            logging. Will be overridden by `MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING`
            evnironment variable.
        resume_logging: bool, default to False. If True, we will resume the system metrics logging
            from the `run_id`, and the first step to log will be the last step of `run_id` + 1, if
            False, system metrics logging will start from step 0.
        node_id: string, default to None. The node ID of the machine where the metrics are
            collected. Will be overridden by `MLFLOW_SYSTEM_METRICS_NODE_ID`
            evnironment variable. This is useful in multi-node training to distinguish the metrics
            from different nodes. For example, if you set node_id to "node_0", the system metrics
            getting logged will be of format "system/node_0/cpu_utilization_percentage".
    """

    def __init__(
        self,
        run_id,
        sampling_interval=10,
        samples_before_logging=1,
        resume_logging=False,
        node_id=None,
    ):
        from mlflow.utils.autologging_utils import BatchMetricsLogger

        # Instantiate default monitors.
        self.monitors = [CPUMonitor(), DiskMonitor(), NetworkMonitor()]
        try:
            gpu_monitor = GPUMonitor()
            self.monitors.append(gpu_monitor)
        except Exception as e:
            _logger.warning(
                f"Skip logging GPU metrics because creating `GPUMonitor` failed with error: {e}."
            )

        self.sampling_interval = MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL.get() or sampling_interval
        self.samples_before_logging = (
            MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING.get() or samples_before_logging
        )

        self._run_id = run_id
        self.mlflow_logger = BatchMetricsLogger(self._run_id)
        self._shutdown_event = threading.Event()
        self._process = None
        self._metrics_prefix = "system/"
        self.node_id = MLFLOW_SYSTEM_METRICS_NODE_ID.get() or node_id
        self._logging_step = self._get_next_logging_step(run_id) if resume_logging else 0

    def _get_next_logging_step(self, run_id):
        from mlflow.tracking.client import MlflowClient

        client = MlflowClient()
        try:
            run = client.get_run(run_id)
        except MlflowException:
            return 0
        system_metric_name = None
        for metric_name in run.data.metrics.keys():
            if metric_name.startswith(self._metrics_prefix):
                system_metric_name = metric_name
                break
        if system_metric_name is None:
            return 0
        metric_history = client.get_metric_history(run_id, system_metric_name)
        return metric_history[-1].step + 1

    def start(self):
        """Start monitoring system metrics."""
        try:
            self._process = threading.Thread(
                target=self.monitor,
                daemon=True,
                name="SystemMetricsMonitor",
            )
            self._process.start()
            _logger.info("Started monitoring system metrics.")
        except Exception as e:
            _logger.warning(f"Failed to start monitoring system metrics: {e}")
            self._process = None

    def monitor(self):
        """Main monitoring loop, which consistently collect and log system metrics."""
        from mlflow.tracking.fluent import get_run

        while not self._shutdown_event.is_set():
            for _ in range(self.samples_before_logging):
                self.collect_metrics()
                self._shutdown_event.wait(self.sampling_interval)
                try:
                    # Get the MLflow run to check if the run is not RUNNING.
                    run = get_run(self._run_id)
                except Exception as e:
                    _logger.warning(f"Failed to get mlflow run: {e}.")
                    return
                if run.info.status != "RUNNING" or self._shutdown_event.is_set():
                    # If the mlflow run is terminated or receives the shutdown signal, stop
                    # monitoring.
                    return
            metrics = self.aggregate_metrics()
            try:
                self.publish_metrics(metrics)
            except Exception as e:
                _logger.warning(
                    f"Failed to log system metrics: {e}, this is expected if the experiment/run is "
                    "already terminated."
                )
                return

    def collect_metrics(self):
        """Collect system metrics."""
        metrics = {}
        for monitor in self.monitors:
            monitor.collect_metrics()
            metrics.update(monitor._metrics)
        return metrics

    def aggregate_metrics(self):
        """Aggregate collected metrics."""
        metrics = {}
        for monitor in self.monitors:
            metrics.update(monitor.aggregate_metrics())
        return metrics

    def publish_metrics(self, metrics):
        """Log collected metrics to MLflow."""
        # Add prefix "system/" to the metrics name for grouping. If `self.node_id` is not None, also
        # add it to the metrics name.
        prefix = self._metrics_prefix + (self.node_id + "/" if self.node_id else "")
        metrics = {prefix + k: v for k, v in metrics.items()}

        self.mlflow_logger.record_metrics(metrics, self._logging_step)
        self._logging_step += 1
        for monitor in self.monitors:
            monitor.clear_metrics()

    def finish(self):
        """Stop monitoring system metrics."""
        if self._process is None:
            return
        _logger.info("Stopping system metrics monitoring...")
        self._shutdown_event.set()
        try:
            self._process.join()
            self.mlflow_logger.flush()
            _logger.info("Successfully terminated system metrics monitoring!")
        except Exception as e:
            _logger.error(f"Error terminating system metrics monitoring process: {e}.")
        self._process = None
