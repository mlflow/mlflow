"""Class for monitoring system stats."""

import logging
import os
import threading

from mlflow.system_metrics.metrics.cpu_monitor import CPUMonitor
from mlflow.system_metrics.metrics.gpu_monitor import GPUMonitor

_logger = logging.getLogger(__name__)


class SystemMetricsMonitor:
    """Class for monitoring system stats.

    This class is used for pulling system metrics and logging them to MLflow. Calling `start()` will
    spawn a thread that logs system metrics periodically. Calling `finish()` will stop the thread.
    Logging is done on a different frequency from pulling metrics, so that the metrics are
    aggregated over the period. Users can change the logging frequency by setting
    `$MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL` and `$MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING`
    environment variables.

    Args:
        mlflow_run: an 'mlflow.entities.run.Run' instance, which is used to bootstrap system metrics
            logging with the MLflow tracking server.
        sampling_interval: float, default to 0.5. The interval (in seconds) at which to pull system
            metrics.
        samples_to_aggregate: int, default to 30. The number of samples to aggregate before logging.
    """

    def __init__(self, mlflow_run, sampling_interval=10, samples_before_logging=1):
        from mlflow.utils.autologging_utils import BatchMetricsLogger

        # Instantiate default monitors.
        self.monitors = [CPUMonitor()]
        gpu_monitor = GPUMonitor()
        if gpu_monitor:
            self.monitors.append(gpu_monitor)
        self.sampling_interval = os.environ.get(
            "MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL", sampling_interval
        )
        self.samples_before_logging = os.environ.get(
            "MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING", samples_before_logging
        )
        if isinstance(self.sampling_interval, str):
            self.sampling_interval = float(self.sampling_interval)
        if isinstance(self.samples_before_logging, str):
            self.samples_before_logging = int(self.samples_before_logging)

        self._run_id = mlflow_run.info.run_id
        self.mlflow_logger = BatchMetricsLogger(self._run_id)
        self._shutdown_event = threading.Event()
        self._process = None

    def start(self):
        """Start monitoring system metrics."""
        try:
            self._process = threading.Thread(
                target=self.monitor,
                daemon=True,
                name="SystemMetricsMonitor",
            )
            self._process.start()
            _logger.info("MLflow: started monitoring system metrics.")
        except Exception as e:
            _logger.warning(f"MLflow: failed to start monitoring system metrics: {e}")
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
                    _logger.warning(f"MLflow: failed to get mlflow run: {e}.")
                    return
                if run.info.status != "RUNNING" or self._shutdown_event.is_set():
                    # If the mlflow run is terminated or receives the shutdown signal, stop monitoring.
                    return
            metrics = self.aggregate_metrics()
            try:
                self.publish_metrics(metrics)
            except Exception as e:
                _logger.warning(
                    f"MLflow: failed to log system metrics: {e}, this is expected if the "
                    "experiment/run is already terminated."
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
        self.mlflow_logger.record_metrics(metrics)
        for monitor in self.monitors:
            monitor.clear_metrics()

    def finish(self):
        """Stop monitoring system metrics."""
        if self._process is None:
            return None
        _logger.info("MLflow: stopping system metrics monitoring...")
        self._shutdown_event.set()
        try:
            self._process.join()
        except Exception as e:
            _logger.error(f"Error joining system metrics monitoring process: {e}")
        self._process = None
