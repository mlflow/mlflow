"""Class for monitoring system stats."""

import threading
import logging

from mlflow.system_metrics.metrics.cpu_monitor import CPUMonitor
from mlflow.system_metrics.metrics.disk_monitor import DiskMonitor
from mlflow.system_metrics.metrics.network_monitor import NetworkMonitor
from mlflow.system_metrics.metrics.gpu_monitor import GPUMonitor

logger = logging.getLogger(__name__)


class SystemMetricsMonitor:
    """Class for monitoring system stats.

    This class is used for pulling system metrics and logging them to MLflow. Calling `start()` will
    spawn a thread that logs system metrics periodically. Calling `finish()` will stop the thread.

    Args:
        mlflow_run: an 'mlflow.entities.run.Run' instance, which is used to bootstrap system metrics
            logging with the MLflow tracking server.
        logging_interval: float, the interval (in seconds) at which to log system metrics to MLflow.
    """

    def __init__(self, mlflow_run, logging_interval=5.0):
        from mlflow.utils.autologging_utils import BatchMetricsLogger

        # Instantiate default monitors.
        self.monitors = [CPUMonitor(), DiskMonitor(), NetworkMonitor()]
        gpu_monitor = GPUMonitor()
        if gpu_monitor:
            self.monitors.append(gpu_monitor)
        self.logging_interval = logging_interval

        self.mlflow_logger = BatchMetricsLogger(mlflow_run.info.run_id)
        self._shutdown_event = threading.Event()
        self._process = None
        # Attach `SystemMetricsMonitor` instance to the `mlflow_run` instance.
        mlflow_run.system_metrics_monitor = self

    def start(self):
        """Start monitoring system metrics."""
        try:
            self._process = threading.Thread(
                target=self.monitor,
                daemon=True,
                name="SystemMetricsMonitor",
            )
            self._process.start()
            logger.info(f"MLflow: started monitoring system metrics.")
        except Exception as e:
            logger.warning(f"MLflow: failed to start monitoring system metrics: {e}")
            self._process = None

    def monitor(self):
        """Main loop for the thread, which consistently collect and log system metrics."""
        while not self._shutdown_event.is_set():
            metrics = self.collect_metrics()
            self._shutdown_event.wait(self.logging_interval)
            if self._shutdown_event.is_set():
                break
            try:
                self.log_metrics(metrics)
            except Exception as e:
                logger.warning(
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

    def log_metrics(self, metrics):
        """Log collected metrics to MLflow."""
        self.mlflow_logger.record_metrics(metrics)

    def finish(self):
        """Stop monitoring system metrics."""
        if self._process is None:
            return None
        logger.info("MLflow: stopping system metrics monitoring...")
        self._shutdown_event.set()
        try:
            self._process.join()
        except Exception as e:
            logger.error(f"Error joining system metrics monitoring process: {e}")
        self._process = None
