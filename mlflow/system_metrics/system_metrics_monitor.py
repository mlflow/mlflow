"""Class for monitoring system stats."""

import logging
import threading

from mlflow.system_metrics.metrics.cpu_monitor import CPUMonitor
from mlflow.system_metrics.metrics.gpu_monitor import GPUMonitor

logger = logging.getLogger(__name__)


class SystemMetricsMonitor:
    """Class for monitoring system stats.

    This class is used for pulling system metrics and logging them to MLflow. Calling `start()` will
    spawn a thread that logs system metrics periodically. Calling `finish()` will stop the thread.

    Args:
        mlflow_run: an 'mlflow.entities.run.Run' instance, which is used to bootstrap system metrics
            logging with the MLflow tracking server.
        logging_interval: float, default to 15.0. The interval (in seconds) at which to log system
            metrics to MLflow.
    """

    def __init__(self, mlflow_run, logging_interval=15.0):
        from mlflow.utils.autologging_utils import BatchMetricsLogger

        # Instantiate default monitors.
        self.monitors = [CPUMonitor()]
        gpu_monitor = GPUMonitor()
        if gpu_monitor:
            self.monitors.append(gpu_monitor)
        self.logging_interval = logging_interval

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
            logger.info("MLflow: started monitoring system metrics.")
        except Exception as e:
            logger.warning(f"MLflow: failed to start monitoring system metrics: {e}")
            self._process = None

    def monitor(self):
        """Main monitoring loop, which consistently collect and log system metrics."""
        from mlflow.tracking.fluent import get_run

        while not self._shutdown_event.is_set():
            metrics = self.collect_metrics()
            self._shutdown_event.wait(self.logging_interval)
            run = get_run(self._run_id)
            if run.info.status == "FINISHED" or self._shutdown_event.is_set():
                # If the mlflow run is terminated or receives the shutdown signal, stop monitoring.
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
