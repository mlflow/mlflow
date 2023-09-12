import threading
import time

import mlflow
from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor


def test_manual_system_metrics_monitor():
    with mlflow.start_run(log_system_metrics=False) as run:
        system_monitor = SystemMetricsMonitor(
            run.info.run_id,
            sampling_interval=0.1,
            samples_before_logging=5,
        )
        system_monitor.start()
        thread_names = [thread.name for thread in threading.enumerate()]
        # Check the system metrics monitoring thread has been started.
        assert "SystemMetricsMonitor" in thread_names

        time.sleep(2)
    mlflow.end_run()
    # Pause for a bit to allow the system metrics monitoring to exit.
    time.sleep(1)
    thread_names = [thread.name for thread in threading.enumerate()]
    # Check the system metrics monitoring thread has exited.
    assert "SystemMetricsMonitor" not in thread_names

    mlflow_run = mlflow.get_run(run.info.run_id)
    metrics = mlflow_run.data.metrics
    assert "cpu_percentage" in metrics
    assert "system_memory_used" in metrics


def test_automatic_system_metrics_monitor():
    mlflow.enable_system_metrics_logging()
    mlflow.set_system_metrics_sampling_interval(0.2)
    mlflow.set_system_metrics_samples_before_logging(5)
    run = mlflow.start_run()
    thread_names = [thread.name for thread in threading.enumerate()]
    # Check the system metrics monitoring thread has been started.
    assert "SystemMetricsMonitor" in thread_names

    # Pause for a bit to allow the system metrics monitoring to exit.
    time.sleep(2)
    mlflow.end_run()

    # Pause for a bit to allow the system metrics monitoring to exit.
    time.sleep(1)
    thread_names = [thread.name for thread in threading.enumerate()]
    # Check the system metrics monitoring thread has exited.
    assert "SystemMetricsMonitor" not in thread_names

    mlflow_run = mlflow.get_run(run.info.run_id)
    metrics = mlflow_run.data.metrics
    assert "cpu_percentage" in metrics
    assert "system_memory_used" in metrics
