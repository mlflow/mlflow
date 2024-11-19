import threading
import time
from typing import Any, Callable

import pytest

import mlflow
from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor


@pytest.fixture(autouse=True)
def disable_system_metrics_logging():
    yield
    # Unset the environment variables to avoid affecting other test cases.
    mlflow.disable_system_metrics_logging()
    mlflow.set_system_metrics_sampling_interval(None)
    mlflow.set_system_metrics_samples_before_logging(None)
    mlflow.set_system_metrics_node_id(None)


def wait_for_condition(
    condition_func: Callable[[], Any], timeout: int = 10, check_interval: int = 1
) -> None:
    start_time = time.time()
    while time.time() - start_time < timeout:
        if condition_func():
            return
        time.sleep(check_interval)
    pytest.fail(f"Condition not met within {timeout} seconds.")


def test_manual_system_metrics_monitor():
    metric_test = "system/cpu_utilization_percentage"
    with mlflow.start_run(log_system_metrics=False) as run:
        system_monitor = SystemMetricsMonitor(
            run.info.run_id,
            sampling_interval=0.1,
            samples_before_logging=2,
        )
        system_monitor.start()
        thread_names = [thread.name for thread in threading.enumerate()]
        # Check the system metrics monitoring thread has been started.
        assert "SystemMetricsMonitor" in thread_names

        wait_for_condition(
            lambda: len(mlflow.MlflowClient().get_metric_history(run.info.run_id, metric_test)) > 1
        )
    wait_for_condition(
        lambda: "SystemMetricsMonitor" not in [thread.name for thread in threading.enumerate()]
    )

    mlflow_run = mlflow.get_run(run.info.run_id)
    metrics = mlflow_run.data.metrics

    expected_metrics_name = [
        "cpu_utilization_percentage",
        "system_memory_usage_megabytes",
        "disk_usage_percentage",
        "disk_usage_megabytes",
        "disk_available_megabytes",
        "network_receive_megabytes",
        "network_transmit_megabytes",
    ]
    expected_metrics_name = [f"system/{name}" for name in expected_metrics_name]
    for name in expected_metrics_name:
        assert name in metrics

    # Check the step is correctly logged.
    metrics_history = mlflow.MlflowClient().get_metric_history(run.info.run_id, metric_test)
    assert metrics_history[-1].step > 0


def test_automatic_system_metrics_monitor():
    metric_test = "system/cpu_utilization_percentage"
    mlflow.enable_system_metrics_logging()
    mlflow.set_system_metrics_sampling_interval(0.2)
    mlflow.set_system_metrics_samples_before_logging(2)
    with mlflow.start_run() as run:
        thread_names = [thread.name for thread in threading.enumerate()]
        # Check the system metrics monitoring thread has been started.
        assert "SystemMetricsMonitor" in thread_names

        wait_for_condition(
            lambda: len(mlflow.MlflowClient().get_metric_history(run.info.run_id, metric_test)) > 1
        )

    wait_for_condition(
        lambda: "SystemMetricsMonitor" not in [thread.name for thread in threading.enumerate()]
    )

    mlflow_run = mlflow.get_run(run.info.run_id)
    metrics = mlflow_run.data.metrics

    expected_metrics_name = [
        "cpu_utilization_percentage",
        "system_memory_usage_megabytes",
        "disk_usage_percentage",
        "disk_usage_megabytes",
        "disk_available_megabytes",
        "network_receive_megabytes",
        "network_transmit_megabytes",
    ]
    expected_metrics_name = [f"system/{name}" for name in expected_metrics_name]
    for name in expected_metrics_name:
        assert name in metrics

    # Check the step is correctly logged.
    metrics_history = mlflow.MlflowClient().get_metric_history(run.info.run_id, metric_test)
    assert metrics_history[-1].step > 0


def test_automatic_system_metrics_monitor_resume_existing_run():
    mlflow.enable_system_metrics_logging()
    mlflow.set_system_metrics_sampling_interval(0.2)
    mlflow.set_system_metrics_samples_before_logging(2)
    with mlflow.start_run() as run:
        time.sleep(2)

    wait_for_condition(
        lambda: "SystemMetricsMonitor" not in [thread.name for thread in threading.enumerate()]
    )

    # Get the last step.
    metrics_history = mlflow.MlflowClient().get_metric_history(
        run.info.run_id, "system/cpu_utilization_percentage"
    )
    last_step = metrics_history[-1].step

    with mlflow.start_run(run.info.run_id) as run:
        time.sleep(2)
    mlflow_run = mlflow.get_run(run.info.run_id)
    metrics = mlflow_run.data.metrics

    expected_metrics_name = [
        "cpu_utilization_percentage",
        "system_memory_usage_megabytes",
        "disk_usage_percentage",
        "disk_usage_megabytes",
        "disk_available_megabytes",
        "network_receive_megabytes",
        "network_transmit_megabytes",
    ]
    expected_metrics_name = [f"system/{name}" for name in expected_metrics_name]
    for name in expected_metrics_name:
        assert name in metrics

    # Check the step is correctly resumed.
    metrics_history = mlflow.MlflowClient().get_metric_history(
        run.info.run_id, "system/cpu_utilization_percentage"
    )
    assert metrics_history[-1].step > last_step


def test_system_metrics_monitor_with_multi_node():
    mlflow.enable_system_metrics_logging()
    mlflow.set_system_metrics_sampling_interval(0.2)
    mlflow.set_system_metrics_samples_before_logging(2)

    with mlflow.start_run() as run:
        run_id = run.info.run_id

    node_ids = ["0", "1", "2", "3"]
    for node_id in node_ids:
        mlflow.set_system_metrics_node_id(node_id)
        with mlflow.start_run(run_id=run_id, log_system_metrics=True):
            wait_for_condition(
                lambda: any(
                    k.startswith(f"system/{node_id}/")
                    for k in mlflow.get_run(run_id).data.metrics.keys()
                )
            )

    mlflow_run = mlflow.get_run(run_id)
    metrics = mlflow_run.data.metrics

    for node_id in node_ids:
        expected_metric_name = f"system/{node_id}/cpu_utilization_percentage"
        assert expected_metric_name in metrics.keys()
