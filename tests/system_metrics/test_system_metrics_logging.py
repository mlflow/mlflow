import logging
import threading
import time
from typing import Any, Callable
from unittest import mock

import pytest

import mlflow
from mlflow.entities import Metric, Run, RunData, RunInfo
from mlflow.exceptions import MlflowException
from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor


@pytest.fixture(autouse=True)
def enable_debug_logging():
    # Enable debug logging to help diagnose flaky test failures
    logger = logging.getLogger("mlflow.system_metrics.system_metrics_monitor")
    original_level = logger.level
    logger.setLevel(logging.DEBUG)
    yield
    logger.setLevel(original_level)


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
            lambda: len(mlflow.MlflowClient().get_metric_history(run.info.run_id, metric_test)) > 1,
            timeout=20,
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
            lambda: len(mlflow.MlflowClient().get_metric_history(run.info.run_id, metric_test)) > 1,
            timeout=20,
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


def _make_fake_run(run_id, metric_objs):
    run_info = RunInfo(
        run_id=run_id,
        experiment_id="0",
        user_id="user",
        status="RUNNING",
        start_time=0,
        end_time=None,
        lifecycle_stage="active",
    )
    return Run(run_info=run_info, run_data=RunData(metrics=metric_objs, params=[], tags=[]))


def test_get_next_logging_step_uses_run_data_without_calling_get_metric_history():
    # Regression test for GH issue #22991: avoid materializing the full metric history
    # in `_get_next_logging_step`. We assert that the resume path only fetches the run
    # (whose data already contains the latest step per metric) and never calls
    # `get_metric_history`, which would page through the entire history.
    run_id = "abc123"
    metric_objs = [
        Metric(key="loss", value=0.5, timestamp=1, step=10),
        Metric(key="system/cpu_utilization_percentage", value=42.0, timestamp=2, step=7),
        Metric(key="system/disk_usage_percentage", value=11.0, timestamp=3, step=9),
    ]
    fake_run = _make_fake_run(run_id, metric_objs)

    with (
        mock.patch(
            "mlflow.tracking.client.MlflowClient.get_run", return_value=fake_run
        ) as mock_get_run,
        mock.patch(
            "mlflow.tracking.client.MlflowClient.get_metric_history"
        ) as mock_get_metric_history,
    ):
        monitor = SystemMetricsMonitor(run_id=run_id, resume_logging=True)

    mock_get_run.assert_called_once_with(run_id)
    mock_get_metric_history.assert_not_called()
    # Resume from the largest step across all `system/*` metrics.
    assert monitor._logging_step == 10


def test_get_next_logging_step_uses_max_system_metric_step():
    run_id = "unordered-system-metrics"
    metric_objs = [
        Metric(key="system/cpu_utilization_percentage", value=42.0, timestamp=2, step=7),
        Metric(key="loss", value=0.5, timestamp=1, step=10),
        Metric(key="system/disk_usage_percentage", value=11.0, timestamp=3, step=9),
        Metric(key="system/network_receive_megabytes", value=2.0, timestamp=4, step=8),
    ]
    fake_run = _make_fake_run(run_id, metric_objs)

    with (
        mock.patch("mlflow.tracking.client.MlflowClient.get_run", return_value=fake_run),
        mock.patch(
            "mlflow.tracking.client.MlflowClient.get_metric_history"
        ) as mock_get_metric_history,
    ):
        monitor = SystemMetricsMonitor(run_id=run_id, resume_logging=True)

    mock_get_metric_history.assert_not_called()
    assert monitor._logging_step == 10


def test_get_next_logging_step_returns_zero_when_no_system_metrics():
    run_id = "no-sys-metrics"
    metric_objs = [Metric(key="loss", value=0.5, timestamp=1, step=10)]
    fake_run = _make_fake_run(run_id, metric_objs)

    with (
        mock.patch("mlflow.tracking.client.MlflowClient.get_run", return_value=fake_run),
        mock.patch(
            "mlflow.tracking.client.MlflowClient.get_metric_history"
        ) as mock_get_metric_history,
    ):
        monitor = SystemMetricsMonitor(run_id=run_id, resume_logging=True)

    mock_get_metric_history.assert_not_called()
    assert monitor._logging_step == 0


def test_get_next_logging_step_returns_zero_when_run_not_found():
    run_id = "missing-run"
    with (
        mock.patch(
            "mlflow.tracking.client.MlflowClient.get_run",
            side_effect=MlflowException("not found"),
        ),
        mock.patch(
            "mlflow.tracking.client.MlflowClient.get_metric_history"
        ) as mock_get_metric_history,
    ):
        monitor = SystemMetricsMonitor(run_id=run_id, resume_logging=True)

    mock_get_metric_history.assert_not_called()
    assert monitor._logging_step == 0


def test_get_next_logging_step_performance_with_large_metric_set():
    # Performance regression coverage for GH issue #22991:
    # Use many latest metrics and ensure resume stays bounded by one `get_run` call
    # with no metric-history paging / full-history materialization.
    run_id = "large-metric-set"
    large_metric_set = [
        Metric(key=f"system/custom_metric_{i}", value=float(i), timestamp=i, step=i)
        for i in range(25_000)
    ]
    large_metric_set.append(Metric(key="loss", value=0.5, timestamp=1, step=99999))
    fake_run = _make_fake_run(run_id, large_metric_set)

    with (
        mock.patch(
            "mlflow.tracking.client.MlflowClient.get_run", return_value=fake_run
        ) as mock_get_run,
        mock.patch(
            "mlflow.tracking.client.MlflowClient.get_metric_history",
            side_effect=AssertionError("get_metric_history must not be called"),
        ),
    ):
        monitor = SystemMetricsMonitor(run_id=run_id, resume_logging=True)

    mock_get_run.assert_called_once_with(run_id)
    assert monitor._logging_step == 25_000


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
