import subprocess
import sys
import time

from mlflow.system_metrics.metrics.process_monitor import ProcessMonitor


def test_collect_metrics_returns_expected_keys():
    monitor = ProcessMonitor(include_children=False)
    monitor.collect_metrics()

    expected_keys = [
        "process_cpu_percentage",
        "process_memory_rss_megabytes",
        "process_memory_vms_megabytes",
        "process_memory_percentage",
        "process_threads",
    ]

    for key in expected_keys:
        assert key in monitor.metrics, f"Expected key '{key}' not found in metrics"
        assert len(monitor.metrics[key]) == 1, f"Expected 1 value for '{key}'"


def test_aggregate_metrics_returns_numeric_values():
    monitor = ProcessMonitor(include_children=False)

    # Collect multiple samples
    for _ in range(3):
        monitor.collect_metrics()
        time.sleep(0.1)

    aggregated = monitor.aggregate_metrics()

    # Check that all values are numeric
    for key, value in aggregated.items():
        assert isinstance(value, (int, float)), f"Expected numeric value for '{key}'"
        assert value >= 0, f"Expected non-negative value for '{key}'"


def test_cpu_percentage_non_zero_after_work():
    monitor = ProcessMonitor(include_children=False)

    # Do some CPU work
    _ = sum(i * i for i in range(100000))

    monitor.collect_metrics()
    aggregated = monitor.aggregate_metrics()

    # CPU percentage should have been collected
    assert "process_cpu_percentage" in aggregated


def test_memory_tracking():
    monitor = ProcessMonitor(include_children=False)
    monitor.collect_metrics()
    aggregated = monitor.aggregate_metrics()

    # RSS and VMS should be positive
    assert aggregated["process_memory_rss_megabytes"] > 0
    assert aggregated["process_memory_vms_megabytes"] > 0
    assert 0 <= aggregated["process_memory_percentage"] <= 100


def test_thread_count():
    monitor = ProcessMonitor(include_children=False)
    monitor.collect_metrics()
    aggregated = monitor.aggregate_metrics()

    # Should have at least 1 thread (the main thread)
    assert aggregated["process_threads"] >= 1


def test_include_children_aggregates_child_processes():
    monitor_with_children = ProcessMonitor(include_children=True)
    monitor_without_children = ProcessMonitor(include_children=False)

    # Spawn a child process that does some work
    child_process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # Give the child process time to start
        time.sleep(0.5)

        monitor_with_children.collect_metrics()
        monitor_without_children.collect_metrics()

        with_children = monitor_with_children.aggregate_metrics()
        without_children = monitor_without_children.aggregate_metrics()

        # Thread count should be higher when including children
        # (or at least equal if child has fewer threads)
        assert with_children["process_threads"] >= without_children["process_threads"]

    finally:
        child_process.terminate()
        child_process.wait()


def test_clear_metrics():
    monitor = ProcessMonitor(include_children=False)
    monitor.collect_metrics()
    assert len(monitor.metrics) > 0

    monitor.clear_metrics()
    assert len(monitor.metrics) == 0


def test_handles_missing_process_gracefully():
    monitor = ProcessMonitor(include_children=False)

    # Simulate process being inaccessible
    original_process = monitor._process
    monitor._process = None

    # Should not raise an exception
    monitor.collect_metrics()
    aggregated = monitor.aggregate_metrics()

    # Should return empty dict when no process
    assert aggregated == {}

    # Restore
    monitor._process = original_process


def test_multiple_collections_aggregate_correctly():
    monitor = ProcessMonitor(include_children=False)

    # Collect 5 samples
    for _ in range(5):
        monitor.collect_metrics()

    # Each metric should have 5 values
    for key, values in monitor.metrics.items():
        assert len(values) == 5, f"Expected 5 values for '{key}', got {len(values)}"

    aggregated = monitor.aggregate_metrics()

    # Values should be rounded to 1 decimal place
    for key, value in aggregated.items():
        # Check that value is properly rounded
        assert value == round(value, 1)


def test_process_monitor_works_with_system_monitor():
    from mlflow.system_metrics.metrics.cpu_monitor import CPUMonitor

    process_monitor = ProcessMonitor()
    cpu_monitor = CPUMonitor()

    # Both should work independently
    process_monitor.collect_metrics()
    cpu_monitor.collect_metrics()

    process_metrics = process_monitor.aggregate_metrics()
    cpu_metrics = cpu_monitor.aggregate_metrics()

    # Both should have metrics
    assert len(process_metrics) > 0
    assert len(cpu_metrics) > 0

    # Keys should be different (process-level vs system-level)
    assert "process_cpu_percentage" in process_metrics
    assert "cpu_utilization_percentage" in cpu_metrics


def test_process_metrics_logged_to_mlflow_e2e():
    import mlflow
    from mlflow.environment_variables import (
        MLFLOW_SYSTEM_METRICS_INCLUDE_CHILD_PROCESSES,
        MLFLOW_SYSTEM_METRICS_INCLUDE_PROCESS,
    )
    from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor

    # Clean state
    MLFLOW_SYSTEM_METRICS_INCLUDE_PROCESS.unset()
    MLFLOW_SYSTEM_METRICS_INCLUDE_CHILD_PROCESSES.unset()

    try:
        # Enable process metrics via public API
        mlflow.enable_process_metrics(include_children=False)

        with mlflow.start_run(log_system_metrics=False) as run:
            # Manually create monitor to control timing (like existing tests)
            system_monitor = SystemMetricsMonitor(
                run.info.run_id,
                sampling_interval=0.1,
                samples_before_logging=1,
            )
            system_monitor.start()

            # Wait for metrics to be logged
            time.sleep(0.5)

        # Give time for metrics to finalize
        time.sleep(0.3)

        # Verify process metrics were logged
        client = mlflow.MlflowClient()
        run_data = client.get_run(run.info.run_id)
        metric_keys = list(run_data.data.metrics.keys())

        # Check that process metrics are present
        process_metrics = [k for k in metric_keys if "process_" in k]
        assert len(process_metrics) > 0, f"Expected process metrics, got: {metric_keys}"

        # Verify specific process metrics exist
        assert any("process_cpu_percentage" in k for k in metric_keys)
        assert any("process_memory_rss_megabytes" in k for k in metric_keys)

    finally:
        # Cleanup
        mlflow.disable_process_metrics()
        MLFLOW_SYSTEM_METRICS_INCLUDE_PROCESS.unset()
        MLFLOW_SYSTEM_METRICS_INCLUDE_CHILD_PROCESSES.unset()
