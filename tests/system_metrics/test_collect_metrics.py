from mlflow.system_metrics.metrics.cpu_monitor import CPUMonitor
from mlflow.system_metrics.metrics.disk_monitor import DiskMonitor
from mlflow.system_metrics.metrics.gpu_monitor import GPUMonitor
from mlflow.system_metrics.metrics.network_monitor import NetworkMonitor


def test_cpu_monitor():
    cpu_monitor = CPUMonitor()
    cpu_monitor.collect_metrics()

    assert isinstance(cpu_monitor.metrics["cpu_utilization_percentage"], list)
    assert isinstance(cpu_monitor.metrics["system_memory_usage_megabytes"], list)

    cpu_monitor.collect_metrics()
    aggregated_metrics = cpu_monitor.aggregate_metrics()
    assert isinstance(aggregated_metrics["cpu_utilization_percentage"], float)
    assert isinstance(aggregated_metrics["system_memory_usage_megabytes"], float)

    cpu_monitor.clear_metrics()
    assert cpu_monitor.metrics == {}


def test_gpu_monitor():
    try:
        gpu_monitor = GPUMonitor()
    except Exception:
        # If pynvml is not installed, or there is no GPU, then `gpu_monitor` creation will fail. In
        # this case we skip the test.
        return

    gpu_monitor.collect_metrics()

    assert isinstance(gpu_monitor.metrics["gpu_0_memory_usage_percentage"], list)
    assert isinstance(gpu_monitor.metrics["gpu_0_memory_usage_megabytes"], list)
    assert isinstance(gpu_monitor.metrics["gpu_0_utilization_percentage"], list)
    assert isinstance(gpu_monitor.metrics["gpu_0_power_usage_watts"], list)
    assert isinstance(gpu_monitor.metrics["gpu_0_power_usage_percentage"], list)

    gpu_monitor.collect_metrics()
    aggregated_metrics = gpu_monitor.aggregate_metrics()
    assert isinstance(aggregated_metrics["gpu_0_memory_usage_percentage"], float)
    assert isinstance(aggregated_metrics["gpu_0_memory_usage_megabytes"], float)
    assert isinstance(aggregated_metrics["gpu_0_utilization_percentage"], float)
    assert isinstance(aggregated_metrics["gpu_0_power_usage_watts"], float)
    assert isinstance(aggregated_metrics["gpu_0_power_usage_percentage"], float)

    gpu_monitor.clear_metrics()
    assert len(gpu_monitor.metrics.keys) == 0


def test_disk_monitor():
    disk_monitor = DiskMonitor()
    disk_monitor.collect_metrics()

    assert len(disk_monitor.metrics.keys()) > 0
    assert isinstance(disk_monitor.metrics["disk_usage_percentage"], list)
    assert isinstance(disk_monitor.metrics["disk_usage_megabytes"], list)
    assert isinstance(disk_monitor.metrics["disk_available_megabytes"], list)

    disk_monitor.collect_metrics()
    aggregated_metrics = disk_monitor.aggregate_metrics()
    assert len(aggregated_metrics.keys()) > 0

    assert isinstance(aggregated_metrics["disk_usage_percentage"], float)
    assert isinstance(aggregated_metrics["disk_usage_megabytes"], float)
    assert isinstance(aggregated_metrics["disk_available_megabytes"], float)

    disk_monitor.clear_metrics()
    assert len(disk_monitor.metrics.keys()) == 0


def test_network_monitor():
    network_monitor = NetworkMonitor()
    network_monitor.collect_metrics()

    assert len(network_monitor.metrics.keys()) > 0
    assert isinstance(network_monitor.metrics["network_receive_megabytes"], float)
    assert isinstance(network_monitor.metrics["network_transmit_megabytes"], float)

    network_monitor.collect_metrics()
    aggregated_metrics = network_monitor.aggregate_metrics()
    assert len(aggregated_metrics.keys()) > 0

    assert isinstance(aggregated_metrics["network_receive_megabytes"], float)
    assert isinstance(aggregated_metrics["network_transmit_megabytes"], float)

    network_monitor.clear_metrics()
    assert len(network_monitor.metrics.keys()) == 0
