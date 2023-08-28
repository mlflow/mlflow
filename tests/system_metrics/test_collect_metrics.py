from mlflow.system_metrics.metrics.cpu_monitor import CPUMonitor
from mlflow.system_metrics.metrics.disk_monitor import DiskMonitor
from mlflow.system_metrics.metrics.gpu_monitor import GPUMonitor
from mlflow.system_metrics.metrics.network_monitor import NetworkMonitor


def test_cpu_monitor():
    cpu_monitor = CPUMonitor()
    cpu_monitor.collect_metrics()

    assert isinstance(cpu_monitor.metrics["cpu_percentage"], list)
    assert isinstance(cpu_monitor.metrics["cpu_memory_used"], list)

    cpu_monitor.collect_metrics()
    aggregated_metrics = cpu_monitor.aggregate_metrics()
    assert isinstance(aggregated_metrics["cpu_percentage"], float)
    assert isinstance(aggregated_metrics["cpu_memory_used"], float)

    cpu_monitor.clear_metrics()
    assert len(cpu_monitor.metrics.keys()) == 0


def test_gpu_monitor():
    gpu_monitor = GPUMonitor()
    if gpu_monitor is None:
        # If pynvml is not installed, or there is no GPU, then `gpu_monitor` will be None. In this
        # case we skip the test.
        return
    gpu_monitor.collect_metrics()

    assert isinstance(gpu_monitor.metrics["gpu_0_memory_used"], list)
    assert isinstance(gpu_monitor.metrics["gpu_0_utilization_rate"], list)

    gpu_monitor.collect_metrics()
    aggregated_metrics = gpu_monitor.aggregate_metrics()
    assert isinstance(aggregated_metrics["gpu_0_memory_used"], float)
    assert isinstance(aggregated_metrics["gpu_0_utilization_rate"], float)

    gpu_monitor.clear_metrics()
    assert len(gpu_monitor.metrics.keys) == 0


def test_disk_monitor():
    disk_monitor = DiskMonitor()
    disk_monitor.collect_metrics()

    assert len(disk_monitor.metrics.keys()) > 0
    for _, value in disk_monitor.metrics.items():
        assert isinstance(value, list)

    disk_monitor.collect_metrics()
    aggregated_metrics = disk_monitor.aggregate_metrics()
    assert len(aggregated_metrics.keys()) > 0
    for key, value in aggregated_metrics.items():
        assert isinstance(value, float)

    disk_monitor.clear_metrics()
    assert len(disk_monitor.metrics.keys()) == 0


def test_network_monitor():
    network_monitor = NetworkMonitor()
    network_monitor.collect_metrics()

    assert len(network_monitor.metrics.keys()) > 0
    for _, value in network_monitor.metrics.items():
        assert isinstance(value, list)

    network_monitor.collect_metrics()
    aggregated_metrics = network_monitor.aggregate_metrics()
    assert len(aggregated_metrics.keys()) > 0
    for key, value in aggregated_metrics.items():
        assert isinstance(value, float)

    network_monitor.clear_metrics()
    assert len(network_monitor.metrics.keys()) == 0
