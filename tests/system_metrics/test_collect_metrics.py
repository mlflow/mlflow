import contextlib
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from mlflow.system_metrics.metrics import gpu_monitor as gpu_monitor_module
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
        # If nvidia-ml-py is not installed, or there is no GPU, then `gpu_monitor` creation
        # will fail. In this case we skip the test.
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


class _FakeNVMLError(Exception):
    pass


@contextlib.contextmanager
def _mock_pynvml(lost_gpus: set[int]):
    def get_handle(index):
        if index in lost_gpus:
            raise _FakeNVMLError("GPU is lost")
        return index

    fake_pynvml = mock.MagicMock(
        NVMLError=_FakeNVMLError,
        nvmlDeviceGetCount=mock.Mock(return_value=3),
        nvmlDeviceGetHandleByIndex=mock.Mock(side_effect=get_handle),
        nvmlDeviceGetMemoryInfo=mock.Mock(return_value=SimpleNamespace(used=1e9, total=4e9)),
        nvmlDeviceGetUtilizationRates=mock.Mock(return_value=SimpleNamespace(gpu=50)),
        nvmlDeviceGetPowerUsage=mock.Mock(return_value=100_000),
        nvmlDeviceGetEnforcedPowerLimit=mock.Mock(return_value=200_000),
    )
    with (
        mock.patch.dict(sys.modules, {"pynvml": fake_pynvml}),
        mock.patch.object(gpu_monitor_module, "pynvml", fake_pynvml, create=True),
    ):
        yield fake_pynvml


def test_gpu_monitor_skips_gpu_whose_handle_lookup_fails():
    with (
        _mock_pynvml(lost_gpus={1}) as fake_pynvml,
        mock.patch.object(gpu_monitor_module, "_logger") as mock_logger,
    ):
        gpu_monitor = GPUMonitor()
        gpu_monitor.collect_metrics()

    mock_logger.warning.assert_called_once_with(
        "Failed to get handle for GPU 1, skip logging its metrics: GPU is lost"
    )
    fake_pynvml.nvmlInit.assert_called_once_with()
    fake_pynvml.nvmlDeviceGetCount.assert_called_once_with()
    assert fake_pynvml.nvmlDeviceGetHandleByIndex.call_args_list == [
        mock.call(0),
        mock.call(1),
        mock.call(2),
    ]
    for query in (
        fake_pynvml.nvmlDeviceGetMemoryInfo,
        fake_pynvml.nvmlDeviceGetUtilizationRates,
        fake_pynvml.nvmlDeviceGetPowerUsage,
        fake_pynvml.nvmlDeviceGetEnforcedPowerLimit,
    ):
        assert query.call_args_list == [mock.call(0), mock.call(2)]
    for i in (0, 2):
        assert gpu_monitor.metrics[f"gpu_{i}_memory_usage_percentage"] == [25.0]
        assert gpu_monitor.metrics[f"gpu_{i}_memory_usage_megabytes"] == [1000.0]
        assert gpu_monitor.metrics[f"gpu_{i}_utilization_percentage"] == [50]
        assert gpu_monitor.metrics[f"gpu_{i}_power_usage_watts"] == [100.0]
        assert gpu_monitor.metrics[f"gpu_{i}_power_usage_percentage"] == [50.0]
    assert not any(name.startswith("gpu_1_") for name in gpu_monitor.metrics)


def test_gpu_monitor_raises_when_no_gpu_handle_is_available():
    with (
        _mock_pynvml(lost_gpus={0, 1, 2}) as fake_pynvml,
        pytest.raises(RuntimeError, match="Failed to get a handle for any GPU"),
    ):
        GPUMonitor()

    fake_pynvml.nvmlInit.assert_called_once_with()
    fake_pynvml.nvmlDeviceGetCount.assert_called_once_with()
    assert fake_pynvml.nvmlDeviceGetHandleByIndex.call_args_list == [
        mock.call(0),
        mock.call(1),
        mock.call(2),
    ]
    fake_pynvml.nvmlDeviceGetMemoryInfo.assert_not_called()


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
