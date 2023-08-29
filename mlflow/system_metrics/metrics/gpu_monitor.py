"""Class for monitoring GPU stats."""

import logging
import sys

from mlflow.system_metrics.metrics.base_metrics_monitor import BaseMetricsMonitor

logger = logging.getLogger(__name__)

try:
    import pynvml
except ImportError:
    print(
        "`GPUMonitor` requires pynvml package. To install, run `pip install pynvml`. Skip "
        "logging GPU metrics."
    )


class GPUMonitor(BaseMetricsMonitor):
    def __new__(cls, *args, **kwargs):
        if "pynvml" not in sys.modules:
            # Only instantiate if `pynvml` is installed.
            return
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError as e:
            logger.warning(f"Failed to initialize NVML, skip logging GPU metrics: {e}")
            return
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, name="cpu"):
        super().__init__(name)
        self.num_gpus = pynvml.nvmlDeviceGetCount()
        self.gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.num_gpus)]

    def collect_metrics(self):
        # Get metrics for the system.
        for i, handle in enumerate(self.gpu_handles):
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self._metrics[f"gpu_{i}_memory_total"] = int(memory.total / 1e6)
            self._metrics[f"gpu_{i}_memory_free"] = int(memory.free / 1e6)
            self._metrics[f"gpu_{i}_memory_used"] = int(memory.used / 1e6)

            device_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            self._metrics[f"gpu_{i}_utilization_rate"] = device_utilization.gpu
            self._metrics[f"gpu_{i}_memory_utilization_rate"] = device_utilization.memory

            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            self._metrics[f"gpu_{i}_temperature"] = temperature
