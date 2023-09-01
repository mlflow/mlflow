"""Class for monitoring GPU stats."""

import logging
import sys

from mlflow.system_metrics.metrics.base_metrics_monitor import BaseMetricsMonitor

logger = logging.getLogger(__name__)

try:
    import pynvml
except ImportError:
    logger.warning(
        "`GPUMonitor` requires pynvml package. To install, run `pip install pynvml`. Skip "
        "logging GPU metrics."
    )


class GPUMonitor(BaseMetricsMonitor):
    """Class for monitoring GPU stats."""

    def __new__(cls, *args, **kwargs):
        # Override `__new__` to return a None object if `pynvml` is not installed or GPU is not
        # found.
        if "pynvml" not in sys.modules:
            # Only instantiate if `pynvml` is installed.
            return
        try:
            # `nvmlInit()` will fail if no GPU is found.
            pynvml.nvmlInit()
        except pynvml.NVMLError as e:
            logger.warning(f"Failed to initialize NVML, skip logging GPU metrics: {e}")
            return
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, name="gpu"):
        super().__init__(name)
        self.num_gpus = pynvml.nvmlDeviceGetCount()
        self.gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.num_gpus)]

    def collect_metrics(self):
        # Get GPU metrics.
        for i, handle in enumerate(self.gpu_handles):
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self._metrics[f"gpu_{i}_memory_used"].append(int(memory.used / 1e6))

            device_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            self._metrics[f"gpu_{i}_utilization_rate"].append(device_utilization.gpu)

    def aggregate_metrics(self):
        for name, values in self._metrics.items():
            self._metrics[name] = sum(values) / len(values)
