"""Class for monitoring GPU stats."""

import logging
import sys

from mlflow.system_metrics.metrics.base_metrics_monitor import BaseMetricsMonitor

_logger = logging.getLogger(__name__)

try:
    import pynvml
except ImportError:
    # If `pynvml` is not installed, a warning will be logged at monitor instantiation.
    # We don't log a warning here to avoid spamming warning at every import.
    pass


class GPUMonitor(BaseMetricsMonitor):
    """Class for monitoring GPU stats."""

    def __init__(self):
        if "pynvml" not in sys.modules:
            # Only instantiate if `pynvml` is installed.
            raise ImportError(
                "`pynvml` is not installed, to log GPU metrics please run `pip install pynvml` "
                "to install it."
            )
        try:
            # `nvmlInit()` will fail if no GPU is found.
            pynvml.nvmlInit()
        except pynvml.NVMLError as e:
            raise RuntimeError(f"Failed to initialize NVML, skip logging GPU metrics: {e}")

        super().__init__()
        self.num_gpus = pynvml.nvmlDeviceGetCount()
        self.gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.num_gpus)]

    def collect_metrics(self):
        # Get GPU metrics.
        for i, handle in enumerate(self.gpu_handles):
            try:
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self._metrics[f"gpu_{i}_memory_usage_percentage"].append(
                    round(memory.used / memory.total * 100, 1)
                )
                self._metrics[f"gpu_{i}_memory_usage_megabytes"].append(memory.used / 1e6)
            except pynvml.NVMLError as e:
                _logger.warning(f"Encountered error {e} when trying to collect GPU memory metrics.")

            try:
                device_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self._metrics[f"gpu_{i}_utilization_percentage"].append(device_utilization.gpu)
            except pynvml.NVMLError as e:
                _logger.warning(
                    f"Encountered error {e} when trying to collect GPU utilization metrics."
                )

            try:
                power_milliwatts = pynvml.nvmlDeviceGetPowerUsage(handle)
                power_capacity_milliwatts = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle)
                self._metrics[f"gpu_{i}_power_usage_watts"].append(power_milliwatts / 1000)
                self._metrics[f"gpu_{i}_power_usage_percentage"].append(
                    (power_milliwatts / power_capacity_milliwatts) * 100
                )
            except pynvml.NVMLError as e:
                _logger.warning(
                    f"Encountered error {e} when trying to collect GPU power usage metrics."
                )

    def aggregate_metrics(self):
        return {k: round(sum(v) / len(v), 1) for k, v in self._metrics.items()}
