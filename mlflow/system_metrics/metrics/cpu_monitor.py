"""Class for monitoring CPU stats."""

import psutil

from mlflow.system_metrics.metrics.base_metrics_monitor import BaseMetricsMonitor


class CPUMonitor(BaseMetricsMonitor):
    def __init__(self, name="cpu"):
        super().__init__(name)

    def collect_metrics(self):
        # Get metrics for the system.
        cpu_percent = psutil.cpu_percent()
        self._metrics["cpu_percentage"] = cpu_percent

        system_memory = psutil.virtual_memory()._asdict()
        for k, v in system_memory.items():
            if "percent" in k:
                self._metrics[f"cpu_memory_{k}"] = v
            else:
                # Convert bytes to MB.
                self._metrics[f"cpu_memory_{k}"] = int(v / 1e6)
