"""Class for monitoring disk stats."""

import psutil
import os

from mlflow.system_metrics.metrics.base_metrics_monitor import BaseMetricsMonitor


class DiskMonitor(BaseMetricsMonitor):
    """Class for monitoring disk stats."""

    def __init__(self, name="disk"):
        super().__init__(name)

    def collect_metrics(self):
        # Get disk usage metrics.
        disk_usage = psutil.disk_usage(os.sep)._asdict()
        for k, v in disk_usage.items():
            if "percent" in k:
                self._metrics[f"disk_memory_{k}"].append(v)
            else:
                # Convert bytes to MB.
                self._metrics[f"disk_memory_{k}"].append(int(v / 1e6))

    def aggregate_metrics(self):
        for name, values in self._metrics.items():
            self._metrics[name] = sum(values) / len(values)
