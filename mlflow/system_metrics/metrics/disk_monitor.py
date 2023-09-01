"""Class for monitoring CPU stats."""

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
                self._metrics[f"disk_memory_{k}"] = v
            else:
                # Convert bytes to MB.
                self._metrics[f"disk_memory_{k}"] = int(v / 1e6)
