"""Class for monitoring disk stats."""

import os

import psutil

from mlflow.system_metrics.metrics.base_metrics_monitor import BaseMetricsMonitor


class DiskMonitor(BaseMetricsMonitor):
    """Class for monitoring disk stats."""

    def collect_metrics(self):
        # Get disk usage metrics.
        disk_usage = psutil.disk_usage(os.sep)
        self._metrics["disk_usage_percentage"].append(disk_usage.percent)
        self._metrics["disk_usage_megabytes"].append(disk_usage.used / 1e6)
        self._metrics["disk_available_megabytes"].append(disk_usage.free / 1e6)

    def aggregate_metrics(self):
        return {k: round(sum(v) / len(v), 1) for k, v in self._metrics.items()}
