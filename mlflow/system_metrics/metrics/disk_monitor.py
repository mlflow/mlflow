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
        metrics = {}
        for name, values in self._metrics.items():
            if len(values) > 0:
                metrics[name] = round(sum(values) / len(values), 1)
        return metrics
