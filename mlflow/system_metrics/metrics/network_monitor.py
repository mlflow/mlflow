"""Class for monitoring network stats."""

import psutil
import os

from mlflow.system_metrics.metrics.base_metrics_monitor import BaseMetricsMonitor


class NetworkMonitor(BaseMetricsMonitor):
    def __init__(self, name="network"):
        super().__init__(name)

    def collect_metrics(self):
        # Get network usage metrics.
        network_usage = psutil.net_io_counters()._asdict()
        for k, v in network_usage.items():
            if "bytes" in k:
                # Convert bytes to MB.
                self._metrics[f"network_{k}"] = int(v / 1e6)
            else:
                self._metrics[f"network_{k}"] = v
