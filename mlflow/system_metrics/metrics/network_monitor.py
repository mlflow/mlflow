"""Class for monitoring network stats."""


import psutil

from mlflow.system_metrics.metrics.base_metrics_monitor import BaseMetricsMonitor


class NetworkMonitor(BaseMetricsMonitor):
    def __init__(self, name="network"):
        super().__init__(name)

    def collect_metrics(self):
        # Get network usage metrics.
        network_usage = psutil.net_io_counters()._asdict()
        self._metrics["network_receive_megabytes"].append(int(network_usage["bytes_recv"] / 1e6))
        self._metrics["network_transmit_megabytes"].append(int(network_usage["bytes_sent"] / 1e6))
