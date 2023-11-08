"""Class for monitoring CPU stats."""

from mlflow.system_metrics.metrics.base_metrics_monitor import BaseMetricsMonitor
from mlflow.utils import process


class CPUMonitor(BaseMetricsMonitor):
    """Class for monitoring CPU stats."""

    def collect_metrics(self):
        # Get CPU metrics.
        cpu_percent = process.cpu_percent()
        self._metrics["cpu_utilization_percentage"].append(cpu_percent)

        system_memory = process.virtual_memory()
        self._metrics["system_memory_usage_megabytes"].append(system_memory.used / 1e6)
        self._metrics["system_memory_usage_percentage"].append(
            system_memory.used / system_memory.total * 100
        )

    def aggregate_metrics(self):
        return {k: round(sum(v) / len(v), 1) for k, v in self._metrics.items()}
