"""Base class of system metrics monitor."""
from collections import defaultdict


class BaseMetricsMonitor:
    """Base class of system metrics monitor.

    Args:
        name: string, name of the monitor.
    """

    def __init__(self, name):
        self.name = name

        self._metrics = defaultdict(list)

    def collect_metrics(self):
        raise NotImplementedError

    def aggregate_metrics(self):
        raise NotImplementedError

    @property
    def metrics(self):
        return self._metrics

    def clear_metrics(self):
        self._metrics.clear()
