"""Base class of system metrics monitor."""


class BaseMetricsMonitor:
    """Base class of system metrics monitor.

    Args:
        name: string, name of the monitor.
    """

    def __init__(self, name):
        self.name = name

        self._metrics = {}

    def collect_metrics(self):
        raise NotImplementedError

    @property
    def metrics(self):
        return self._metrics
