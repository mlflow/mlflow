"""Base class of system metrics monitor."""
import abc
from collections import defaultdict


class BaseMetricsMonitor(abc.ABC):
    """Base class of system metrics monitor."""

    def __init__(self):
        self._metrics = defaultdict(list)

    @abc.abstractmethod
    def collect_metrics(self):
        """Method to collect metrics.

        Subclass should implement this method to collect metrics and store in `self._metrics`.
        """

    @abc.abstractmethod
    def aggregate_metrics(self):
        """Method to aggregate metrics.

        Subclass should implement this method to aggregate the metrics and return it in a dict.
        """

    @property
    def metrics(self):
        return self._metrics

    def clear_metrics(self):
        self._metrics.clear()
