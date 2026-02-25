import logging

from mlflow.utils.autologging_utils import (
    BatchMetricsLogger,
    ExceptionSafeAbstractClass,
)

_logger = logging.getLogger(__name__)


class AutologCallback(metaclass=ExceptionSafeAbstractClass):
    def __init__(self, metrics_logger: BatchMetricsLogger) -> None:
        self.metrics_logger = metrics_logger

    def _iter_metrics(self, info) -> dict[str, float]:
        """
        Extract metrics from the info object and return them as a dictionary.
        """
        return {
            f"{dataset_name}-{metric_name}".replace("@", "_at_"): metric_values[-1]
            for dataset_name, metric_dict in info.metrics.items()
            for metric_name, metric_values in metric_dict.items()
        }

    def after_iteration(self, info) -> bool:
        """
        Called by CatBoost after each training iteration.

        Args:
            info: A types.SimpleNamespace with attributes:
                - iteration (int): current 0-indexed iteration number
                - metrics (dict): nested dict of metric histories per dataset, e.g.
                    {
                        "learn": {"Logloss": [0.6, 0.5, 0.45], ...},
                        "validation": {"Logloss": [0.7, 0.6, 0.55], ...},
                    }
                  Each metric maps to a list of all values from iteration 0 to current.

        Returns:
            True to continue training, False to stop.
        """

        self.metrics_logger.record_metrics(
            metrics=self._iter_metrics(info), step=info.iteration
        )

        return True
