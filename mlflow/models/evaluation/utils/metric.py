import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from mlflow.metrics.base import MetricValue
from mlflow.models.evaluation.base import EvaluationMetric

_logger = logging.getLogger(__name__)


@dataclass
class MetricDefinition:
    """
    A namedtuple representing a metric function and its properties.

    function : the metric function
    name : the name of the metric function
    index : the index of the function in the ``extra_metrics`` argument of mlflow.evaluate
    """

    function: Callable[..., Any]
    name: str
    index: int
    version: Optional[str] = None
    genai_metric_args: Optional[dict[str, Any]] = None

    @classmethod
    def from_index_and_metric(cls, index: int, metric: EvaluationMetric):
        return cls(
            function=metric.eval_fn,
            index=index,
            name=metric.name,
            version=metric.version,
            genai_metric_args=metric.genai_metric_args,
        )

    def evaluate(self, eval_fn_args) -> Optional[MetricValue]:
        """
        This function calls the metric function and performs validations on the returned
        result to ensure that they are in the expected format. It will warn and will not log metrics
        that are in the wrong format.

        Args:
            eval_fn_args: A dictionary of args needed to compute the eval metrics.

        Returns:
            MetricValue
        """
        if self.index < 0:
            exception_header = f"Did not log builtin metric '{self.name}' because it"
        else:
            exception_header = (
                f"Did not log metric '{self.name}' at index "
                f"{self.index} in the `extra_metrics` parameter because it"
            )

        metric: MetricValue = self.function(*eval_fn_args)

        def _is_numeric(value):
            return isinstance(value, (int, float, np.number))

        def _is_string(value):
            return isinstance(value, str)

        if metric is None:
            _logger.warning(f"{exception_header} returned None.")
            return

        if _is_numeric(metric):
            return MetricValue(aggregate_results={self.name: metric})

        if not isinstance(metric, MetricValue):
            _logger.warning(f"{exception_header} did not return a MetricValue.")
            return

        scores = metric.scores
        justifications = metric.justifications
        aggregates = metric.aggregate_results

        if scores is not None:
            if not isinstance(scores, list):
                _logger.warning(
                    f"{exception_header} must return MetricValue with scores as a list."
                )
                return
            if any(not (_is_numeric(s) or _is_string(s) or s is None) for s in scores):
                _logger.warning(
                    f"{exception_header} must return MetricValue with numeric or string scores."
                )
                return

        if justifications is not None:
            if not isinstance(justifications, list):
                _logger.warning(
                    f"{exception_header} must return MetricValue with justifications as a list."
                )
                return
            if any(not (_is_string(just) or just is None) for just in justifications):
                _logger.warning(
                    f"{exception_header} must return MetricValue with string justifications."
                )
                return

        if aggregates is not None:
            if not isinstance(aggregates, dict):
                _logger.warning(
                    f"{exception_header} must return MetricValue with aggregate_results as a dict."
                )
                return

            if any(
                not (isinstance(k, str) and (_is_numeric(v) or v is None))
                for k, v in aggregates.items()
            ):
                _logger.warning(
                    f"{exception_header} must return MetricValue with aggregate_results with "
                    "str keys and numeric values."
                )
                return

        return metric
