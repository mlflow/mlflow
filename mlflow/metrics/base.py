from dataclasses import dataclass
from typing import Dict, List

from mlflow.models import EvaluationMetric as EvaluationMetric
from mlflow.models import make_metric as make_metric


@dataclass
class MetricValue:
    """
    The value of a metric.
    :param scores: The value of the metric per row
    :param justifications: The justification (if applicable) for the respective score
    :param aggregate_results: A dictionary mapping the name of the aggregation to its value
    """

    scores: List[float] = None
    justifications: List[float] = None
    aggregate_results: Dict[str, float] = None
