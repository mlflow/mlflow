from dataclasses import dataclass
from typing import Dict, List

from mlflow.utils.annotations import experimental


@experimental
@dataclass
class MetricValue:
    """
    The value of a metric.

    :param scores: The value of the metric per row
    :param justifications: The justification (if applicable) for the respective score
    :param aggregate_results: A dictionary mapping the name of the aggregation to its value
    """

    scores: List[float] = None
    justifications: List[str] = None
    aggregate_results: Dict[str, float] = None
