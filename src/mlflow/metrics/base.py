from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np

from mlflow.utils.annotations import experimental
from mlflow.utils.validation import _is_numeric


def standard_aggregations(scores):
    return {
        "mean": np.mean(scores),
        "variance": np.var(scores),
        "p90": np.percentile(scores, 90),
    }


@experimental
@dataclass
class MetricValue:
    """
    The value of a metric.


    Args:
        scores: The value of the metric per row
        justifications: The justification (if applicable) for the respective score
        aggregate_results: A dictionary mapping the name of the aggregation to its value
    """

    scores: Union[List[str], List[float]] = None
    justifications: List[str] = None
    aggregate_results: Dict[str, float] = None

    def __init__(self, scores=None, justifications=None, aggregate_results=None):
        self.scores = scores
        self.justifications = justifications
        self.aggregate_results = aggregate_results

        if (
            self.aggregate_results is None
            and isinstance(scores, (list, tuple))
            and all(_is_numeric(score) for score in scores)
        ):
            self.aggregate_results = standard_aggregations(scores)
