"""Generate the metrics logged into MLflow."""

import collections
import logging

import numpy as np

from mlflow.entities.assessment import Feedback
from mlflow.genai.evaluation.entities import EvalResult
from mlflow.genai.judges.builtin import CategoricalRating
from mlflow.genai.scorers.base import AggregationFunc, Scorer

_logger = logging.getLogger(__name__)


_AGGREGATE_FUNCTIONS = {
    "min": np.min,
    "max": np.max,
    "mean": np.mean,
    "median": np.median,
    "variance": np.var,
    "p90": lambda x: np.percentile(x, 90) if x else None,
}


def compute_aggregated_metrics(
    eval_results: list[EvalResult],
    scorers: list[Scorer],
) -> dict[str, float]:
    """
    Generates the aggregated per-run metrics from the evaluation result for each row.

    Args:
        eval_results: List of EvalResult objects
        scorers: List of scorers

    Returns:
        Dictionary of aggregated metrics (metric name -> aggregated value)
    """
    # Collect assessment values in a flat dictionary
    assessment_values: dict[str, list[float]] = collections.defaultdict(list)
    for eval_result in eval_results:
        for assessment in eval_result.assessments:
            if isinstance(assessment, Feedback):
                value = _cast_assessment_value_to_float(assessment)
                if value is not None:  # Exclude None from aggregation
                    assessment_values[assessment.name].append(value)

    # List all aggregations to compute for each scorer
    scorer_aggregations = {}
    for scorer in scorers:
        if scorer.aggregations is not None:
            scorer_aggregations[scorer.name] = scorer.aggregations
        else:
            scorer_aggregations[scorer.name] = ["mean"]  # default to compute mean only

    # Compute aggregates
    result = {}
    for name, values in assessment_values.items():
        if not values:
            continue

        # Get the function name from the returned assessment name.
        scorer_function_name = name.split("/", 1)[-1]

        # Compute aggregations for the scorer, defaulting to just ["mean"]
        aggregations_to_compute = scorer_aggregations.get(scorer_function_name, ["mean"])
        aggregation_results = _compute_aggregations(values, aggregations_to_compute)

        # Each aggregation should be logged as a separate metric
        for agg_name, agg_value in aggregation_results.items():
            result[f"{name}/{agg_name}"] = agg_value

    return result


def _cast_assessment_value_to_float(assessment: Feedback) -> float | None:
    """Cast the value of an assessment to a float."""
    if isinstance(assessment.value, (int, float, bool)):
        return float(assessment.value)
    elif (
        isinstance(assessment.value, str)
        and CategoricalRating(assessment.value.lower()) != CategoricalRating.UNKNOWN
    ):
        return float(assessment.value.lower() == CategoricalRating.YES)
    else:
        _logger.error(f"Invalid assessment value for {assessment.name}: {assessment.value}")


def _compute_aggregations(
    scores: list[float], aggregations: list[str | AggregationFunc]
) -> dict[str, float]:
    """Compute aggregate statistics for a list of scores based on specified aggregations.

    Args:
        scores: List of numeric scores to aggregate
        aggregations: List of aggregation types to compute (e.g. ["min", "max", "mean"])

    Returns:
        Dictionary mapping aggregation names to computed values
    """
    results = {}
    for aggregation in aggregations:
        # Aggregations specified as strings, e.g., "mean"
        if isinstance(aggregation, str):
            if aggregation not in _AGGREGATE_FUNCTIONS:
                raise ValueError(f"Invalid aggregation: {aggregation}")
            results[aggregation] = _AGGREGATE_FUNCTIONS[aggregation](scores)
        # Aggregations specified as a custom function.
        else:
            try:
                results[aggregation.__name__] = aggregation(scores)
            except Exception as e:
                _logger.error(f"Error computing aggregation {aggregation} due to: {e}")
                continue

    return results
