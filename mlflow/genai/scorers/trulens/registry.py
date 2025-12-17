"""Registry of TruLens feedback functions and their configurations."""

from __future__ import annotations

from mlflow.exceptions import MlflowException

# Map metric names to their feedback function configurations
_METRIC_REGISTRY = {
    "Groundedness": {
        "method": "groundedness_measure_with_cot_reasons",
        "args": ["source", "statement"],
        "description": "Evaluates if outputs are grounded in context",
    },
    "ContextRelevance": {
        "method": "context_relevance_with_cot_reasons",
        "args": ["question", "context"],
        "description": "Evaluates context relevance to query",
    },
    "AnswerRelevance": {
        "method": "relevance_with_cot_reasons",
        "args": ["prompt", "response"],
        "description": "Evaluates answer relevance to query",
    },
    "Coherence": {
        "method": "coherence_with_cot_reasons",
        "args": ["text"],
        "description": "Evaluates logical flow of outputs",
    },
}


def get_feedback_method_name(metric_name: str) -> str:
    """
    Get the TruLens feedback method name for a given metric.

    Args:
        metric_name: Name of the metric (e.g., "Groundedness")

    Returns:
        The TruLens method name to call

    Raises:
        MlflowException: If the metric is not supported
    """
    if metric_name not in _METRIC_REGISTRY:
        available_metrics = ", ".join(sorted(_METRIC_REGISTRY.keys()))
        raise MlflowException.invalid_parameter_value(
            f"Unknown TruLens metric: '{metric_name}'. Available metrics: {available_metrics}"
        )

    return _METRIC_REGISTRY[metric_name]["method"]


def get_metric_config(metric_name: str) -> dict:
    """
    Get the configuration for a given metric.

    Args:
        metric_name: Name of the metric

    Returns:
        Configuration dictionary with method, args, and description
    """
    if metric_name not in _METRIC_REGISTRY:
        available_metrics = ", ".join(sorted(_METRIC_REGISTRY.keys()))
        raise MlflowException.invalid_parameter_value(
            f"Unknown TruLens metric: '{metric_name}'. Available metrics: {available_metrics}"
        )

    return _METRIC_REGISTRY[metric_name]
