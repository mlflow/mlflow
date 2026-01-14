from __future__ import annotations

from mlflow.exceptions import MlflowException

# Map metric names to their TruLens feedback method names
_METRIC_REGISTRY: dict[str, str] = {
    "Groundedness": "groundedness_measure_with_cot_reasons",
    "ContextRelevance": "context_relevance_with_cot_reasons",
    "AnswerRelevance": "relevance_with_cot_reasons",
    "Coherence": "coherence_with_cot_reasons",
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

    return _METRIC_REGISTRY[metric_name]
