from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mlflow.exceptions import MlflowException


@dataclass
class TruLensMetricConfig:
    """Configuration for a TruLens metric."""

    method_name: str
    arg_mapping: dict[str, str]  # Maps generic names to TruLens-specific arg names


# Map metric names to their TruLens configuration
# arg_mapping keys: "input", "output", "context" -> TruLens-specific argument names
_METRIC_REGISTRY: dict[str, TruLensMetricConfig] = {
    "Groundedness": TruLensMetricConfig(
        method_name="groundedness_measure_with_cot_reasons",
        arg_mapping={"context": "source", "output": "statement"},
    ),
    "ContextRelevance": TruLensMetricConfig(
        method_name="context_relevance_with_cot_reasons",
        arg_mapping={"input": "question", "context": "context"},
    ),
    "AnswerRelevance": TruLensMetricConfig(
        method_name="relevance_with_cot_reasons",
        arg_mapping={"input": "prompt", "output": "response"},
    ),
    "Coherence": TruLensMetricConfig(
        method_name="coherence_with_cot_reasons",
        arg_mapping={"output": "text"},
    ),
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
    config = _get_metric_config(metric_name)
    return config.method_name


def get_metric_arg_mapping(metric_name: str) -> dict[str, str]:
    """
    Get the argument mapping for a given metric.

    Args:
        metric_name: Name of the metric (e.g., "Groundedness")

    Returns:
        Dict mapping generic arg names (input, output, context) to TruLens arg names

    Raises:
        MlflowException: If the metric is not supported
    """
    config = _get_metric_config(metric_name)
    return config.arg_mapping


def _get_metric_config(metric_name: str) -> TruLensMetricConfig:
    if metric_name not in _METRIC_REGISTRY:
        available_metrics = ", ".join(sorted(_METRIC_REGISTRY.keys()))
        raise MlflowException.invalid_parameter_value(
            f"Unknown TruLens metric: '{metric_name}'. Available metrics: {available_metrics}"
        )
    return _METRIC_REGISTRY[metric_name]


def build_trulens_args(
    metric_name: str,
    input_str: str,
    output_str: str,
    context_str: str,
) -> dict[str, Any]:
    """
    Build TruLens feedback function arguments using the metric's arg mapping.

    Args:
        metric_name: Name of the TruLens metric
        input_str: The input/question string
        output_str: The output/response string
        context_str: The context/source string

    Returns:
        Dict of arguments for the TruLens feedback function
    """
    arg_mapping = get_metric_arg_mapping(metric_name)

    generic_values = {
        "input": input_str,
        "output": output_str,
        "context": context_str,
    }

    return {
        trulens_arg: generic_values[generic_key]
        for generic_key, trulens_arg in arg_mapping.items()
        if generic_values.get(generic_key)
    }
