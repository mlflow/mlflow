"""Registry of Phoenix evaluators and their configurations."""

from __future__ import annotations

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.phoenix.utils import check_phoenix_installed

_METRIC_REGISTRY = {
    "Hallucination": {
        "evaluator_class": "HallucinationEvaluator",
        "positive_label": "factual",
        "negative_label": "hallucinated",
        "required_fields": ["input", "output", "reference"],
    },
    "Relevance": {
        "evaluator_class": "RelevanceEvaluator",
        "positive_label": "relevant",
        "negative_label": "irrelevant",
        "required_fields": ["input", "reference"],
    },
    "Toxicity": {
        "evaluator_class": "ToxicityEvaluator",
        "positive_label": "non-toxic",
        "negative_label": "toxic",
        "required_fields": ["input"],
    },
    "QA": {
        "evaluator_class": "QAEvaluator",
        "positive_label": "correct",
        "negative_label": "incorrect",
        "required_fields": ["input", "output", "reference"],
    },
    "Summarization": {
        "evaluator_class": "SummarizationEvaluator",
        "positive_label": "good",
        "negative_label": "bad",
        "required_fields": ["input", "output"],
    },
}


def get_evaluator_class(metric_name: str):
    """
    Get the Phoenix evaluator class for a given metric name.

    Args:
        metric_name: Name of the metric (e.g., "Hallucination")

    Returns:
        The Phoenix evaluator class

    Raises:
        MlflowException: If the metric is not supported
    """
    check_phoenix_installed()

    if metric_name not in _METRIC_REGISTRY:
        available_metrics = ", ".join(sorted(_METRIC_REGISTRY.keys()))
        raise MlflowException.invalid_parameter_value(
            f"Unknown Phoenix metric: '{metric_name}'. Available metrics: {available_metrics}"
        )

    import phoenix.evals as phoenix_evals

    evaluator_class_name = _METRIC_REGISTRY[metric_name]["evaluator_class"]
    return getattr(phoenix_evals, evaluator_class_name)


def get_metric_config(metric_name: str) -> dict:
    """
    Get the configuration for a given metric.

    Args:
        metric_name: Name of the metric

    Returns:
        Configuration dictionary with positive_label and required_fields
    """
    if metric_name not in _METRIC_REGISTRY:
        available_metrics = ", ".join(sorted(_METRIC_REGISTRY.keys()))
        raise MlflowException.invalid_parameter_value(
            f"Unknown Phoenix metric: '{metric_name}'. Available metrics: {available_metrics}"
        )

    return _METRIC_REGISTRY[metric_name]
