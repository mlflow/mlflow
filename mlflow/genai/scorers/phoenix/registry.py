from __future__ import annotations

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.phoenix.utils import check_phoenix_installed

_METRIC_REGISTRY = {
    "Hallucination": "HallucinationEvaluator",
    "Relevance": "RelevanceEvaluator",
    "Toxicity": "ToxicityEvaluator",
    "QA": "QAEvaluator",
    "Summarization": "SummarizationEvaluator",
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

    evaluator_class_name = _METRIC_REGISTRY[metric_name]
    return getattr(phoenix_evals, evaluator_class_name)
