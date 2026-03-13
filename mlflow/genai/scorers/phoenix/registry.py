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

    For metrics in the registry, uses the registered class name. For unknown metrics,
    attempts to dynamically import <MetricName>Evaluator from phoenix.evals.

    Args:
        metric_name: Name of the metric (e.g., "Hallucination")

    Returns:
        The Phoenix evaluator class

    Raises:
        MlflowException: If the metric cannot be imported or phoenix is not installed
    """
    check_phoenix_installed()

    import phoenix.evals as phoenix_evals

    if metric_name in _METRIC_REGISTRY:
        evaluator_class_name = _METRIC_REGISTRY[metric_name]
    else:
        # Attempt dynamic import for metrics not in registry
        evaluator_class_name = f"{metric_name}Evaluator"

    try:
        return getattr(phoenix_evals, evaluator_class_name)
    except AttributeError:
        available_metrics = ", ".join(sorted(_METRIC_REGISTRY.keys()))
        raise MlflowException.invalid_parameter_value(
            f"Unknown Phoenix metric: '{metric_name}'. Could not find '{evaluator_class_name}' "
            f"in 'phoenix.evals'. Available pre-configured metrics: {available_metrics}"
        )
