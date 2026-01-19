from __future__ import annotations

from mlflow.exceptions import MlflowException

# Simple mapping: metric name -> TruLens feedback method name
_METRIC_REGISTRY: dict[str, str] = {
    "Groundedness": "groundedness_measure_with_cot_reasons",
    "ContextRelevance": "context_relevance_with_cot_reasons",
    "AnswerRelevance": "relevance_with_cot_reasons",
    "Coherence": "coherence_with_cot_reasons",
}


def get_feedback_method_name(metric_name: str) -> str:
    if metric_name not in _METRIC_REGISTRY:
        available_metrics = ", ".join(sorted(_METRIC_REGISTRY.keys()))
        raise MlflowException.invalid_parameter_value(
            f"Unknown TruLens metric: '{metric_name}'. Available metrics: {available_metrics}"
        )
    return _METRIC_REGISTRY[metric_name]
