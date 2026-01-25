from __future__ import annotations

from mlflow.exceptions import MlflowException

# Mapping: metric name -> TruLens feedback method name
_METRIC_REGISTRY: dict[str, str] = {
    # RAG metrics
    "Groundedness": "groundedness_measure_with_cot_reasons",
    "ContextRelevance": "context_relevance_with_cot_reasons",
    "AnswerRelevance": "relevance_with_cot_reasons",
    "Coherence": "coherence_with_cot_reasons",
    # Agent trace metrics
    "logical_consistency": "logical_consistency_with_cot_reasons",
    "execution_efficiency": "execution_efficiency_with_cot_reasons",
    "plan_adherence": "plan_adherence_with_cot_reasons",
    "plan_quality": "plan_quality_with_cot_reasons",
    "tool_selection": "tool_selection_with_cot_reasons",
    "tool_calling": "tool_calling_with_cot_reasons",
}


def get_feedback_method_name(metric_name: str) -> str:
    if metric_name not in _METRIC_REGISTRY:
        available_metrics = ", ".join(sorted(_METRIC_REGISTRY.keys()))
        raise MlflowException.invalid_parameter_value(
            f"Unknown TruLens metric: '{metric_name}'. Available metrics: {available_metrics}"
        )
    return _METRIC_REGISTRY[metric_name]
