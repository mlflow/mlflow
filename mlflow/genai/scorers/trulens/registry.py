from __future__ import annotations

import re
from typing import Any

# Mapping: metric name -> (feedback method name, argument mapping)
# Argument mapping: generic key -> TruLens-specific argument name
_METRIC_REGISTRY: dict[str, tuple[str, dict[str, str]]] = {
    # RAG metrics
    "Groundedness": (
        "groundedness_measure_with_cot_reasons",
        {"context": "source", "output": "statement"},
    ),
    "ContextRelevance": (
        "context_relevance_with_cot_reasons",
        {"input": "question", "context": "context"},
    ),
    "AnswerRelevance": (
        "relevance_with_cot_reasons",
        {"input": "prompt", "output": "response"},
    ),
    "Coherence": (
        "coherence_with_cot_reasons",
        {"output": "text"},
    ),
    # Agent trace metrics (no argument mapping needed - they use trace directly)
    "logical_consistency": ("logical_consistency_with_cot_reasons", {}),
    "execution_efficiency": ("execution_efficiency_with_cot_reasons", {}),
    "plan_adherence": ("plan_adherence_with_cot_reasons", {}),
    "plan_quality": ("plan_quality_with_cot_reasons", {}),
    "tool_selection": ("tool_selection_with_cot_reasons", {}),
    "tool_calling": ("tool_calling_with_cot_reasons", {}),
}


def get_feedback_method_name(metric_name: str) -> str:
    if metric_name in _METRIC_REGISTRY:
        return _METRIC_REGISTRY[metric_name][0]
    # Convert CamelCase to snake_case and append _with_cot_reasons
    snake_case = re.sub(r"(?<!^)(?=[A-Z])", "_", metric_name).lower()
    return f"{snake_case}_with_cot_reasons"


def get_argument_mapping(metric_name: str) -> dict[str, str]:
    if metric_name in _METRIC_REGISTRY:
        return _METRIC_REGISTRY[metric_name][1]
    # Return default empty mapping for unknown metrics
    return {}


def build_trulens_args(
    metric_name: str,
    input_str: str,
    output_str: str,
    context_str: str,
) -> dict[str, Any]:
    arg_mapping = get_argument_mapping(metric_name)
    generic_values = {"input": input_str, "output": output_str, "context": context_str}

    return {
        trulens_arg: generic_values[generic_key]
        for generic_key, trulens_arg in arg_mapping.items()
        if generic_values.get(generic_key)
    }
