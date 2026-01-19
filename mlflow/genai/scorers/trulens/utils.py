from __future__ import annotations

from typing import Any

from mlflow.entities.trace import Trace
from mlflow.genai.utils.trace_utils import (
    extract_retrieval_context_from_trace,
    parse_inputs_to_str,
    parse_outputs_to_str,
    resolve_expectations_from_trace,
    resolve_inputs_from_trace,
    resolve_outputs_from_trace,
)

# Mapping from metric name to TruLens argument names
# Each metric expects different argument names for its feedback function
_ARG_MAPPING: dict[str, dict[str, str]] = {
    "Groundedness": {"context": "source", "output": "statement"},
    "ContextRelevance": {"input": "question", "context": "context"},
    "AnswerRelevance": {"input": "prompt", "output": "response"},
    "Coherence": {"output": "text"},
}


def map_scorer_inputs_to_trulens_args(
    metric_name: str,
    inputs: Any = None,
    outputs: Any = None,
    expectations: dict[str, Any] | None = None,
    trace: Trace | None = None,
) -> dict[str, Any]:
    if trace:
        inputs = resolve_inputs_from_trace(inputs, trace)
        outputs = resolve_outputs_from_trace(outputs, trace)
        expectations = resolve_expectations_from_trace(expectations, trace)

    input_str = parse_inputs_to_str(inputs) if inputs is not None else ""
    output_str = parse_outputs_to_str(outputs) if outputs is not None else ""
    context_str = _extract_context(expectations, trace)

    return _build_trulens_args(
        metric_name=metric_name,
        input_str=input_str,
        output_str=output_str,
        context_str=context_str,
    )


def _build_trulens_args(
    metric_name: str,
    input_str: str,
    output_str: str,
    context_str: str,
) -> dict[str, Any]:
    arg_mapping = _ARG_MAPPING.get(metric_name, {})
    generic_values = {"input": input_str, "output": output_str, "context": context_str}

    return {
        trulens_arg: generic_values[generic_key]
        for generic_key, trulens_arg in arg_mapping.items()
        if generic_values.get(generic_key)
    }


def _extract_context(
    expectations: dict[str, Any] | None,
    trace: Trace | None,
) -> str:
    """Extract context from expectations or trace retrieval spans."""
    context_str = ""

    if expectations:
        context = (
            expectations.get("context")
            or expectations.get("reference")
            or expectations.get("expected_output")
        )
        if context:
            if isinstance(context, list):
                context_str = "\n".join(str(c) for c in context)
            else:
                context_str = str(context)

    if not context_str and trace:
        if span_id_to_context := extract_retrieval_context_from_trace(trace):
            contexts = []
            for ctx_list in span_id_to_context.values():
                for ctx in ctx_list:
                    if isinstance(ctx, dict) and "content" in ctx:
                        contexts.append(ctx["content"])
                    else:
                        contexts.append(str(ctx))
            if contexts:
                context_str = "\n".join(contexts)

    return context_str


def format_trulens_rationale(reasons: dict[str, Any] | None) -> str | None:
    """
    Format TruLens reasons dict into a readable rationale string.

    Args:
        reasons: Dict of reasons from TruLens

    Returns:
        Formatted rationale string or None if no reasons
    """
    if not reasons:
        return None

    parts = []
    for key, value in reasons.items():
        value_str = "; ".join(str(v) for v in value) if isinstance(value, list) else str(value)
        parts.append(f"{key}: {value_str}")

    return " | ".join(parts) if parts else None
