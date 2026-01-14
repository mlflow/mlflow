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


def map_scorer_inputs_to_trulens_args(
    metric_name: str,
    inputs: Any = None,
    outputs: Any = None,
    expectations: dict[str, Any] | None = None,
    trace: Trace | None = None,
) -> dict[str, Any]:
    """
    Convert MLflow scorer inputs to TruLens feedback function arguments.

    Args:
        metric_name: Name of the TruLens metric
        inputs: The input to evaluate
        outputs: The output to evaluate
        expectations: Expected values and context for evaluation
        trace: MLflow trace for evaluation

    Returns:
        Dictionary of arguments for the TruLens feedback function
    """
    if trace:
        inputs = resolve_inputs_from_trace(inputs, trace)
        outputs = resolve_outputs_from_trace(outputs, trace)
        expectations = resolve_expectations_from_trace(expectations, trace)

    # Parse inputs and outputs to strings
    input_str = parse_inputs_to_str(inputs) if inputs is not None else ""
    output_str = parse_outputs_to_str(outputs) if outputs is not None else ""

    # Extract context from expectations or trace
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

    # If no context from expectations, try to extract from trace retrieval spans
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

    # Map to TruLens-specific argument names based on metric
    if metric_name == "Groundedness":
        return {
            "source": context_str,
            "statement": output_str,
        }
    elif metric_name == "ContextRelevance":
        return {
            "question": input_str,
            "context": context_str,
        }
    elif metric_name == "AnswerRelevance":
        return {
            "prompt": input_str,
            "response": output_str,
        }
    elif metric_name == "Coherence":
        return {
            "text": output_str,
        }
    else:
        # Generic fallback
        return {
            "input": input_str,
            "output": output_str,
            "context": context_str,
        }


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
