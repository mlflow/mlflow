"""Utility functions for Phoenix integration."""

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


def map_scorer_inputs_to_phoenix_record(
    metric_name: str,
    inputs: Any = None,
    outputs: Any = None,
    expectations: dict[str, Any] | None = None,
    trace: Trace | None = None,
) -> dict[str, Any]:
    """
    Convert MLflow scorer inputs to Phoenix evaluator record format.

    Args:
        metric_name: Name of the Phoenix metric
        inputs: The input to evaluate
        outputs: The output to evaluate
        expectations: Expected values and context for evaluation
        trace: MLflow trace for evaluation

    Returns:
        Dictionary formatted for Phoenix evaluator
    """
    if trace:
        inputs = resolve_inputs_from_trace(inputs, trace)
        outputs = resolve_outputs_from_trace(outputs, trace)
        expectations = resolve_expectations_from_trace(expectations, trace)

    record = {}

    # Parse inputs to string
    if inputs is not None:
        input_str = parse_inputs_to_str(inputs)
        record["input"] = input_str

    # Parse outputs to string
    if outputs is not None:
        output_str = parse_outputs_to_str(outputs)
        record["output"] = output_str

    # Toxicity evaluator expects text in 'input' field, not 'output'
    # If only outputs provided and no input, use output as input for Toxicity
    if metric_name == "Toxicity" and "input" not in record and "output" in record:
        record["input"] = record["output"]

    # Handle context/reference from expectations or trace
    reference = None
    if expectations:
        reference = (
            expectations.get("context")
            or expectations.get("reference")
            or expectations.get("expected_output")
        )

    # If no reference from expectations, try to extract from trace retrieval spans
    if not reference and trace:
        span_id_to_context = extract_retrieval_context_from_trace(trace)
        if span_id_to_context:
            contexts = []
            for ctx_list in span_id_to_context.values():
                for ctx in ctx_list:
                    if isinstance(ctx, dict) and "content" in ctx:
                        contexts.append(ctx["content"])
                    else:
                        contexts.append(str(ctx))
            if contexts:
                reference = "\n".join(contexts)

    if reference:
        record["reference"] = reference

    return record
