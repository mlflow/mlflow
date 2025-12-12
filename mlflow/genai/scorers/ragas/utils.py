"""Utility functions and constants for RAGAS integration."""

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

RAGAS_NOT_INSTALLED_ERROR_MESSAGE = (
    "RAGAS metrics require the 'ragas' package. Please install it with: pip install ragas"
)


def map_scorer_inputs_to_ragas_sample(
    inputs: Any = None,
    outputs: Any = None,
    expectations: dict[str, Any] | None = None,
    trace: Trace | None = None,
):
    """
    Convert MLflow scorer inputs to RAGAS SingleTurnSample format.

    Args:
        inputs: The input to evaluate
        outputs: The output to evaluate
        expectations: Expected values and context for evaluation
        trace: MLflow trace for evaluation

    Returns:
        RAGAS SingleTurnSample object
    """
    from ragas.dataset_schema import SingleTurnSample

    if trace:
        inputs = resolve_inputs_from_trace(inputs, trace)
        outputs = resolve_outputs_from_trace(outputs, trace)
        expectations = resolve_expectations_from_trace(expectations, trace)

    user_input = parse_inputs_to_str(inputs) if inputs is not None else None
    response = parse_outputs_to_str(outputs) if outputs is not None else None

    span_id_to_context = extract_retrieval_context_from_trace(trace) if trace else {}
    retrieved_contexts = [str(ctx) for contexts in span_id_to_context.values() for ctx in contexts]

    reference = None
    if expectations and "expected_output" in expectations:
        reference = parse_outputs_to_str(expectations["expected_output"])

    return SingleTurnSample(
        user_input=user_input,
        response=response,
        retrieved_contexts=retrieved_contexts or None,
        reference=reference,
        reference_contexts=retrieved_contexts or None,
    )
