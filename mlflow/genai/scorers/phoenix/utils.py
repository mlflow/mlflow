from __future__ import annotations

from typing import Any

from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.utils.trace_utils import (
    extract_retrieval_context_from_trace,
    parse_inputs_to_str,
    parse_outputs_to_str,
    resolve_expectations_from_trace,
    resolve_inputs_from_trace,
    resolve_outputs_from_trace,
)


class _NoOpRateLimiter:
    """Minimal rate limiter stub for Phoenix model compatibility."""

    def __init__(self):
        self._verbose = False


def check_phoenix_installed():
    """Check if Phoenix is installed and raise an error if not."""
    try:
        import phoenix.evals  # noqa: F401
    except ImportError:
        raise MlflowException.invalid_parameter_value(
            "Phoenix evaluators require the 'arize-phoenix-evals' package. "
            "Install it with: pip install arize-phoenix-evals"
        )


def map_scorer_inputs_to_phoenix_record(
    inputs: Any = None,
    outputs: Any = None,
    expectations: dict[str, Any] | None = None,
    trace: Trace | None = None,
) -> dict[str, Any]:
    """
    Convert MLflow scorer inputs to Phoenix evaluator record format.

    Args:
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

    if inputs is not None:
        record["input"] = parse_inputs_to_str(inputs)

    if outputs is not None:
        record["output"] = parse_outputs_to_str(outputs)

    # Extract reference from expectations using standard MLflow keys
    reference = None
    if expectations:
        reference = (
            expectations.get("expected_response")
            or expectations.get("context")
            or expectations.get("reference")
            or expectations.get("expected_output")
        )

    # Fall back to extracting context from trace retrieval spans
    if not reference and trace:
        if span_id_to_context := extract_retrieval_context_from_trace(trace):
            contexts = []
            for ctx_list in span_id_to_context.values():
                for ctx in ctx_list:
                    if isinstance(ctx, dict) and "content" in ctx:
                        contexts.append(ctx["content"])
                    else:
                        contexts.append(str(ctx))
            reference = "\n".join(contexts) if contexts else None

    if reference:
        record["reference"] = reference

    return record
