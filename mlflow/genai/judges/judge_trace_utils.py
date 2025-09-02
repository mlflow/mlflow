"""Utilities for extracting data from MLflow traces."""

from mlflow.entities.trace import Trace
from mlflow.genai.utils.trace_utils import parse_inputs_to_str, parse_outputs_to_str


def extract_request_from_trace(trace: Trace) -> str:
    """
    Extract request text from an MLflow trace object.

    Args:
        trace: MLflow trace object

    Returns:
        Extracted request text as string
    """
    # Check for the existence of trace.data and trace.data.spans and also the inputs
    if not hasattr(trace, "data") or trace.data is None:
        return ""

    if not hasattr(trace.data, "spans") or not trace.data.spans:
        return ""

    first_span = trace.data.spans[0]
    if not hasattr(first_span, "inputs") or first_span.inputs is None:
        return ""

    return parse_inputs_to_str(first_span.inputs)


def extract_response_from_trace(trace: Trace) -> str:
    """
    Extract response text from an MLflow trace object.

    Args:
        trace: MLflow trace object

    Returns:
        Extracted response text as string
    """
    # Check for the existence of trace.data and trace.data.spans and also the outputs
    if not hasattr(trace, "data") or trace.data is None:
        return ""

    if not hasattr(trace.data, "spans") or not trace.data.spans:
        return ""

    first_span = trace.data.spans[0]
    if not hasattr(first_span, "outputs") or first_span.outputs is None:
        return ""

    return parse_outputs_to_str(first_span.outputs)
