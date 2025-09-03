"""Utilities for extracting data from MLflow traces."""

from mlflow.entities.trace import Trace
from mlflow.genai.utils.trace_utils import parse_inputs_to_str, parse_outputs_to_str


def extract_request_from_trace(trace: Trace) -> str | None:
    """
    Extract request text from an MLflow trace object.

    Args:
        trace: MLflow trace object

    Returns:
        Extracted request text as string, or None if no spans
    """
    if not trace.data.spans:
        return None

    return parse_inputs_to_str(trace.data.spans[0].inputs)


def extract_response_from_trace(trace: Trace) -> str | None:
    """
    Extract response text from an MLflow trace object.

    Args:
        trace: MLflow trace object

    Returns:
        Extracted response text as string, or None if no spans
    """
    if not trace.data.spans:
        return None

    return parse_outputs_to_str(trace.data.spans[0].outputs)
