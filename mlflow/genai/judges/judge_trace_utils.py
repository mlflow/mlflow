"""Utilities for extracting data from MLflow traces."""

import json
from typing import Any

from mlflow.entities.trace import Trace
from mlflow.genai.utils.trace_utils import parse_inputs_to_str, parse_outputs_to_str


def extract_text_from_data(data: Any, field_type: str) -> str:
    """
    Extract text from various data formats in traces.

    Handles strings, dicts, lists, and None values by converting
    them to appropriate string representations.

    Args:
        data: The data to extract text from
        field_type: Type of field ('request' or 'response') for key priority

    Returns:
        Extracted text as string
    """
    if data is None:
        return ""

    # If already a string, return it
    if isinstance(data, str):
        return data

    # If it's a dict, try to extract the most relevant field
    if isinstance(data, dict):
        # Define the keys to try based on field type
        if field_type == "request":
            keys_to_try = (
                "messages",
                "prompt",
                "query",
                "question",
                "text",
                "input",
                "content",
                "message",
            )
        else:  # response
            keys_to_try = (
                "content",
                "text",
                "answer",
                "response",
                "output",
                "result",
                "message",
                "messages",
            )

        # Try each key in order
        for key in keys_to_try:
            if key in data:
                value = data[key]
                # If the value is a dict or list, convert to string
                if isinstance(value, (dict, list)):
                    return json.dumps(value)
                else:
                    return str(value)

        # If no specific keys found, return the full dict as string
        return json.dumps(data)

    # For any other type, convert to string
    return str(data)


def extract_request_from_trace(trace: Trace) -> str:
    """
    Extract request text from an MLflow trace object.

    Args:
        trace: MLflow trace object

    Returns:
        Extracted request text as string
    """
    return parse_inputs_to_str(trace.data.spans[0].inputs)


def extract_response_from_trace(trace: Trace) -> str:
    """
    Extract response text from an MLflow trace object.

    Args:
        trace: MLflow trace object

    Returns:
        Extracted response text as string
    """
    return parse_outputs_to_str(trace.data.spans[0].outputs)
