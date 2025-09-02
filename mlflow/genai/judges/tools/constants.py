"""
Constants for MLflow GenAI judge tools.

This module contains constant values used across the judge tools system,
providing a single reference point for tool names and other constants.
"""

from mlflow.utils.annotations import experimental


# Tool names
@experimental(version="3.4.0")
class ToolNames:
    """Registry of judge tool names."""

    GET_TRACE_INFO = "get_trace_info"
    GET_ROOT_SPAN = "get_root_span"
    GET_SPAN = "get_span"
    LIST_SPANS = "list_spans"
    SEARCH_TRACE_REGEX = "search_trace_regex"
