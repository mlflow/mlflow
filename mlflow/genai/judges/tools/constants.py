"""
Constants for MLflow GenAI judge tools.

This module contains constant values used across the judge tools system,
providing a single reference point for tool names and other constants.
"""


# Tool names
class ToolNames:
    """Registry of judge tool names."""

    GET_TRACE_INFO = "get_trace_info"
    GET_ROOT_SPAN = "get_root_span"
    GET_SPAN = "get_span"
    LIST_SPANS = "list_spans"
    SEARCH_TRACE_REGEX = "search_trace_regex"
    GET_SPAN_PERFORMANCE_AND_TIMING_REPORT = "get_span_performance_and_timing_report"
    READ_SKILL_MARKDOWN_CONTENT = "read_skill_markdown_content"
    READ_SKILL_COMPANION_FILE = "read_skill_companion_file"
    _GET_TRACES_IN_SESSION = "_get_traces_in_session"
    _SEARCH_TRACES = "_search_traces"


# Tool names for skill-related tools. Used by adapters to filter which tools
# are exposed when skills are provided without a trace.
SKILL_TOOL_NAMES = {ToolNames.READ_SKILL_MARKDOWN_CONTENT, ToolNames.READ_SKILL_COMPANION_FILE}
