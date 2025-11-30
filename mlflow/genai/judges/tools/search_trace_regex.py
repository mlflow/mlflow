"""
Tool for searching traces using regex patterns.

This module provides functionality to search through entire traces (including
spans, metadata, tags, requests, and responses) using regular expressions
with case-insensitive matching.
"""

import re
from dataclasses import dataclass

from mlflow.entities.trace import Trace
from mlflow.genai.judges.tools.base import JudgeTool
from mlflow.genai.judges.tools.constants import ToolNames
from mlflow.types.llm import FunctionToolDefinition, ToolDefinition, ToolParamsSchema
from mlflow.utils.annotations import experimental


@experimental(version="3.4.0")
@dataclass
class RegexMatch:
    """Represents a single regex match found in a trace."""

    span_id: str
    matched_text: str
    surrounding_text: str


@experimental(version="3.4.0")
@dataclass
class SearchTraceRegexResult:
    """Result of searching a trace with a regex pattern."""

    pattern: str
    total_matches: int
    matches: list[RegexMatch]
    error: str | None = None


@experimental(version="3.4.0")
class SearchTraceRegexTool(JudgeTool):
    """
    Tool for searching through entire traces using regex patterns.

    Performs case-insensitive regex search across all trace fields including
    spans, metadata, tags, requests, responses, and other fields. Returns
    matched text with surrounding context to help understand where matches occur.
    """

    @property
    def name(self) -> str:
        """Return the tool name."""
        return ToolNames.SEARCH_TRACE_REGEX

    def get_definition(self) -> ToolDefinition:
        """Get the tool definition for LiteLLM/OpenAI function calling."""
        return ToolDefinition(
            function=FunctionToolDefinition(
                name=ToolNames.SEARCH_TRACE_REGEX,
                description=(
                    "Search through the entire trace using a regular expression pattern. "
                    "Performs case-insensitive matching across all trace fields including spans, "
                    "metadata, tags, requests, and responses. Returns all matches with surrounding "
                    "context. Useful for finding specific patterns, values, or text anywhere in "
                    "the trace."
                ),
                parameters=ToolParamsSchema(
                    type="object",
                    properties={
                        "pattern": {
                            "type": "string",
                            "description": (
                                "Regular expression pattern to search for. The search is "
                                "case-insensitive. Examples: 'error.*timeout', 'user_id:\\s*\\d+', "
                                "'function_name\\(.*\\)'"
                            ),
                        },
                        "max_matches": {
                            "type": "integer",
                            "description": "Maximum number of matches to return (default: 50)",
                            "default": 50,
                        },
                        "surrounding_content_length": {
                            "type": "integer",
                            "description": (
                                "Number of characters to include before and after each match "
                                "for context (default: 100)"
                            ),
                            "default": 100,
                        },
                    },
                    required=["pattern"],
                ),
            ),
            type="function",
        )

    def invoke(
        self,
        trace: Trace,
        pattern: str,
        max_matches: int = 50,
        surrounding_content_length: int = 100,
    ) -> SearchTraceRegexResult:
        """
        Search through the trace using a regex pattern.

        Args:
            trace: The MLflow trace object to search through
            pattern: Regular expression pattern to search for
            max_matches: Maximum number of matches to return
            surrounding_content_length: Number of characters to include before and after each
                match for context

        Returns:
            SearchTraceRegexResult containing the search results
        """
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return SearchTraceRegexResult(
                pattern=pattern,
                total_matches=0,
                matches=[],
                error=f"Invalid regex pattern: {e}",
            )

        trace_json = trace.to_json()
        matches = []
        total_found = 0
        for match in regex.finditer(trace_json):
            if total_found >= max_matches:
                break
            matches.append(
                self._create_regex_match(
                    match, trace_json, surrounding_content_length=surrounding_content_length
                )
            )
            total_found += 1

        return SearchTraceRegexResult(
            pattern=pattern,
            total_matches=total_found,
            matches=matches,
        )

    def _create_regex_match(
        self,
        match: re.Match[str],
        text: str,
        span_id: str = "trace",
        surrounding_content_length: int = 100,
    ) -> RegexMatch:
        """Create a RegexMatch with surrounding context from a regex match object."""
        matched_text = match.group()
        start, end = match.span()
        context_start = max(0, start - surrounding_content_length)
        context_end = min(len(text), end + surrounding_content_length)
        surrounding = text[context_start:context_end]
        if context_start > 0:
            surrounding = "..." + surrounding
        if context_end < len(text):
            surrounding = surrounding + "..."
        return RegexMatch(
            span_id=span_id,
            matched_text=matched_text,
            surrounding_text=surrounding,
        )
