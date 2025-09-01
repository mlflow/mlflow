"""
Tool for searching traces using regex patterns.

This module provides functionality to search through all spans in a trace
using regular expressions with case-insensitive matching.
"""

import json
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
    """Represents a single regex match found in a span."""

    span_id: str
    matched_text: str
    surrounding_text: str  # Text with ~100 chars before and after, with ellipses


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
    Tool for searching through all spans in a trace using regex patterns.

    Performs case-insensitive regex search across all span fields including
    inputs, outputs, and attributes. Returns matched text with surrounding
    context to help understand where matches occur.
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
                    "Search through all spans in the trace using a regular expression pattern. "
                    "Performs case-insensitive matching and returns all matches with surrounding "
                    "context. Useful for finding specific patterns, values, or text across the "
                    "entire trace."
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
                    },
                    required=["pattern"],
                ),
            ),
            type="function",
        )

    def invoke(self, trace: Trace, pattern: str, max_matches: int = 50) -> SearchTraceRegexResult:
        """
        Search through the trace using a regex pattern.

        Args:
            trace: The MLflow trace object to search through
            pattern: Regular expression pattern to search for
            max_matches: Maximum number of matches to return

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

        if not trace.data or not trace.data.spans:
            return SearchTraceRegexResult(
                pattern=pattern,
                total_matches=0,
                matches=[],
                error="Trace has no spans to search",
            )

        matches = []
        total_found = 0

        for span in trace.data.spans:
            # Search through all span fields that might contain text
            searchable_texts = []

            # Add inputs
            if span.inputs:
                for key, value in span.inputs.items():
                    if isinstance(value, str):
                        searchable_texts.append(f"input.{key}: {value}")
                    else:
                        searchable_texts.append(f"input.{key}: {json.dumps(value)}")

            # Add outputs
            if span.outputs:
                for key, value in span.outputs.items():
                    if isinstance(value, str):
                        searchable_texts.append(f"output.{key}: {value}")
                    else:
                        searchable_texts.append(f"output.{key}: {json.dumps(value)}")

            # Add attributes
            if span.attributes:
                for key, value in span.attributes.items():
                    if isinstance(value, str):
                        searchable_texts.append(f"attribute.{key}: {value}")
                    else:
                        searchable_texts.append(f"attribute.{key}: {json.dumps(value)}")

            # Add span name and other string fields
            if span.name:
                searchable_texts.append(f"name: {span.name}")

            # Search through all text fields
            for text in searchable_texts:
                for match in regex.finditer(text):
                    if total_found >= max_matches:
                        break

                    matched_text = match.group()
                    start, end = match.span()

                    # Get surrounding context (100 chars before and after)
                    context_start = max(0, start - 100)
                    context_end = min(len(text), end + 100)

                    surrounding = text[context_start:context_end]

                    # Add ellipses if we truncated
                    if context_start > 0:
                        surrounding = "..." + surrounding
                    if context_end < len(text):
                        surrounding = surrounding + "..."

                    matches.append(
                        RegexMatch(
                            span_id=span.span_id,
                            matched_text=matched_text,
                            surrounding_text=surrounding,
                        )
                    )

                    total_found += 1

                if total_found >= max_matches:
                    break

            if total_found >= max_matches:
                break

        return SearchTraceRegexResult(
            pattern=pattern,
            total_matches=total_found,
            matches=matches,
        )
