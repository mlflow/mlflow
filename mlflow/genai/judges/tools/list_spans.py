"""
Tool definitions for MLflow GenAI judges.

This module provides concrete JudgeTool implementations that judges can use
to analyze traces and extract information during evaluation.
"""

from dataclasses import dataclass

from mlflow.entities.trace import Trace
from mlflow.genai.judges.tools.base import JudgeTool
from mlflow.genai.judges.tools.types import SpanInfo
from mlflow.genai.judges.tools.utils import create_page_token, parse_page_token
from mlflow.types.llm import (
    FunctionToolDefinition,
    ParamProperty,
    ToolDefinition,
    ToolParamsSchema,
)
from mlflow.utils.annotations import experimental


@experimental(version="3.4.0")
@dataclass
class ListSpansResult:
    """Result from listing spans with optional pagination."""

    spans: list[SpanInfo]
    next_page_token: str | None = None


def _create_span_info(span) -> SpanInfo:
    """Create SpanInfo from a span object."""
    start_time_ms = span.start_time_ns / 1_000_000
    end_time_ms = span.end_time_ns / 1_000_000
    duration_ms = end_time_ms - start_time_ms

    # Get attribute names
    attribute_names = list(span.attributes.keys()) if span.attributes else []

    return SpanInfo(
        span_id=span.span_id,
        name=span.name,
        span_type=span.span_type,
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
        duration_ms=duration_ms,
        parent_id=span.parent_id,
        status=span.status,
        is_root=(span.parent_id is None),
        attribute_names=attribute_names,
    )


@experimental(version="3.4.0")
class ListSpansTool(JudgeTool):
    """
    Tool for listing and analyzing spans within a trace.

    This tool provides functionality to extract and analyze span information
    from MLflow traces, including span names, types, durations, and metadata.
    """

    @property
    def name(self) -> str:
        return "list_spans"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            function=FunctionToolDefinition(
                name="list_spans",
                description=(
                    "List information about spans within a trace with pagination support. "
                    "Returns span metadata including span_id, name, span_type, timing data "
                    "(start_time_ms, end_time_ms, duration_ms), parent_id, status, and "
                    "attribute_names (list of attribute keys). This provides an overview of "
                    "all spans but does not fetch full span content."
                ),
                parameters=ToolParamsSchema(
                    type="object",
                    properties={
                        "max_results": ParamProperty(
                            type="integer",
                            description="Maximum number of spans to return (default: 100)",
                        ),
                        "page_token": ParamProperty(
                            type="string",
                            description="Token for retrieving the next page of results",
                        ),
                    },
                    required=[],
                ),
            ),
            type="function",
        )

    def invoke(
        self, trace: Trace, max_results: int = 100, page_token: str | None = None
    ) -> ListSpansResult:
        """
        List spans from a trace with pagination support.

        Args:
            trace: The MLflow trace object to analyze
            max_results: Maximum number of spans to return (default: 100)
            page_token: Token for retrieving the next page of results

        Returns:
            ListSpansResult containing spans list and optional next page token
        """
        if not trace or not trace.data or not trace.data.spans:
            return ListSpansResult(spans=[])

        start_index = parse_page_token(page_token)

        # Get the slice of spans for this page
        all_spans = trace.data.spans
        end_index = start_index + max_results
        page_spans = all_spans[start_index:end_index]

        # Build span info for this page
        spans_info = [_create_span_info(span) for span in page_spans]

        # Determine next page token - only include if there are more pages
        next_page_token = None
        if end_index < len(all_spans):
            next_page_token = create_page_token(end_index)

        return ListSpansResult(spans=spans_info, next_page_token=next_page_token)
