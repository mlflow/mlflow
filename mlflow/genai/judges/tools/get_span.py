"""
Get span tool for MLflow GenAI judges.

This module provides a tool for retrieving a specific span by ID.
"""

import json

from mlflow.entities.trace import Trace
from mlflow.genai.judges.tools.base import JudgeTool
from mlflow.genai.judges.tools.constants import ToolNames
from mlflow.genai.judges.tools.types import SpanResult
from mlflow.genai.judges.tools.utils import create_page_token, parse_page_token
from mlflow.types.llm import FunctionToolDefinition, ToolDefinition, ToolParamsSchema
from mlflow.utils.annotations import experimental


@experimental(version="3.4.0")
class GetSpanTool(JudgeTool):
    """
    Tool for retrieving a specific span by its ID.

    Returns the complete span data including inputs, outputs, attributes, and events.
    """

    @property
    def name(self) -> str:
        return ToolNames.GET_SPAN

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            function=FunctionToolDefinition(
                name=ToolNames.GET_SPAN,
                description=(
                    "Retrieve a specific span by its ID. Returns the complete span data "
                    "including inputs, outputs, attributes, events, and timing information. "
                    "Use this when you need to examine the full details of a particular span. "
                    "Large content may be paginated. Consider selecting only relevant attributes "
                    "to reduce data size and improve efficiency."
                ),
                parameters=ToolParamsSchema(
                    type="object",
                    properties={
                        "span_id": {
                            "type": "string",
                            "description": "The ID of the span to retrieve",
                        },
                        "attributes_to_fetch": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "List of specific attributes to fetch from the span. If specified, "
                                "only these attributes will be returned. If not specified, all "
                                "attributes are returned. It is recommended to use list_spans "
                                "first to see available attribute names, then select relevant ones."
                            ),
                        },
                        "max_content_length": {
                            "type": "integer",
                            "description": "Maximum content size in bytes (default: 100000)",
                        },
                        "page_token": {
                            "type": "string",
                            "description": "Token to retrieve the next page of content",
                        },
                    },
                    required=["span_id"],
                ),
            ),
            type="function",
        )

    def invoke(
        self,
        trace: Trace,
        span_id: str,
        attributes_to_fetch: list[str] | None = None,
        max_content_length: int = 100000,
        page_token: str | None = None,
    ) -> SpanResult:
        """
        Get a specific span by ID from the trace.

        Args:
            trace: The MLflow trace object to analyze
            span_id: The ID of the span to retrieve
            attributes_to_fetch: List of specific attributes to fetch (None for all)
            max_content_length: Maximum content size in bytes to return
            page_token: Token to retrieve the next page (offset in bytes)

        Returns:
            SpanResult with the span content as JSON string
        """
        if not trace or not trace.data or not trace.data.spans:
            return SpanResult(
                span_id=None, content=None, content_size_bytes=0, error="Trace has no spans"
            )

        target_span = None
        for span in trace.data.spans:
            if span.span_id == span_id:
                target_span = span
                break

        if not target_span:
            return SpanResult(
                span_id=None,
                content=None,
                content_size_bytes=0,
                error=f"Span with ID '{span_id}' not found in trace",
            )

        span_dict = target_span.to_dict()
        if attributes_to_fetch is not None and span_dict.get("attributes"):
            filtered_attributes = {}
            for attr in attributes_to_fetch:
                if attr in span_dict["attributes"]:
                    filtered_attributes[attr] = span_dict["attributes"][attr]
            span_dict["attributes"] = filtered_attributes

        full_content = json.dumps(span_dict, default=str, indent=2)
        total_size = len(full_content.encode("utf-8"))
        start_offset = parse_page_token(page_token)
        end_offset = min(start_offset + max_content_length, total_size)
        content_chunk = full_content[start_offset:end_offset]
        next_page_token = create_page_token(end_offset) if end_offset < total_size else None

        return SpanResult(
            span_id=target_span.span_id,
            content=content_chunk,
            content_size_bytes=len(content_chunk.encode("utf-8")),
            page_token=next_page_token,
            error=None,
        )
