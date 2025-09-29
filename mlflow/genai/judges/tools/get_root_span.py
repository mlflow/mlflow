"""
Get root span tool for MLflow GenAI judges.

This module provides a tool for retrieving the root span of a trace,
which contains the top-level inputs and outputs.
"""

from mlflow.entities.trace import Trace
from mlflow.genai.judges.tools.base import JudgeTool
from mlflow.genai.judges.tools.constants import ToolNames
from mlflow.genai.judges.tools.get_span import GetSpanTool
from mlflow.genai.judges.tools.types import SpanResult
from mlflow.types.llm import FunctionToolDefinition, ToolDefinition, ToolParamsSchema
from mlflow.utils.annotations import experimental


@experimental(version="3.4.0")
class GetRootSpanTool(JudgeTool):
    """
    Tool for retrieving the root span from a trace.

    The root span contains the top-level inputs to the agent and final outputs.
    """

    @property
    def name(self) -> str:
        return ToolNames.GET_ROOT_SPAN

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            function=FunctionToolDefinition(
                name=ToolNames.GET_ROOT_SPAN,
                description=(
                    "Retrieve the root span of the trace, which contains the top-level inputs "
                    "to the agent and final outputs. Note that in some traces, the root span "
                    "may not contain outputs, but it typically should. If the root span doesn't "
                    "have outputs, you may need to look at other spans to find the final results. "
                    "The content is returned as a JSON string. Large content may be paginated. "
                    "Consider selecting only relevant attributes to reduce data size and improve "
                    "efficiency."
                ),
                parameters=ToolParamsSchema(
                    type="object",
                    properties={
                        "attributes_to_fetch": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "List of specific attributes to fetch from the span. If specified, "
                                "only these attributes will be returned. If not specified, all "
                                "attributes are returned. Use list_spans first to see available "
                                "attribute names, then select only the relevant ones."
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
                    required=[],
                ),
            ),
            type="function",
        )

    def invoke(
        self,
        trace: Trace,
        attributes_to_fetch: list[str] | None = None,
        max_content_length: int = 100000,
        page_token: str | None = None,
    ) -> SpanResult:
        """
        Get the root span from the trace.

        Args:
            trace: The MLflow trace object to analyze
            attributes_to_fetch: List of specific attributes to fetch (None for all)
            max_content_length: Maximum content size in bytes to return
            page_token: Token to retrieve the next page (offset in bytes)

        Returns:
            SpanResult with the root span content as JSON string
        """
        if not trace or not trace.data or not trace.data.spans:
            return SpanResult(
                span_id=None, content=None, content_size_bytes=0, error="Trace has no spans"
            )

        root_span_id = None
        for span in trace.data.spans:
            if span.parent_id is None:
                root_span_id = span.span_id
                break

        if not root_span_id:
            return SpanResult(
                span_id=None,
                content=None,
                content_size_bytes=0,
                error="No root span found in trace",
            )

        return GetSpanTool().invoke(
            trace, root_span_id, attributes_to_fetch, max_content_length, page_token
        )
