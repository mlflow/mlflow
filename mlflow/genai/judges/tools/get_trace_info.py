"""
Get trace info tool for MLflow GenAI judges.

This module provides a tool for retrieving trace metadata including
timing, location, state, and other high-level information.
"""

from mlflow.entities.trace import Trace
from mlflow.entities.trace_info import TraceInfo
from mlflow.genai.judges.tools.base import JudgeTool
from mlflow.genai.judges.tools.constants import ToolNames
from mlflow.types.llm import (
    FunctionToolDefinition,
    ToolDefinition,
    ToolParamsSchema,
)
from mlflow.utils.annotations import experimental


@experimental(version="3.4.0")
class GetTraceInfoTool(JudgeTool):
    """
    Tool for retrieving high-level metadata about a trace.

    This provides trace metadata like ID, timing, state, and location without
    the detailed span data.
    """

    @property
    def name(self) -> str:
        return ToolNames.GET_TRACE_INFO

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            function=FunctionToolDefinition(
                name=ToolNames.GET_TRACE_INFO,
                description=(
                    "Retrieve high-level metadata about the trace including ID, timing, state, "
                    "location, and request/response previews. This provides an overview of the "
                    "trace without detailed span data. Use this to understand the overall trace "
                    "context, execution duration, and whether the trace completed successfully."
                ),
                parameters=ToolParamsSchema(
                    type="object",
                    properties={},
                    required=[],
                ),
            ),
            type="function",
        )

    def invoke(self, trace: Trace) -> TraceInfo:
        """
        Get metadata about the trace.

        Args:
            trace: The MLflow trace object to analyze

        Returns:
            TraceInfo object
        """
        return trace.info
