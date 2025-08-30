"""
List spans tool for MLflow GenAI judges.

This module provides a tool for retrieving all spans from a trace,
allowing judges to analyze the detailed execution flow and span hierarchy.
"""

from typing import Any

from mlflow.entities.trace import Trace
from mlflow.genai.judges.tools.base import JudgeTool
from mlflow.genai.judges.tools.constants import ToolNames
from mlflow.types.llm import (
    FunctionToolDefinition,
    ToolDefinition,
    ToolParamsSchema,
)
from mlflow.utils.annotations import experimental


@experimental(version="3.4.0")
class ListSpansTool(JudgeTool):
    """
    Tool for retrieving all spans from a trace.

    This provides access to detailed span data including timing, inputs, outputs,
    and hierarchical relationships within the trace execution.
    """

    @property
    def name(self) -> str:
        return ToolNames.LIST_SPANS

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            function=FunctionToolDefinition(
                name=ToolNames.LIST_SPANS,
                description=(
                    "Retrieve all spans from the trace. Each span represents a unit of work "
                    "within the trace execution and contains detailed information like timing, "
                    "inputs, outputs, status, and hierarchical relationships. Use this to "
                    "analyze the detailed execution flow, identify performance bottlenecks, "
                    "or examine the internal structure of the traced operation."
                ),
                parameters=ToolParamsSchema(
                    type="object",
                    properties={},
                    required=[],
                ),
            ),
            type="function",
        )

    def invoke(self, trace: Trace) -> list[dict[str, Any]]:
        """
        Get all spans from the trace.

        Args:
            trace: The MLflow trace object to analyze

        Returns:
            List of span dictionaries containing detailed span information
        """
        return [span.to_dict() for span in trace.data.spans]
