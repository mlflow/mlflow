"""
Get trace info tool for MLflow GenAI judges.

This module provides a tool for retrieving trace metadata including
timing, location, state, and other high-level information.
"""

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.tools.base import JudgeTool
from mlflow.genai.judges.tools.constants import ToolNames
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
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
                    properties={
                        "trace_id": {
                            "type": "string",
                            "description": ("The ID of the MLflow trace to analyze"),
                        },
                    },
                    required=["trace_id"],
                ),
            ),
            type="function",
        )

    def invoke(self, trace_id: str) -> dict[str, object]:
        """
        Get metadata about the trace.

        Args:
            trace_id: The ID of the MLflow trace to analyze

        Returns:
            Dictionary of trace metadata (without assessments).
        """
        trace = mlflow.get_trace(trace_id)
        if trace is None:
            raise MlflowException(
                f"Trace with ID '{trace_id}' not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        info = trace.info.to_dict()
        # Assessments can be enormous and are not useful for trace analysis
        info.pop("assessments", None)
        return info
