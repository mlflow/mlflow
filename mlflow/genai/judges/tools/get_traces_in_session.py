"""
Get traces in session tool for MLflow GenAI judges.

This module provides a tool for retrieving traces from the same session
to enable multi-turn evaluation capabilities.
"""

from mlflow.entities.trace import Trace
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.tools.base import JudgeTool
from mlflow.genai.judges.tools.constants import ToolNames
from mlflow.genai.judges.tools.search_traces import SearchTracesTool
from mlflow.genai.judges.tools.types import JudgeToolTraceInfo
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracing.constant import TraceMetadataKey
from mlflow.types.llm import FunctionToolDefinition, ToolDefinition, ToolParamsSchema
from mlflow.utils.annotations import experimental


@experimental(version="3.5.0")
class GetTracesInSession(JudgeTool):
    """
    Tool for retrieving traces from the same session for multi-turn evaluation.

    This tool extracts the session ID from the current trace and searches for other traces
    within the same session to provide conversational context to judges.
    """

    @property
    def name(self) -> str:
        return ToolNames._GET_TRACES_IN_SESSION

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            function=FunctionToolDefinition(
                name=ToolNames._GET_TRACES_IN_SESSION,
                description=(
                    "Retrieve traces from the same session for multi-turn evaluation. "
                    "Extracts the session ID from the current trace and searches for other "
                    "traces in the same session to provide conversational context. "
                    "Returns a list of JudgeToolTraceInfo objects containing trace metadata, "
                    "request, and response."
                ),
                parameters=ToolParamsSchema(
                    type="object",
                    properties={
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of traces to return (default: 20)",
                            "default": 20,
                        },
                        "order_by": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "List of order by clauses for sorting results "
                                "(default: ['timestamp ASC'] for chronological order)"
                            ),
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
        max_results: int = 20,
        order_by: list[str] | None = None,
    ) -> list[JudgeToolTraceInfo]:
        """
        Retrieve traces from the same session.

        Args:
            trace: The current MLflow trace object
            max_results: Maximum number of traces to return
            order_by: List of order by clauses for sorting results

        Returns:
            List of JudgeToolTraceInfo objects containing trace metadata, request, and response

        Raises:
            MlflowException: If session ID is not found or has invalid format
        """
        session_id = trace.info.trace_metadata.get(TraceMetadataKey.TRACE_SESSION)

        if not session_id:
            raise MlflowException(
                "No session ID found in trace metadata. Traces in session require a session ID "
                "to identify related traces within the same conversation session.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        if not session_id.replace("-", "").replace("_", "").isalnum():
            raise MlflowException(
                (
                    f"Invalid session ID format: {session_id}. Session IDs should contain only "
                    "alphanumeric characters, hyphens, and underscores."
                ),
                error_code=INVALID_PARAMETER_VALUE,
            )

        filter_string = (
            f"metadata.`{TraceMetadataKey.TRACE_SESSION}` = '{session_id}' "
            f"AND trace.timestamp < {trace.info.request_time}"
        )

        return SearchTracesTool().invoke(
            trace=trace, filter_string=filter_string, order_by=order_by, max_results=max_results
        )
