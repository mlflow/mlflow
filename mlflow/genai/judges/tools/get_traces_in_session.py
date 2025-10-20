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
from mlflow.genai.judges.tools.types import TraceInfo
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
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
        return ToolNames.GET_TRACES_IN_SESSION

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            function=FunctionToolDefinition(
                name=ToolNames.GET_TRACES_IN_SESSION,
                description=(
                    "Retrieve traces from the same session for multi-turn evaluation. "
                    "Extracts the session ID from the current trace's tags and searches for other "
                    "traces in the same session to provide conversational context. Returns a list "
                    "of TraceInfo objects containing trace metadata, request, and response."
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
    ) -> list[TraceInfo]:
        """
        Retrieve traces from the same session.

        Args:
            trace: The current MLflow trace object
            max_results: Maximum number of traces to return
            order_by: List of order by clauses for sorting results

        Returns:
            List of TraceInfo objects containing trace metadata, request, and response

        Raises:
            MlflowException: If session ID is not found or has invalid format
        """
        session_id = trace.info.tags.get("session.id")

        if not session_id:
            raise MlflowException(
                "No session.id found in trace tags. Traces in session require a session ID "
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

        if order_by is None:
            order_by = ["timestamp ASC"]

        filter_string = (
            f"tags.`session.id` = '{session_id}' AND trace.timestamp < {trace.info.request_time}"
        )

        return SearchTracesTool().invoke(trace, filter_string, order_by, max_results)
