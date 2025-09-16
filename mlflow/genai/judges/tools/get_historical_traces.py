"""
Get historical traces tool for MLflow GenAI judges.

This module provides a tool for retrieving historical traces from the same session
to enable multi-turn evaluation capabilities.
"""

import logging

import mlflow
from mlflow.entities.trace import Trace
from mlflow.entities.trace_location import TraceLocationType
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.tools.base import JudgeTool
from mlflow.genai.judges.tools.constants import ToolNames
from mlflow.genai.judges.tools.types import HistoricalTrace
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.types.llm import (
    FunctionToolDefinition,
    ToolDefinition,
    ToolParamsSchema,
)
from mlflow.utils.annotations import experimental

_logger = logging.getLogger(__name__)


@experimental(version="3.4.0")
class GetHistoricalTracesTool(JudgeTool):
    """
    Tool for retrieving historical traces from the same session for multi-turn evaluation.

    This tool extracts the session ID from the current trace and searches for other traces
    within the same session to provide conversational context to judges.
    """

    @property
    def name(self) -> str:
        return ToolNames.GET_HISTORICAL_TRACES

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            function=FunctionToolDefinition(
                name=ToolNames.GET_HISTORICAL_TRACES,
                description=(
                    "Retrieve historical traces from the same session for multi-turn evaluation. "
                    "Extracts the session ID from the current trace's tags and searches for other "
                    "traces in the same session to provide conversational context. Returns a list "
                    "of HistoricalTrace objects containing trace metadata, request, and response."
                ),
                parameters=ToolParamsSchema(
                    type="object",
                    properties={
                        "max_results": {
                            "type": "integer",
                            "description": (
                                "Maximum number of historical traces to return (default: 20)"
                            ),
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

    def _validate_session_id(self, session_id: str | None) -> str:
        """
        Validate session ID from trace tags.

        Args:
            session_id: Session ID from trace tags

        Returns:
            Validated session ID

        Raises:
            MlflowException: If session ID is missing or has invalid format
        """
        if not session_id:
            raise MlflowException(
                "No session.id found in trace tags. Historical traces require a session ID "
                "to identify related traces within the same conversation session.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        # Basic validation to prevent potential injection
        if not session_id.replace("-", "").replace("_", "").isalnum():
            raise MlflowException(
                f"Invalid session ID format: {session_id}. "
                "Session IDs should contain only alphanumeric characters, hyphens, "
                "and underscores.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        return session_id

    def _validate_experiment_id(self, trace: Trace) -> str:
        """
        Validate and extract experiment ID from trace.

        Args:
            trace: The MLflow trace object

        Returns:
            Experiment ID

        Raises:
            MlflowException: If trace is not from MLflow experiment or has no experiment ID
        """
        if not trace.info.trace_location:
            raise MlflowException(
                "Current trace has no trace location. Cannot determine experiment context.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        if trace.info.trace_location.type != TraceLocationType.MLFLOW_EXPERIMENT:
            raise MlflowException(
                f"Current trace is not from an MLflow experiment "
                f"(type: {trace.info.trace_location.type}). "
                "Historical traces can only be retrieved for traces within MLflow experiments.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        if not (
            trace.info.trace_location.mlflow_experiment
            and trace.info.trace_location.mlflow_experiment.experiment_id
        ):
            raise MlflowException(
                "Current trace has no experiment_id. Cannot search for historical traces.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        return trace.info.trace_location.mlflow_experiment.experiment_id

    def invoke(
        self,
        trace: Trace,
        max_results: int = 20,
        order_by: list[str] | None = None,
    ) -> list[HistoricalTrace]:
        """
        Retrieve historical traces from the same session.

        Args:
            trace: The current MLflow trace object
            max_results: Maximum number of historical traces to return
            order_by: List of order by clauses for sorting results

        Returns:
            List of HistoricalTrace objects containing trace info, request, and response

        Raises:
            MlflowException: If session ID is not found, trace is not from MLflow experiment,
                           or search fails
        """
        # Validate session ID and experiment context
        session_id = self._validate_session_id(trace.info.tags.get("session.id"))
        experiment_id = self._validate_experiment_id(trace)
        experiment_ids = [experiment_id]

        # Default to chronological order
        if order_by is None:
            order_by = ["timestamp ASC"]

        # Build filter string for session ID and timestamp
        # Only get traces that occurred before the current trace
        filter_string = (
            f"tags.`session.id` = '{session_id}' AND trace.timestamp < {trace.info.request_time}"
        )

        _logger.debug(
            f"Searching for historical traces with session_id='{session_id}', "
            f"experiment_ids={experiment_ids}, max_results={max_results}"
        )

        try:
            # Search for traces with the same session ID
            df = mlflow.search_traces(
                experiment_ids=experiment_ids,
                filter_string=filter_string,
                max_results=max_results,
                order_by=order_by,
                extract_fields=["trace_id", "trace", "request", "response"],
            )

            historical_traces = []

            for _, row in df.iterrows():
                try:
                    # Parse trace from JSON
                    trace_obj = Trace.from_json(row["trace"])

                    # Create HistoricalTrace object
                    historical_trace = HistoricalTrace(
                        trace_info=trace_obj.info,
                        request=row["request"],
                        response=row["response"],
                    )
                    historical_traces.append(historical_trace)

                except Exception as e:
                    _logger.warning(
                        f"Failed to process trace {row.get('trace_id', 'unknown')} "
                        f"from session {session_id}: {e}"
                    )
                    continue

            _logger.debug(
                f"Retrieved {len(historical_traces)} historical traces for session {session_id}"
            )
            return historical_traces

        except Exception as e:
            raise MlflowException(
                f"Failed to search historical traces for session {session_id}: {e!s}",
                error_code="INTERNAL_ERROR",
            ) from e
