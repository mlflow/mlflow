"""
Search traces tool for MLflow GenAI judges.

This module provides a tool for searching and retrieving traces from an MLflow experiment
based on filter criteria, ordering, and result limits. It enables judges to analyze
traces within the same experiment context.
"""

import logging

import mlflow
from mlflow.entities.assessment import Assessment, Expectation, Feedback
from mlflow.entities.trace import Trace
from mlflow.entities.trace_location import TraceLocationType
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.tools.base import JudgeTool
from mlflow.genai.judges.tools.constants import ToolNames
from mlflow.genai.judges.tools.types import (
    JudgeToolExpectation,
    JudgeToolFeedback,
    JudgeToolTraceInfo,
)
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.types.llm import (
    FunctionToolDefinition,
    ToolDefinition,
    ToolParamsSchema,
)
from mlflow.utils.annotations import experimental

_logger = logging.getLogger(__name__)


def _convert_assessments_to_tool_types(
    assessments: list[Assessment],
) -> list[JudgeToolExpectation | JudgeToolFeedback]:
    tool_types: list[JudgeToolExpectation | JudgeToolFeedback] = []
    for assessment in assessments:
        if isinstance(assessment, Expectation):
            tool_types.append(
                JudgeToolExpectation(
                    name=assessment.name,
                    source=assessment.source.source_type,
                    rationale=assessment.rationale,
                    span_id=assessment.span_id,
                    assessment_id=assessment.assessment_id,
                    value=assessment.value,
                )
            )
        elif isinstance(assessment, Feedback):
            tool_types.append(
                JudgeToolFeedback(
                    name=assessment.name,
                    source=assessment.source.source_type,
                    rationale=assessment.rationale,
                    span_id=assessment.span_id,
                    assessment_id=assessment.assessment_id,
                    value=assessment.value,
                    error_code=assessment.error.error_code if assessment.error else None,
                    error_message=assessment.error.error_message if assessment.error else None,
                    stack_trace=assessment.error.stack_trace if assessment.error else None,
                    overrides=assessment.overrides,
                    valid=assessment.valid,
                )
            )
    return tool_types


def _get_experiment_id(trace: Trace) -> str:
    """
    Get and validate experiment ID from trace.

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
            "Traces can only be retrieved for traces within MLflow experiments.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    if not (
        trace.info.trace_location.mlflow_experiment
        and trace.info.trace_location.mlflow_experiment.experiment_id
    ):
        raise MlflowException(
            "Current trace has no experiment_id. Cannot search for traces.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    return trace.info.trace_location.mlflow_experiment.experiment_id


@experimental(version="3.5.0")
class SearchTracesTool(JudgeTool):
    """
    Tool for searching and retrieving traces from an MLflow experiment.

    This tool enables judges to search for traces within the same experiment context
    as the current trace being evaluated. It supports filtering, ordering, and
    pagination of results. The tool returns trace metadata including request/response
    data, execution metrics, and assessments for analysis.
    """

    @property
    def name(self) -> str:
        return ToolNames._SEARCH_TRACES

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            function=FunctionToolDefinition(
                name=ToolNames._SEARCH_TRACES,
                description=(
                    "Search for traces within the same MLflow experiment as the current trace. "
                    "Returns trace metadata including trace_id, request_time, state, request, "
                    "response, execution_duration, and assessments. Supports filtering with "
                    "MLflow search syntax (e.g., 'attributes.status = \"OK\"'), custom ordering "
                    "(e.g., ['timestamp DESC']), and result limits. Use this to analyze patterns "
                    "across traces or find specific traces matching criteria."
                ),
                parameters=ToolParamsSchema(
                    type="object",
                    properties={
                        "filter_string": {
                            "type": "string",
                            "description": (
                                "Optional filter string using SQL-like search syntax. "
                                "If not specified, all traces are returned.\n\n"
                                "SUPPORTED FIELDS:\n"
                                "- Attributes: request_id, timestamp_ms, execution_time_ms, "
                                "status, name, run_id\n"
                                "- Tags: Use 'tags.' or 'tag.' prefix "
                                "(e.g., tags.operation_type, tag.model_name)\n"
                                "- Metadata: Use 'metadata.' prefix (e.g., metadata.run_id)\n"
                                "- Use backticks for special characters: tags.`model-name`\n\n"
                                "VALUE SYNTAX:\n"
                                "- String values MUST be quoted: status = 'OK'\n"
                                "- Numeric values don't need quotes: execution_time_ms > 1000\n"
                                "- Tag and metadata values MUST be quoted as strings\n\n"
                                "COMPARATORS:\n"
                                "- Numeric (timestamp_ms, execution_time_ms): "
                                ">, >=, =, !=, <, <=\n"
                                "- String (name, status, request_id): =, !=, IN, NOT IN\n"
                                "- Tags/Metadata: =, !=\n\n"
                                "STATUS VALUES: 'OK', 'ERROR', 'IN_PROGRESS'\n\n"
                                "EXAMPLES:\n"
                                "- status = 'OK'\n"
                                "- execution_time_ms > 1000\n"
                                "- tags.model_name = 'gpt-4'\n"
                                "- tags.`model-version` = 'v2' AND status = 'OK'\n"
                                "- timestamp_ms >= 1234567890000 AND execution_time_ms < 5000\n"
                                "- status IN ('OK', 'ERROR')\n"
                                "- tags.environment = 'production' AND status = 'ERROR' "
                                "AND execution_time_ms > 500\n"
                                "- status = 'OK' AND tag.importance = 'high'"
                            ),
                            "default": None,
                        },
                        "order_by": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Optional list of order by expressions (e.g., ['timestamp DESC']). "
                                "Defaults to ['timestamp ASC'] for chronological order."
                            ),
                            "default": ["timestamp ASC"],
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of traces to return (default: 20)",
                            "default": 20,
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
        filter_string: str | None = None,
        order_by: list[str] | None = None,
        max_results: int = 20,
    ) -> list[JudgeToolTraceInfo]:
        """
        Search for traces within the same experiment as the current trace.

        Args:
            trace: The current MLflow trace object (used to determine experiment context)
            filter_string: Optional filter using MLflow search syntax
                (e.g., 'attributes.status = "OK"')
            order_by: Optional list of order by expressions (e.g., ['timestamp DESC'])
            max_results: Maximum number of traces to return (default: 20)

        Returns:
            List of JudgeToolTraceInfo objects containing trace metadata, request/response data,
            and assessments for each matching trace

        Raises:
            MlflowException: If trace has no experiment context or search fails
        """
        # Extract and validate experiment ID from trace
        experiment_id = _get_experiment_id(trace)
        locations = [experiment_id]

        # Default to chronological order
        if order_by is None:
            order_by = ["timestamp ASC"]

        _logger.debug(
            "Searching for traces with properties:\n\n"
            + "\n".join(
                [
                    f"* experiment_id={experiment_id}",
                    f"* filter_string={filter_string}",
                    f"* order_by={order_by}",
                    f"* max_results={max_results}",
                ]
            )
        )

        try:
            trace_objs = mlflow.search_traces(
                locations=locations,
                filter_string=filter_string,
                order_by=order_by,
                max_results=max_results,
                return_type="list",
            )

        except Exception as e:
            raise MlflowException(
                f"Failed to search traces: {e!s}",
                error_code="INTERNAL_ERROR",
            ) from e

        traces = []

        for trace_obj in trace_objs:
            try:
                trace_info = JudgeToolTraceInfo(
                    trace_id=trace_obj.info.trace_id,
                    request_time=trace_obj.info.request_time,
                    state=trace_obj.info.state,
                    request=trace_obj.data.request,
                    response=trace_obj.data.response,
                    execution_duration=trace_obj.info.execution_duration,
                    assessments=_convert_assessments_to_tool_types(trace_obj.info.assessments),
                )
                traces.append(trace_info)
            except Exception as e:
                _logger.warning(f"Failed to process trace {trace_obj.info.trace_id}: {e}")
                continue

        _logger.debug(f"Retrieved {len(traces)} traces")
        return traces
