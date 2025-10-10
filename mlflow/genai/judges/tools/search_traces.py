""" """  # TODO

import logging

import mlflow
from mlflow.entities.trace import Trace
from mlflow.entities.trace_location import TraceLocationType
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.tools.base import JudgeTool
from mlflow.genai.judges.tools.constants import ToolNames
from mlflow.genai.judges.tools.types import TraceInfo
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.types.llm import (
    FunctionToolDefinition,
    ToolDefinition,
    ToolParamsSchema,
)
from mlflow.utils.annotations import experimental

_logger = logging.getLogger(__name__)


@experimental(version="3.5.0")
class SearchTracesTool(JudgeTool):
    """ """  # TODO

    @property
    def name(self) -> str:
        return ToolNames.SEARCH_TRACES

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            function=FunctionToolDefinition(
                name=ToolNames.SEARCH_TRACES,
                description=("PLACEHOLDER"),  # TODO
                parameters=ToolParamsSchema(
                    type="object",
                    properties={},  # TODO
                    required=[],
                ),
            ),
            type="function",
        )

    def _get_experiment_id(self, trace: Trace) -> str:
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

    def invoke(
        self,
        trace: Trace,
        filter_string: str | None = None,
        order_by: list[str] | None = None,
        max_results: int = 20,
    ) -> list[TraceInfo]:
        """ """  # TODO
        # Extract and validate experiment ID from trace
        experiment_id = self._validate_experiment_id(trace)
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
            df = mlflow.search_traces(
                locations=locations,
                filter_string=filter_string,
                order_by=order_by,
                max_results=max_results,
                extract_fields=["trace_id", "trace", "request", "response"],
            )

        except Exception as e:
            raise MlflowException(
                f"Failed to search traces: {e!s}",
                error_code="INTERNAL_ERROR",
            ) from e

        traces = []

        for _, row in df.iterrows():
            try:
                # Parse trace from JSON
                trace_obj = Trace.from_json(row["trace"])

                # Create HistoricalTrace object
                trace_info = TraceInfo(
                    trace_id=trace_obj.info.trace_id,
                    request_time=trace_obj.info.request_time,
                    state=trace_obj.info.state,
                    request=row["request"],
                    response=row["response"],
                    execution_duration=trace_obj.info.execution_duration,
                    assessments=trace_obj.info.assessments,
                )
                traces.append(trace_info)

            except Exception as e:
                _logger.warning(
                    f"Failed to process trace {row.get('trace_id', 'unknown')}: {e}"
                )
                continue

        _logger.debug(f"Retrieved {len(traces)} traces")
        return traces
