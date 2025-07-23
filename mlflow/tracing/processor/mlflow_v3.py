from typing import Optional

from opentelemetry.sdk.trace import Span as OTelSpan
from opentelemetry.sdk.trace.export import SpanExporter

import mlflow
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.tracing.processor.base_mlflow import BaseMlflowSpanProcessor
from mlflow.tracing.utils import generate_trace_id_v3


class MlflowV3SpanProcessor(BaseMlflowSpanProcessor):
    """
    Defines custom hooks to be executed when a span is started or ended (before exporting).

    This processor is used for exporting traces to MLflow Tracking Server
    using the V3 trace schema and API.
    """

    def __init__(self):
        super().__init__()

    def _start_trace(self, root_span: OTelSpan) -> TraceInfo:
        """
        Create a new TraceInfo object and register it with the trace manager.

        This method is called in the on_start method of the base class.
        """
        from mlflow.tracing.provider import _MLFLOW_TRACE_USER_DESTINATION
        from mlflow.tracing.destination import MlflowExperiment

        trace_info = TraceInfo(
            trace_id=generate_trace_id_v3(root_span),
            trace_location=TraceLocation.from_experiment_id(
                self._get_experiment_id_for_trace(root_span)
            ),
            request_time=root_span.start_time // 1_000_000,  # nanosecond to millisecond
            execution_duration=None,
            state=TraceState.IN_PROGRESS,
            trace_metadata=self._get_basic_trace_metadata(),
            tags=self._get_basic_trace_tags(root_span),
        )

        tracking_uri = None
        if isinstance(_MLFLOW_TRACE_USER_DESTINATION.get(), MlflowExperiment):
            tracking_uri = _MLFLOW_TRACE_USER_DESTINATION.get().tracking_uri or mlflow.get_tracking_uri()

        self._trace_manager.register_trace(root_span.context.trace_id, trace_info, tracking_uri)

        return trace_info
