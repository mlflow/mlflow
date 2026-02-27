import logging

from opentelemetry.sdk.trace import Span as OTelSpan
from opentelemetry.sdk.trace.export import SpanExporter

from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.tracing.processor.base_mlflow import BaseMlflowSpanProcessor
from mlflow.tracing.utils import generate_trace_id_v3, get_experiment_id_for_trace

_logger = logging.getLogger(__name__)


class MlflowV3SpanProcessor(BaseMlflowSpanProcessor):
    """
    Defines custom hooks to be executed when a span is started or ended (before exporting).

    This processor is used for exporting traces to MLflow Tracking Server
    using the V3 trace schema and API.
    """

    def __init__(
        self,
        span_exporter: SpanExporter,
        export_metrics: bool,
    ):
        super().__init__(span_exporter, export_metrics)

    def _start_trace(self, root_span: OTelSpan) -> TraceInfo:
        """
        Create a new TraceInfo object and register it with the trace manager.

        This method is called in the on_start method of the base class.
        """
        experiment_id = get_experiment_id_for_trace(root_span)
        if experiment_id is None:
            _logger.debug(
                "Experiment ID is not set for trace. It may not be exported to MLflow backend."
            )

        trace_info = TraceInfo(
            trace_id=generate_trace_id_v3(root_span),
            trace_location=TraceLocation.from_experiment_id(experiment_id),
            request_time=root_span.start_time // 1_000_000,  # nanosecond to millisecond
            execution_duration=None,
            state=TraceState.IN_PROGRESS,
            trace_metadata=self._get_basic_trace_metadata(),
            tags=self._get_basic_trace_tags(root_span),
        )
        self._trace_manager.register_trace(root_span.context.trace_id, trace_info)
        return trace_info
