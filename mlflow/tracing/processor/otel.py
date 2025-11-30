from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter

from mlflow.entities.span import create_mlflow_span
from mlflow.entities.trace_info import TraceInfo, TraceLocation, TraceState
from mlflow.environment_variables import MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT
from mlflow.tracing.constant import TRACE_SCHEMA_VERSION, TRACE_SCHEMA_VERSION_KEY
from mlflow.tracing.processor.otel_metrics_mixin import OtelMetricsMixin
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import generate_trace_id_v3


class OtelSpanProcessor(OtelMetricsMixin, BatchSpanProcessor):
    """
    SpanProcessor implementation to export MLflow traces to a OpenTelemetry collector.

    Extending OpenTelemetry BatchSpanProcessor to add some custom hooks to be executed when a span
    is started or ended (before exporting).
    """

    def __init__(self, span_exporter: SpanExporter, export_metrics: bool) -> None:
        super().__init__(span_exporter)
        self._export_metrics = export_metrics
        # In opentelemetry-sdk 1.34.0, the `span_exporter` field was removed from the
        # `BatchSpanProcessor` class.
        # https://github.com/open-telemetry/opentelemetry-python/issues/4616
        #
        # The `span_exporter` field was restored as a property in 1.34.1
        # https://github.com/open-telemetry/opentelemetry-python/pull/4621
        #
        # We use a try-except block to maintain compatibility with both versions.
        try:
            self.span_exporter = span_exporter
        except AttributeError:
            pass

        # Only register traces with trace manager when NOT in dual export mode
        # In dual export mode, MLflow span processors handle trace registration
        self._should_register_traces = not MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT.get()
        if self._should_register_traces:
            self._trace_manager = InMemoryTraceManager.get_instance()

    def on_start(self, span: OTelReadableSpan, parent_context=None):
        if self._should_register_traces:
            if not span.parent:
                trace_info = self._create_trace_info(span)
                trace_id = trace_info.trace_id
                self._trace_manager.register_trace(span.context.trace_id, trace_info)
            else:
                trace_id = self._trace_manager.get_mlflow_trace_id_from_otel_id(
                    span.context.trace_id
                )
            self._trace_manager.register_span(create_mlflow_span(span, trace_id))

        super().on_start(span, parent_context)

    def on_end(self, span: OTelReadableSpan):
        if self._export_metrics:
            self.record_metrics_for_span(span)

        if self._should_register_traces and not span.parent:
            self._trace_manager.pop_trace(span.context.trace_id)

        super().on_end(span)

    def _create_trace_info(self, span: OTelReadableSpan) -> TraceInfo:
        """Create a TraceInfo object from an OpenTelemetry span."""
        return TraceInfo(
            trace_id=generate_trace_id_v3(span),
            trace_location=TraceLocation.from_experiment_id(None),
            request_time=span.start_time // 1_000_000,  # nanosecond to millisecond
            execution_duration=None,
            state=TraceState.IN_PROGRESS,
            trace_metadata={TRACE_SCHEMA_VERSION_KEY: str(TRACE_SCHEMA_VERSION)},
            tags={},
        )
