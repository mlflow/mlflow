import json

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan
from opentelemetry.sdk.trace import Span as OTelSpan
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter

from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.tracing.constant import TRACE_SCHEMA_VERSION, TRACE_SCHEMA_VERSION_KEY, SpanAttributeKey
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import generate_trace_id_v3


class OtelSpanProcessor(BatchSpanProcessor):
    """
    SpanProcessor implementation to export MLflow traces to a OpenTelemetry collector.

    Extending OpenTelemetry BatchSpanProcessor to add some custom hooks to be executed when a span
    is started or ended (before exporting).
    """

    def __init__(self, span_exporter: SpanExporter):
        super().__init__(span_exporter)
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
        self._trace_manager = InMemoryTraceManager.get_instance()

    def on_start(self, span: OTelSpan, parent_context: Context | None = None):
        """
        Handle the start of a span. This method is called when an OpenTelemetry span is started.

        Args:
            span: An OpenTelemetry Span object that is started.
            parent_context: The context of the span. Note that this is only passed when the context
                object is explicitly specified to OpenTelemetry start_span call. If the parent
                span is obtained from the global context, it won't be passed here so we should not
                rely on it.
        """
        trace_id = generate_trace_id_v3(span)
        trace_info = TraceInfo(
            trace_id=trace_id,
            trace_location=TraceLocation.from_experiment_id(None),
            request_time=span.start_time // 1_000_000,  # nanosecond to millisecond
            execution_duration=None,
            state=TraceState.IN_PROGRESS,
            trace_metadata={TRACE_SCHEMA_VERSION_KEY: str(TRACE_SCHEMA_VERSION)},
            tags={},
        )
        span.set_attribute(SpanAttributeKey.REQUEST_ID, json.dumps(trace_id))

        self._trace_manager.register_trace(span.context.trace_id, trace_info)

        super().on_start(span, parent_context)

    def on_end(self, span: OTelReadableSpan):
        # Pops the trace entry from the in-memory trace manager to avoid memory leak
        if span._parent is None:
            self._trace_manager.pop_trace(span.context.trace_id)

        super().on_end(span)
