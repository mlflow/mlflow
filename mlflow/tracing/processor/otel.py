import json
import uuid
from typing import Optional

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan
from opentelemetry.sdk.trace import Span as OTelSpan
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter

from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_status import TraceStatus
from mlflow.tracing.constant import TRACE_SCHEMA_VERSION, TRACE_SCHEMA_VERSION_KEY, SpanAttributeKey
from mlflow.tracing.trace_manager import InMemoryTraceManager


class OtelSpanProcessor(BatchSpanProcessor):
    """
    SpanProcessor implementation to export MLflow traces to a OpenTelemetry collector.

    Extending OpenTelemetry BatchSpanProcessor to add some custom hooks to be executed when a span
    is started or ended (before exporting).
    """

    def __init__(self, span_exporter: SpanExporter):
        super().__init__(span_exporter)
        self.span_exporter = span_exporter
        self._trace_manager = InMemoryTraceManager.get_instance()

    def on_start(self, span: OTelSpan, parent_context: Optional[Context] = None):
        """
        Handle the start of a span. This method is called when an OpenTelemetry span is started.

        Args:
            span: An OpenTelemetry Span object that is started.
            parent_context: The context of the span. Note that this is only passed when the context
                object is explicitly specified to OpenTelemetry start_span call. If the parent
                span is obtained from the global context, it won't be passed here so we should not
                rely on it.
        """
        # Generate a random request ID and trace info just for the sake of consistency
        # with other tracing destinations. Doing this makes it much easier to handle
        # multiple tracing destinations.
        request_id = uuid.uuid4().hex
        trace_info = TraceInfo(
            request_id=request_id,
            experiment_id=None,
            timestamp_ms=span.start_time // 1_000_000,  # nanosecond to millisecond
            execution_time_ms=None,
            status=TraceStatus.IN_PROGRESS,
            request_metadata={TRACE_SCHEMA_VERSION_KEY: str(TRACE_SCHEMA_VERSION)},
            tags={},
        )
        span.set_attribute(SpanAttributeKey.REQUEST_ID, json.dumps(request_id))

        self._trace_manager.register_trace(span.context.trace_id, trace_info)

        super().on_start(span, parent_context)

    def on_end(self, span: OTelReadableSpan):
        # Pops the trace entry from the in-memory trace manager to avoid memory leak
        if span._parent is None:
            self._trace_manager.pop_trace(span.context.trace_id)

        super().on_end(span)
