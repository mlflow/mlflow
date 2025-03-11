import json
import logging
from typing import Optional

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan
from opentelemetry.sdk.trace import Span as OTelSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter

from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_status import TraceStatus
from mlflow.tracing.constant import TRACE_SCHEMA_VERSION, TRACE_SCHEMA_VERSION_KEY, SpanAttributeKey
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import (
    deduplicate_span_names_in_place,
    get_otel_attribute,
    maybe_get_dependencies_schemas,
)
from mlflow.tracking.fluent import _get_experiment_id

_logger = logging.getLogger(__name__)


class DatabricksSpanProcessor(SimpleSpanProcessor):
    """
    Defines custom hooks to be executed when a span is started or ended (before exporting).

    This process implements simple responsibilities to generate MLflow-style trace
    object from OpenTelemetry spans and store them in memory.
    """

    def __init__(
        self,
        span_exporter: SpanExporter,
        experiment_id: Optional[str] = None,
    ):
        self.span_exporter = span_exporter
        self._trace_manager = InMemoryTraceManager.get_instance()
        self._experiment_id = experiment_id

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

        request_id = self._create_or_get_request_id(span)
        span.set_attribute(SpanAttributeKey.REQUEST_ID, json.dumps(request_id))

        tags = {}
        if dependencies_schema := maybe_get_dependencies_schemas():
            tags.update(dependencies_schema)

        if span._parent is None:
            trace_info = TraceInfo(
                request_id=request_id,
                experiment_id=self._experiment_id or _get_experiment_id(),
                timestamp_ms=span.start_time // 1_000_000,  # nanosecond to millisecond
                execution_time_ms=None,
                status=TraceStatus.IN_PROGRESS,
                request_metadata={TRACE_SCHEMA_VERSION_KEY: str(TRACE_SCHEMA_VERSION)},
                tags=tags,
            )
            self._trace_manager.register_trace(span.context.trace_id, trace_info)

    def _create_or_get_request_id(self, span: OTelSpan) -> str:
        if span._parent is None:
            return str(span.context.trace_id)  # Use otel-generated trace_id as request_id
        else:
            return self._trace_manager.get_request_id_from_trace_id(span.context.trace_id)

    def on_end(self, span: OTelReadableSpan) -> None:
        """
        Handle the end of a span. This method is called when an OpenTelemetry span is ended.

        Args:
            span: An OpenTelemetry ReadableSpan object that is ended.
        """
        # Processing the trace only when it is a root span.
        if span._parent is None:
            request_id = get_otel_attribute(span, SpanAttributeKey.REQUEST_ID)
            with self._trace_manager.get_trace(request_id) as trace:
                if trace is None:
                    _logger.debug(f"Trace data with request ID {request_id} not found.")
                    return

                trace.info.execution_time_ms = (span.end_time - span.start_time) // 1_000_000
                trace.info.status = TraceStatus.from_otel_status(span.status)
                deduplicate_span_names_in_place(list(trace.span_dict.values()))

        super().on_end(span)
