import copy
import logging

from opentelemetry.attributes import BoundedAttributes
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter

from mlflow.entities.span import create_mlflow_span
from mlflow.entities.trace_info import TraceInfo, TraceLocation, TraceState
from mlflow.environment_variables import (
    MLFLOW_ENABLE_OTEL_GENAI_SEMCONV,
    MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT,
)
from mlflow.tracing.constant import TRACE_SCHEMA_VERSION, TRACE_SCHEMA_VERSION_KEY, SpanAttributeKey
from mlflow.tracing.processor.otel_metrics_mixin import OtelMetricsMixin
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import generate_trace_id_v3

_logger = logging.getLogger(__name__)


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
            manager_trace = self._trace_manager.pop_trace(span.context.trace_id)
            # Emit user-defined tags as individual span attributes so they survive OTLP export
            # and can be restored to trace tag rows on the server side.
            if manager_trace is not None:
                try:
                    span = self._with_trace_tag_attributes(span, manager_trace.trace.info.tags)
                except Exception:
                    _logger.debug("Failed to emit trace tags as span attributes", exc_info=True)

        if MLFLOW_ENABLE_OTEL_GENAI_SEMCONV.get():
            span = self._translate_span(span)

        super().on_end(span)

    def _with_trace_tag_attributes(
        self, span: OTelReadableSpan, tags: dict[str, str]
    ) -> OTelReadableSpan:
        tag_attributes = {
            SpanAttributeKey.TRACE_TAG_PREFIX + key: value
            for key, value in tags.items()
            if not key.startswith("mlflow.")
        }
        if not tag_attributes:
            return span
        # The attributes object is shared with the live span and with every other span
        # processor, and opentelemetry-sdk >= 1.43 freezes it before `on_end`. Export a copy
        # carrying the tags instead of writing to the shared object. Keep the original limits
        # and dropped count so the copy matches what an in-place write would have produced.
        original = span._attributes
        attributes = BoundedAttributes(
            maxlen=getattr(original, "maxlen", None),
            attributes={**(span.attributes or {}), **tag_attributes},
            max_value_len=getattr(original, "max_value_len", None),
        )
        attributes.dropped += getattr(original, "dropped", 0)
        span = copy.copy(span)
        span._attributes = attributes
        return span

    def _translate_span(self, span: OTelReadableSpan) -> OTelReadableSpan:
        try:
            from mlflow.tracing.export.genai_semconv.translator import translate_span_to_genai

            return translate_span_to_genai(span)
        except Exception:
            _logger.debug("Failed to translate span to GenAI semconv", exc_info=True)
            return span

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
