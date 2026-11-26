import logging
import threading

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan
from opentelemetry.sdk.trace import Span as OTelSpan
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter
from opentelemetry.trace import (
    NoOpTracer,
    NoOpTracerProvider,
    ProxyTracerProvider,
    get_tracer_provider,
    set_tracer_provider,
)

from mlflow.entities import SpanType
from mlflow.entities.span import create_mlflow_span
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracing.provider import (
    _add_span_processor_first,
    _get_tracer_for_span_end,
    _get_tracer_for_span_start,
)
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import (
    _bypass_attribute_guard,
    get_mlflow_span_for_otel_span,
    get_otel_attribute,
)

_logger = logging.getLogger(__name__)

PLUGIN_NAME = "mlflow_google_adk"

_SPAN_TYPE_BY_OPERATION = {
    "invoke_agent": SpanType.AGENT,
    "invoke_workflow": SpanType.AGENT,
    "execute_tool": SpanType.TOOL,
    "generate_content": SpanType.LLM,
}


class GoogleADKSpanProcessor(SimpleSpanProcessor):
    """Forwards ADK's OpenTelemetry spans to MLflow's tracer and sets MLflow span types."""

    def __init__(self):
        self.span_exporter = SpanExporter()
        self._processing_local = threading.local()

    def on_start(self, span: OTelSpan, parent_context: Context | None = None):
        # Recursion guard: with a shared tracer provider, forwarding re-enters this processor.
        if getattr(self._processing_local, "in_on_start", False):
            return
        self._processing_local.in_on_start = True
        try:
            tracer, should_forward = _get_tracer_for_span_start(self, __name__)
            if isinstance(tracer, NoOpTracer) or not should_forward:
                return
            tracer.span_processor.on_start(span, parent_context)
            trace_id = get_otel_attribute(span, SpanAttributeKey.REQUEST_ID)
            if trace_id is None:
                return
            InMemoryTraceManager.get_instance().register_span(create_mlflow_span(span, trace_id))
        finally:
            self._processing_local.in_on_start = False

    def on_end(self, span: OTelReadableSpan) -> None:
        if getattr(self._processing_local, "in_on_end", False):
            return
        self._processing_local.in_on_end = True
        try:
            if mlflow_span := get_mlflow_span_for_otel_span(span):
                with _bypass_attribute_guard(mlflow_span._span):
                    if span_type := _span_type(span):
                        mlflow_span.set_span_type(span_type)
            tracer, should_forward = _get_tracer_for_span_end(self, __name__)
            if should_forward and not isinstance(tracer, NoOpTracer):
                tracer.span_processor.on_end(span)
        finally:
            self._processing_local.in_on_end = False


def _span_type(span: OTelReadableSpan) -> str | None:
    attributes = span.attributes or {}
    if span.name == "invocation":
        return SpanType.AGENT
    if span.name == "call_llm":
        return SpanType.LLM
    return _SPAN_TYPE_BY_OPERATION.get(attributes.get("gen_ai.operation.name"))


def patched_runner_init(original, self, *args, **kwargs):
    original(self, *args, **kwargs)
    plugin_manager = getattr(self, "plugin_manager", None)
    if plugin_manager is not None and plugin_manager.get_plugin(PLUGIN_NAME) is None:
        # Imported here: the plugin subclasses ADK's BasePlugin, and this module must load
        # without google-adk installed.
        from mlflow.google_adk.plugin import MlflowTurnPlugin

        plugin_manager.register_plugin(MlflowTurnPlugin())


def setup_google_adk_tracing():
    processor = GoogleADKSpanProcessor()
    provider = get_tracer_provider()
    if isinstance(provider, (NoOpTracerProvider, ProxyTracerProvider)):
        new_provider = SDKTracerProvider()
        new_provider.add_span_processor(processor)
        set_tracer_provider(new_provider)
    elif not any(
        isinstance(p, GoogleADKSpanProcessor)
        for p in provider._active_span_processor._span_processors
    ):
        _add_span_processor_first(provider, processor)


def teardown_google_adk_tracing():
    provider = get_tracer_provider()
    if isinstance(provider, SDKTracerProvider):
        span_processors = getattr(provider._active_span_processor, "_span_processors", ())
        provider._active_span_processor._span_processors = tuple(
            p for p in span_processors if not isinstance(p, GoogleADKSpanProcessor)
        )
