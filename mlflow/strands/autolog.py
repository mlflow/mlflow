import json
import logging

from opentelemetry.context import Context
from opentelemetry.sdk.trace import (
    ReadableSpan as OTelReadableSpan,
)
from opentelemetry.sdk.trace import (
    Span as OTelSpan,
)
from opentelemetry.sdk.trace import (
    TracerProvider as SDKTracerProvider,
)
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter
from opentelemetry.trace import (
    NoOpTracer,
    NoOpTracerProvider,
    ProxyTracerProvider,
    get_tracer_provider,
    set_tracer_provider,
)

from mlflow.entities import SpanType
from mlflow.entities.span import LiveSpan, create_mlflow_span
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from mlflow.tracing.provider import _get_tracer
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import (
    _bypass_attribute_guard,
    get_mlflow_span_for_otel_span,
    get_otel_attribute,
)

_logger = logging.getLogger(__name__)


class StrandsSpanProcessor(SimpleSpanProcessor):
    def __init__(self):
        self.span_exporter = SpanExporter()

    def on_start(self, span: OTelSpan, parent_context: Context | None = None):
        tracer = _get_tracer(__name__)
        if isinstance(tracer, NoOpTracer):
            return

        tracer.span_processor.on_start(span, parent_context)
        trace_id = get_otel_attribute(span, SpanAttributeKey.REQUEST_ID)
        mlflow_span = create_mlflow_span(span, trace_id)
        InMemoryTraceManager.get_instance().register_span(mlflow_span)

    def on_end(self, span: OTelReadableSpan) -> None:
        mlflow_span = get_mlflow_span_for_otel_span(span)
        if mlflow_span is None:
            _logger.debug("Span not found in the map. Skipping end.")
            return
        with _bypass_attribute_guard(mlflow_span._span):
            _set_span_type(mlflow_span, span)
            _set_inputs_outputs(mlflow_span, span)
            _set_token_usage(mlflow_span, span)
        tracer = _get_tracer(__name__)
        tracer.span_processor.on_end(span)


def setup_strands_tracing():
    processor = StrandsSpanProcessor()
    provider = get_tracer_provider()
    if isinstance(provider, (NoOpTracerProvider, ProxyTracerProvider)):
        new_provider = SDKTracerProvider()
        new_provider.add_span_processor(processor)
        set_tracer_provider(new_provider)
    else:
        if not any(
            isinstance(p, StrandsSpanProcessor)
            for p in provider._active_span_processor._span_processors
        ):
            provider.add_span_processor(processor)


def teardown_strands_tracing():
    provider = get_tracer_provider()
    if isinstance(provider, SDKTracerProvider):
        span_processors = getattr(provider._active_span_processor, "_span_processors", ())
        provider._active_span_processor._span_processors = tuple(
            p for p in span_processors if not isinstance(p, StrandsSpanProcessor)
        )


def _set_span_type(mlflow_span: LiveSpan, span: OTelReadableSpan) -> None:
    operation = span.attributes.get("gen_ai.operation.name")
    # "invoke_agent" for single agent and "invoke_{agent_name}" for multi agents
    if isinstance(operation, str) and operation.startswith("invoke_"):
        mlflow_span.set_span_type(SpanType.AGENT)
    elif operation == "execute_tool":
        mlflow_span.set_span_type(SpanType.TOOL)
    elif operation == "chat":
        mlflow_span.set_span_type(SpanType.CHAT_MODEL)
    else:
        pass


def _parse_json(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _set_inputs_outputs(mlflow_span: LiveSpan, span: OTelReadableSpan) -> None:
    inputs = []
    outputs = []
    for event in span.events:
        if event.name in {"gen_ai.user.message", "gen_ai.tool.message"}:
            content = _parse_json(event.attributes.get("content"))
            role = "user" if event.name == "gen_ai.user.message" else "tool"
            inputs.append({"role": role, "content": content})
        elif event.name == "gen_ai.choice":
            message = _parse_json(event.attributes.get("message"))
            outputs.append(message)
    if inputs:
        mlflow_span.set_inputs(inputs)
    if outputs:
        mlflow_span.set_outputs(outputs if len(outputs) > 1 else outputs[0])


def _set_token_usage(mlflow_span: LiveSpan, span: OTelReadableSpan) -> None:
    # Strands agents contain complete token usage information in the AGENT span
    # We don't need to set token usage for the AGENT span to avoid double counting
    if mlflow_span.get_attribute(SpanAttributeKey.SPAN_TYPE) == SpanType.AGENT:
        return

    usage = {}
    if (v := span.attributes.get("gen_ai.usage.input_tokens")) is not None:
        usage[TokenUsageKey.INPUT_TOKENS] = v
    if (v := span.attributes.get("gen_ai.usage.output_tokens")) is not None:
        usage[TokenUsageKey.OUTPUT_TOKENS] = v
    if (v := span.attributes.get("gen_ai.usage.total_tokens")) is not None:
        usage[TokenUsageKey.TOTAL_TOKENS] = v
    if usage:
        mlflow_span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage)
