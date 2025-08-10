import json
import logging
from contextlib import contextmanager
from typing import Generator

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan
from opentelemetry.sdk.trace import Span as OTelSpan
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter
from opentelemetry.trace import (
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
from mlflow.tracing.utils import get_mlflow_span_for_otel_span, get_otel_attribute

_logger = logging.getLogger(__name__)


class StrandsSpanProcessor(SimpleSpanProcessor):
    def __init__(self):
        self.span_exporter = SpanExporter()

    def on_start(self, span: OTelSpan, parent_context: Context | None = None):
        tracer = _get_tracer(__name__)
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


def _set_span_type(mlflow_span: LiveSpan, span: OTelReadableSpan) -> None:
    operation = span.attributes.get("gen_ai.operation.name")
    if isinstance(operation, str) and (
        operation == "invoke_agent" or operation.startswith("invoke_")
    ):
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
    for event in span.events:
        if event.name in {"gen_ai.user.message", "gen_ai.tool.message"}:
            content = _parse_json(event.attributes.get("content"))
            if event.name == "gen_ai.user.message":
                mlflow_span.set_inputs({"content": content})
            else:
                mlflow_span.set_inputs(content)
        elif event.name == "gen_ai.choice":
            message = _parse_json(event.attributes.get("message"))
            mlflow_span.set_outputs(message)


def _set_token_usage(mlflow_span: LiveSpan, span: OTelReadableSpan) -> None:
    usage = {}
    if (v := span.attributes.get("gen_ai.usage.input_tokens")) is not None:
        usage[TokenUsageKey.INPUT_TOKENS] = v
    if (v := span.attributes.get("gen_ai.usage.output_tokens")) is not None:
        usage[TokenUsageKey.OUTPUT_TOKENS] = v
    if (v := span.attributes.get("gen_ai.usage.total_tokens")) is not None:
        usage[TokenUsageKey.TOTAL_TOKENS] = v
    if usage:
        mlflow_span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage)


@contextmanager
def _bypass_attribute_guard(span: OTelSpan) -> Generator[None, None, None]:
    original_end_time = span._end_time
    span._end_time = None
    try:
        yield
    finally:
        span._end_time = original_end_time
