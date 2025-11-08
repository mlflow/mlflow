import json
import logging
from typing import Any

from haystack.tracing import OpenTelemetryTracer, enable_tracing
from opentelemetry import trace
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

from mlflow.entities import LiveSpan, SpanType
from mlflow.entities.span import create_mlflow_span
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from mlflow.tracing.provider import _get_tracer
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import (
    _bypass_attribute_guard,
    generate_trace_id_v3,
    get_mlflow_span_for_otel_span,
)

_logger = logging.getLogger(__name__)


def setup_haystack_tracing():
    from haystack import tracing as hs_tracing

    hs_tracing.tracer.is_content_tracing_enabled = True

    provider = get_tracer_provider()
    hs_processor = HaystackSpanProcessor()
    if isinstance(provider, (NoOpTracerProvider, ProxyTracerProvider)):
        new_provider = SDKTracerProvider()
        new_provider.add_span_processor(hs_processor)
        set_tracer_provider(new_provider)
    else:
        if not any(
            isinstance(p, HaystackSpanProcessor)
            for p in provider._active_span_processor._span_processors
        ):
            provider.add_span_processor(hs_processor)

    tracer = trace.get_tracer(__name__)
    enable_tracing(OpenTelemetryTracer(tracer))


def _infer_span_type_from_haystack(
    comp_type: str | None,
    comp_alias: str | None,
    span: OTelReadableSpan,
) -> SpanType:
    s = (comp_type or comp_alias or span.name or "").lower()

    if any(
        k in s
        for k in (
            "llm",
            "chat",
            "generator",
            "completion",
            "textgen",
            "chatgenerator",
            "openai",
            "anthropic",
            "mistral",
            "cohere",
            "gemini",
        )
    ):
        return SpanType.LLM

    if "embedder" in s:
        return SpanType.EMBEDDING

    if "retriever" in s:
        return SpanType.RETRIEVER

    if "ranker" in s:
        return SpanType.RERANKER

    if "agent" in s:
        return SpanType.AGENT

    return SpanType.TOOL


class HaystackSpanProcessor(SimpleSpanProcessor):
    def __init__(self):
        self.span_exporter = SpanExporter()
        self._pipeline_io: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}

    def on_start(self, span: OTelSpan, parent_context: Context | None = None):
        tracer = _get_tracer(__name__)
        tracer.span_processor.on_start(span, parent_context)

        trace_id = generate_trace_id_v3(span)
        mlflow_span = create_mlflow_span(span, trace_id)
        InMemoryTraceManager.get_instance().register_span(mlflow_span)

    def on_end(self, span: OTelReadableSpan) -> None:
        mlflow_span = get_mlflow_span_for_otel_span(span)
        if mlflow_span is None:
            _logger.debug("Span not found in the map. Skipping end.")
            return

        with _bypass_attribute_guard(mlflow_span._span):
            if span.name in ("haystack.pipeline.run", "haystack.async_pipeline.run"):
                self.set_pipeline_info(mlflow_span, span)
            elif span.name in ("haystack.component.run"):
                self.set_component_info(mlflow_span, span)

        tracer = _get_tracer(__name__)
        tracer.span_processor.on_end(span)

    def set_component_info(self, mlflow_span: LiveSpan, span: OTelReadableSpan) -> None:
        comp_alias = span.attributes.get("haystack.component.name")
        comp_type = span.attributes.get("haystack.component.type")
        mlflow_span.set_span_type(_infer_span_type_from_haystack(comp_type, comp_alias, span))

        # Haystack spans originally have name='haystack.component.run'. We need to update both the
        #  _name field of the Otel span and the _original_name field of the MLflow span to
        # customize the span name here, as otherwise it would be overwritten in the
        # deduplication process
        span_name = comp_type or comp_alias or span.name
        mlflow_span._span._name = span_name
        mlflow_span._original_name = span_name

        if (inputs := span.attributes.get("haystack.component.input")) is not None:
            try:
                mlflow_span.set_inputs(json.loads(inputs))
            except Exception:
                mlflow_span.set_inputs(inputs)
        if (outputs := span.attributes.get("haystack.component.output")) is not None:
            try:
                mlflow_span.set_outputs(json.loads(outputs))
            except Exception:
                mlflow_span.set_outputs(outputs)

        if usage := _parse_token_usage(mlflow_span.outputs):
            mlflow_span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage)

        parent_id = mlflow_span.parent_id
        if parent_id:
            key = comp_alias or comp_type or mlflow_span.name
            inputs_agg, outputs_agg = self._pipeline_io.setdefault(parent_id, ({}, {}))
            if mlflow_span.inputs is not None:
                inputs_agg[key] = mlflow_span.inputs
            if mlflow_span.outputs is not None:
                outputs_agg[key] = mlflow_span.outputs

    def set_pipeline_info(self, mlflow_span: LiveSpan, span: OTelReadableSpan) -> None:
        # Pipelines are CHAINs
        mlflow_span.set_span_type(SpanType.CHAIN)

        pipe_name = span.attributes.get("haystack.pipeline.name")
        if pipe_name:
            mlflow_span._span._name = pipe_name

        if (inputs := span.attributes.get("haystack.pipeline.input")) is not None:
            try:
                mlflow_span.set_inputs(json.loads(inputs))
            except Exception:
                mlflow_span.set_inputs(inputs)
        if (outputs := span.attributes.get("haystack.pipeline.output")) is not None:
            try:
                mlflow_span.set_outputs(json.loads(outputs))
            except Exception:
                mlflow_span.set_outputs(outputs)

        if mlflow_span.span_id in self._pipeline_io:
            inputs_agg, outputs_agg = self._pipeline_io.pop(mlflow_span.span_id)
            if mlflow_span.inputs is None and inputs_agg:
                mlflow_span.set_inputs(inputs_agg)
            if mlflow_span.outputs is None and outputs_agg:
                mlflow_span.set_outputs(outputs_agg)


def _parse_token_usage(outputs: Any) -> dict[str, int] | None:
    try:
        if not isinstance(outputs, dict):
            return None

        replies = outputs.get("replies")
        if isinstance(replies, list) and len(replies) > 0:
            usage = (
                replies[0].get("meta", {}).get("usage", {}) if isinstance(replies[0], dict) else {}
            )

        meta = outputs.get("meta")
        if isinstance(meta, list) and len(meta) > 0:
            usage = meta[0].get("usage", {}) if isinstance(meta[0], dict) else {}

        if isinstance(usage, dict):
            in_tok = usage.get("prompt_tokens", 0)
            out_tok = usage.get("completion_tokens", 0)
            tot_tok = usage.get("total_tokens", 0)
            return {
                TokenUsageKey.INPUT_TOKENS: in_tok,
                TokenUsageKey.OUTPUT_TOKENS: out_tok,
                TokenUsageKey.TOTAL_TOKENS: tot_tok,
            }
    except Exception:
        _logger.debug("Failed to parse token usage from outputs.", exc_info=True)


def teardown_haystack_tracing():
    provider = get_tracer_provider()
    if isinstance(provider, SDKTracerProvider):
        span_processors = getattr(provider._active_span_processor, "_span_processors", ())
        provider._active_span_processor._span_processors = tuple(
            p for p in span_processors if not isinstance(p, HaystackSpanProcessor)
        )
