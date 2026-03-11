"""
Core translator for converting MLflow spans to OpenTelemetry GenAI Semantic Convention format.

Phase 1: Universal attributes (model, provider, tokens, span type) normalized across all
autologging integrations.
Phase 2: Format-specific message content (gen_ai.input.messages, gen_ai.output.messages),
request params, and response attrs via per-integration converters.
"""

import json
import logging
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import SpanKind

from mlflow.entities.span import SpanType
from mlflow.tracing.constant import GenAiSemconvKey, SpanAttributeKey
from mlflow.tracing.export.genai_semconv.converter import GenAiSemconvConverter
from mlflow.tracing.utils import get_otel_attribute

_logger = logging.getLogger(__name__)

# Span type → GenAI operation name mapping.
# https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
_SPAN_TYPE_TO_OPERATION: dict[str, str] = {
    SpanType.CHAT_MODEL: "chat",
    SpanType.LLM: "generate_content",
    SpanType.EMBEDDING: "embeddings",
    SpanType.TOOL: "execute_tool",
    SpanType.AGENT: "invoke_agent",
}

# GenAI semconv requires CLIENT for inference spans and INTERNAL for tool/agent spans.
# https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
_OPERATION_TO_SPAN_KIND: dict[str, SpanKind] = {
    "chat": SpanKind.CLIENT,
    "text_completion": SpanKind.CLIENT,
    "embeddings": SpanKind.CLIENT,
    "generate_content": SpanKind.CLIENT,
    "execute_tool": SpanKind.INTERNAL,
    "invoke_agent": SpanKind.INTERNAL,
}


def translate_span_to_genai(span: ReadableSpan) -> ReadableSpan:
    """
    Translate a single MLflow span to GenAI Semantic Convention format.

    Args:
        span: The original OTel ReadableSpan with mlflow.* attributes.

    Returns:
        A new ReadableSpan with GenAI semconv attributes.
    """
    original_attrs = dict(span.attributes or {})

    genai_attrs = _translate_universal_attributes(span)

    if not genai_attrs:
        # No GenAI mapping — strip mlflow.* attrs and pass through
        return _create_passthrough_span(span, original_attrs)

    # Phase 2: Format-specific message translation via converter
    message_format = get_otel_attribute(span, SpanAttributeKey.MESSAGE_FORMAT)
    inputs = get_otel_attribute(span, SpanAttributeKey.INPUTS)
    outputs = get_otel_attribute(span, SpanAttributeKey.OUTPUTS)

    if inputs is not None or outputs is not None:
        if converter := _get_converter(message_format, inputs):
            try:
                genai_attrs.update(converter.translate(inputs, outputs))
            except Exception:
                _logger.debug("Failed to convert messages for format %r, skipping", message_format)

    # Merge: Keep non-mlflow.* attrs, add GenAI attrs
    merged_attrs = {k: v for k, v in original_attrs.items() if not k.startswith("mlflow.")}
    merged_attrs.update(genai_attrs)

    new_name = _build_genai_span_name(span.name, genai_attrs)
    new_kind = _get_genai_span_kind(genai_attrs, span.kind)

    return _build_readable_span(span, name=new_name, attributes=merged_attrs, kind=new_kind)


def _translate_universal_attributes(span: ReadableSpan) -> dict[str, Any]:
    genai_attrs: dict[str, Any] = {}

    # 1. Operation name from span type (use mapped value or pass through as-is)
    if span_type := get_otel_attribute(span, SpanAttributeKey.SPAN_TYPE):
        genai_attrs[GenAiSemconvKey.OPERATION_NAME] = _SPAN_TYPE_TO_OPERATION.get(
            span_type, span_type
        )

    # 2. Model
    if model := get_otel_attribute(span, SpanAttributeKey.MODEL):
        genai_attrs[GenAiSemconvKey.REQUEST_MODEL] = model

    # 3. Provider
    if provider := get_otel_attribute(span, SpanAttributeKey.MODEL_PROVIDER):
        genai_attrs[GenAiSemconvKey.PROVIDER_NAME] = provider

    # 4. Token usage
    if isinstance(usage := get_otel_attribute(span, SpanAttributeKey.CHAT_USAGE), dict):
        if (input_tokens := usage.get("input_tokens")) is not None:
            genai_attrs[GenAiSemconvKey.USAGE_INPUT_TOKENS] = input_tokens
        if (output_tokens := usage.get("output_tokens")) is not None:
            genai_attrs[GenAiSemconvKey.USAGE_OUTPUT_TOKENS] = output_tokens

    # 5. Tool attributes (for TOOL spans)
    if span_type == SpanType.TOOL:
        if (inputs := get_otel_attribute(span, SpanAttributeKey.INPUTS)) is not None:
            genai_attrs[GenAiSemconvKey.TOOL_CALL_ARGUMENTS] = json.dumps(inputs)
        if (outputs := get_otel_attribute(span, SpanAttributeKey.OUTPUTS)) is not None:
            genai_attrs[GenAiSemconvKey.TOOL_CALL_RESULT] = json.dumps(outputs)

    return genai_attrs


def _get_converter(
    message_format: str | None, inputs: dict[str, Any] | None = None
) -> GenAiSemconvConverter | None:
    match message_format:
        case "openai":
            if inputs is not None and "input" in inputs:
                from mlflow.openai.genai_semconv_converter import OpenAIResponsesConverter

                return OpenAIResponsesConverter()

            from mlflow.openai.genai_semconv_converter import OpenAIChatCompletionConverter

            return OpenAIChatCompletionConverter()
        case _:
            from mlflow.openai.genai_semconv_converter import OpenAIChatCompletionConverter

            return OpenAIChatCompletionConverter()


def _build_genai_span_name(original_name: str, genai_attrs: dict[str, Any]) -> str:
    """
    Build GenAI semconv span name: "{operation} {model}" (e.g., "chat gpt-4o").

    Falls back to the original span name if operation or model is missing.
    """
    operation = genai_attrs.get(GenAiSemconvKey.OPERATION_NAME)
    model = genai_attrs.get(GenAiSemconvKey.REQUEST_MODEL)

    if operation and model:
        return f"{operation} {model}"
    return operation or original_name


def _get_genai_span_kind(genai_attrs: dict[str, Any], original_kind: SpanKind) -> SpanKind:
    """
    Get the correct SpanKind for GenAI semconv.

    GenAI semconv requires CLIENT for inference spans and INTERNAL for tool/agent spans.
    """
    operation = genai_attrs.get(GenAiSemconvKey.OPERATION_NAME)
    return _OPERATION_TO_SPAN_KIND.get(operation, original_kind) if operation else original_kind


def _create_passthrough_span(span: ReadableSpan, original_attrs: dict[str, Any]) -> ReadableSpan:
    cleaned_attrs = {k: v for k, v in original_attrs.items() if not k.startswith("mlflow.")}
    return _build_readable_span(span, name=span.name, attributes=cleaned_attrs, kind=span.kind)


def _build_readable_span(
    original: ReadableSpan,
    name: str,
    attributes: dict[str, Any],
    kind: SpanKind,
) -> ReadableSpan:
    """
    Construct a new ReadableSpan with overridden name, attributes, and kind.

    ReadableSpan objects are frozen, so we must create new instances.
    """
    return ReadableSpan(
        name=name,
        context=original.context,
        parent=original.parent,
        resource=original.resource,
        attributes=attributes,
        events=original.events,
        links=original.links,
        kind=kind,
        instrumentation_scope=original.instrumentation_scope,
        status=original.status,
        start_time=original.start_time,
        end_time=original.end_time,
    )
