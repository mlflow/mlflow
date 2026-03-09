"""
Core translator for converting MLflow spans to OpenTelemetry GenAI Semantic Convention format.

Translates universal attributes (model, provider, tokens, span type) that are already
normalized across all autologging integrations. Format-specific message content conversion
will be added in a follow-up via per-integration converters.
"""

import json
import logging
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import SpanKind

from mlflow.entities.span import SpanType
from mlflow.tracing.constant import GenAiSemconvKey, SpanAttributeKey
from mlflow.tracing.utils import get_otel_attribute

_logger = logging.getLogger(__name__)

# Span type → GenAI operation name mapping.
# https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
_SPAN_TYPE_TO_OPERATION: dict[str, str | None] = {
    SpanType.CHAT_MODEL: "chat",
    SpanType.LLM: "generate_content",
    SpanType.EMBEDDING: "embeddings",
    SpanType.TOOL: "execute_tool",
    SpanType.AGENT: "invoke_agent",
    # No natural GenAI semconv equivalent — pass through as-is
    SpanType.RETRIEVER: None,
    SpanType.RERANKER: None,
    SpanType.CHAIN: None,
    SpanType.WORKFLOW: None,
    SpanType.PARSER: None,
    SpanType.MEMORY: None,
    SpanType.GUARDRAIL: None,
    SpanType.EVALUATOR: None,
    SpanType.TASK: None,
    SpanType.UNKNOWN: None,
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

    # Merge: Keep non-mlflow.* attrs, add GenAI attrs
    merged_attrs = {k: v for k, v in original_attrs.items() if not k.startswith("mlflow.")}
    merged_attrs.update(genai_attrs)

    new_name = _build_genai_span_name(span.name, genai_attrs)
    new_kind = _get_genai_span_kind(genai_attrs, span.kind)

    return _build_readable_span(span, name=new_name, attributes=merged_attrs, kind=new_kind)


def _translate_universal_attributes(span: ReadableSpan) -> dict[str, Any]:
    genai_attrs: dict[str, Any] = {}

    # 1. Operation name from span type
    if span_type := get_otel_attribute(span, SpanAttributeKey.SPAN_TYPE):
        if operation := _SPAN_TYPE_TO_OPERATION.get(span_type):
            genai_attrs[GenAiSemconvKey.OPERATION_NAME] = operation

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


def _build_genai_span_name(original_name: str, genai_attrs: dict[str, Any]) -> str:
    """
    Build GenAI semconv span name: "{operation} {model}" (e.g., "chat gpt-4o").

    Falls back to the original span name if operation or model is missing.
    """
    operation = genai_attrs.get(GenAiSemconvKey.OPERATION_NAME)
    model = genai_attrs.get(GenAiSemconvKey.REQUEST_MODEL)

    if operation and model:
        return f"{operation} {model}"
    if operation:
        return operation
    return original_name


def _get_genai_span_kind(genai_attrs: dict[str, Any], original_kind: SpanKind) -> SpanKind:
    """
    Get the correct SpanKind for GenAI semconv.

    GenAI semconv requires CLIENT for inference spans and INTERNAL for tool/agent spans.
    """
    operation = genai_attrs.get(GenAiSemconvKey.OPERATION_NAME)
    if operation and operation in _OPERATION_TO_SPAN_KIND:
        return _OPERATION_TO_SPAN_KIND[operation]
    return original_kind


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


