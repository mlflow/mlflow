"""
Utilities for translating OTEL span kinds to MLflow span types.

This module provides functions to translate span kind attributes from various
OTEL semantic conventions (OpenInference, Traceloop) to MLflow span types.
"""

import json
import logging
from typing import Any

from mlflow.entities.span import Span, SpanType
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from mlflow.tracing.utils import dump_span_attribute_value

_logger = logging.getLogger(__name__)

OPENINFERENCE_SPAN_KIND_ATTRIBUTE_KEY = "openinference.span.kind"
# Mapping from OpenInference span kinds to MLflow span types
# Reference: https://github.com/Arize-ai/openinference/blob/50eaf3c943d818f12fdc8e37b7c305c763c82050/python/openinference-semantic-conventions/src/openinference/semconv/trace/__init__.py#L356
OPENINFERENCE_SPAN_KIND_TO_MLFLOW_TYPE = {
    "TOOL": SpanType.TOOL,
    "CHAIN": SpanType.CHAIN,
    "LLM": SpanType.LLM,
    "RETRIEVER": SpanType.RETRIEVER,
    "EMBEDDING": SpanType.EMBEDDING,
    "AGENT": SpanType.AGENT,
    "RERANKER": SpanType.RERANKER,
    "UNKNOWN": SpanType.UNKNOWN,
    "GUARDRAIL": SpanType.GUARDRAIL,
    "EVALUATOR": SpanType.EVALUATOR,
}

TRACELOOP_SPAN_KIND_ATTRIBUTE_KEY = "traceloop.span.kind"
# Mapping from Traceloop span kinds to MLflow span types
# Reference: https://github.com/traceloop/openllmetry/blob/e66894fd7f8324bd7b2972d7f727da39e7d93181/packages/opentelemetry-semantic-conventions-ai/opentelemetry/semconv_ai/__init__.py#L301
TRACELOOP_SPAN_KIND_TO_MLFLOW_TYPE = {
    "workflow": SpanType.WORKFLOW,
    "task": SpanType.TASK,
    "agent": SpanType.AGENT,
    "tool": SpanType.TOOL,
    "unknown": SpanType.UNKNOWN,
}

INPUT_TOKEN_ATTRIBUTE_KEYS = {
    # OTEL semantic conventions: https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/#genai-attributes
    "gen_ai.usage.input_tokens",
    # openllmetry semantic conventions: https://github.com/traceloop/openllmetry/blob/e66894fd7f8324bd7b2972d7f727da39e7d93181/packages/opentelemetry-semantic-conventions-ai/opentelemetry/semconv_ai/__init__.py#L80
    "gen_ai.usage.prompt_tokens",
    # openinference semantic conventions: https://github.com/Arize-ai/openinference/blob/c80c81b8d6fa564598bd359cdd7313f4472ceca8/python/openinference-semantic-conventions/src/openinference/semconv/trace/__init__.py#L93
    "llm.token_count.prompt",
}

OUTPUT_TOKEN_ATTRIBUTE_KEYS = {
    # OTEL semantic conventions: https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/#genai-attributes
    "gen_ai.usage.output_tokens",
    # openllmetry semantic conventions: https://github.com/traceloop/openllmetry/blob/e66894fd7f8324bd7b2972d7f727da39e7d93181/packages/opentelemetry-semantic-conventions-ai/opentelemetry/semconv_ai/__init__.py#L78
    "gen_ai.usage.completion_tokens",
    # openinference semantic conventions: https://github.com/Arize-ai/openinference/blob/c80c81b8d6fa564598bd359cdd7313f4472ceca8/python/openinference-semantic-conventions/src/openinference/semconv/trace/__init__.py#L81
    "llm.token_count.completion",
}

TOTAL_TOKEN_ATTRIBUTE_KEYS = {
    # openllmetry semantic conventions: https://github.com/traceloop/openllmetry/blob/e66894fd7f8324bd7b2972d7f727da39e7d93181/packages/opentelemetry-semantic-conventions-ai/opentelemetry/semconv_ai/__init__.py#L91
    "llm.usage.total_tokens",
    # openinference semantic conventions: https://github.com/Arize-ai/openinference/blob/c80c81b8d6fa564598bd359cdd7313f4472ceca8/python/openinference-semantic-conventions/src/openinference/semconv/trace/__init__.py#L119
    "llm.token_count.total",
}

INPUTS_KEYS = {
    # openinference semantic conventions: https://github.com/Arize-ai/openinference/blob/c80c81b8d6fa564598bd359cdd7313f4472ceca8/python/openinference-semantic-conventions/src/openinference/semconv/trace/__init__.py#L12
    "input.value",
    # openllmetry semantic conventions: https://github.com/traceloop/openllmetry/blob/e66894fd7f8324bd7b2972d7f727da39e7d93181/packages/opentelemetry-semantic-conventions-ai/opentelemetry/semconv_ai/__init__.py#L138
    "traceloop.entity.input",
}

OUTPUTS_KEYS = {
    # openinference semantic conventions: https://github.com/Arize-ai/openinference/blob/c80c81b8d6fa564598bd359cdd7313f4472ceca8/python/openinference-semantic-conventions/src/openinference/semconv/trace/__init__.py#L6
    "output.value",
    # openllmetry semantic conventions: https://github.com/traceloop/openllmetry/blob/e66894fd7f8324bd7b2972d7f727da39e7d93181/packages/opentelemetry-semantic-conventions-ai/opentelemetry/semconv_ai/__init__.py#L139
    "traceloop.entity.output",
}


def translate_span_type_from_otel(attributes: dict[str, Any]) -> str | None:
    """
    Translate OTEL span kind attributes to MLflow span type.

    This function checks for OpenInference and Traceloop span kind attributes
    and maps them to MLflow span types.

    Args:
        attributes: Dictionary of span attributes

    Returns:
        MLflow span type string or None if no mapping found
    """
    # Check for OpenInference span kind
    if openinference_kind := attributes.get(OPENINFERENCE_SPAN_KIND_ATTRIBUTE_KEY):
        # Handle JSON-serialized values
        if isinstance(openinference_kind, str):
            try:
                openinference_kind = json.loads(openinference_kind)
            except (json.JSONDecodeError, TypeError):
                pass  # Use the string value as-is

        mlflow_type = OPENINFERENCE_SPAN_KIND_TO_MLFLOW_TYPE.get(openinference_kind)
        if mlflow_type is None:
            _logger.debug(
                f"OpenInference span kind '{openinference_kind}' is not supported "
                "by MLflow Span Type"
            )
        return mlflow_type

    # Check for Traceloop span kind
    if traceloop_kind := attributes.get(TRACELOOP_SPAN_KIND_ATTRIBUTE_KEY):
        # Handle JSON-serialized values
        if isinstance(traceloop_kind, str):
            try:
                traceloop_kind = json.loads(traceloop_kind)
            except (json.JSONDecodeError, TypeError):
                pass  # Use the string value as-is

        mlflow_type = TRACELOOP_SPAN_KIND_TO_MLFLOW_TYPE.get(traceloop_kind)
        if mlflow_type is None:
            _logger.debug(
                f"Traceloop span kind '{traceloop_kind}' is not supported by MLflow Span Type"
            )
        return mlflow_type


def translate_span_when_storing(span: Span) -> dict[str, Any]:
    """
    Apply attributes translations to a span's dictionary when storing.

    Supported translations:
    - Token usage attributes
    - Inputs and outputs attributes

    Args:
        span: Span object

    Returns:
        Translated span dictionary
    """

    span_dict = span.to_dict()
    attributes = span_dict.get("attributes", {})

    if span.parent_id is None:
        # update inputs and outputs for root span
        if SpanAttributeKey.INPUTS not in attributes:
            if input_value := next(
                (attributes.get(key) for key in INPUTS_KEYS if key in attributes), None
            ):
                attributes[SpanAttributeKey.INPUTS] = input_value
        if SpanAttributeKey.OUTPUTS not in attributes:
            if output_value := next(
                (attributes.get(key) for key in OUTPUTS_KEYS if key in attributes), None
            ):
                attributes[SpanAttributeKey.OUTPUTS] = output_value

    # translate token usage
    if SpanAttributeKey.CHAT_USAGE not in attributes:
        input_tokens = next(
            (attributes.get(key) for key in INPUT_TOKEN_ATTRIBUTE_KEYS if key in attributes), None
        )
        output_tokens = next(
            (attributes.get(key) for key in OUTPUT_TOKEN_ATTRIBUTE_KEYS if key in attributes), None
        )
        total_tokens = next(
            (attributes.get(key) for key in TOTAL_TOKEN_ATTRIBUTE_KEYS if key in attributes), None
        )
        if input_tokens and output_tokens and (total_tokens is None):
            total_tokens = int(input_tokens) + int(output_tokens)
        if total_tokens:
            usage_dict = {
                TokenUsageKey.INPUT_TOKENS: int(input_tokens),
                TokenUsageKey.OUTPUT_TOKENS: int(output_tokens),
                TokenUsageKey.TOTAL_TOKENS: int(total_tokens),
            }
            attributes[SpanAttributeKey.CHAT_USAGE] = dump_span_attribute_value(usage_dict)

    span_dict["attributes"] = attributes
    return span_dict


def translate_loaded_span(span_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Apply attributes translations to a span dictionary when loading.

    Supported translations:
    - OTEL span kind attributes

    Args:
        span_dict: Span dictionary (will be modified in-place)

    Returns:
        Modified span dictionary
    """
    attributes = span_dict.get("attributes", {})

    if SpanAttributeKey.SPAN_TYPE not in attributes:
        if mlflow_type := translate_span_type_from_otel(attributes):
            # Serialize to match how MLflow stores attributes
            attributes[SpanAttributeKey.SPAN_TYPE] = dump_span_attribute_value(mlflow_type)

    span_dict["attributes"] = attributes
    return span_dict
