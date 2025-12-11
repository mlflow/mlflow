"""
Utilities for translating OTEL span attributes to MLflow span format.

This module provides functions to translate span attributes from various
OTEL semantic conventions (OpenInference, Traceloop, GenAI) to MLflow span types.
It uses modular translator classes for each OTEL schema for better organization
and performance.
"""

import json
import logging
from typing import Any

from mlflow.entities.span import Span
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from mlflow.tracing.otel.translation.base import OtelSchemaTranslator
from mlflow.tracing.otel.translation.genai_semconv import GenAiTranslator
from mlflow.tracing.otel.translation.google_adk import GoogleADKTranslator
from mlflow.tracing.otel.translation.open_inference import OpenInferenceTranslator
from mlflow.tracing.otel.translation.traceloop import TraceloopTranslator
from mlflow.tracing.otel.translation.vercel_ai import VercelAITranslator
from mlflow.tracing.otel.translation.voltagent import VoltAgentTranslator
from mlflow.tracing.utils import dump_span_attribute_value

_logger = logging.getLogger(__name__)

_TRANSLATORS: list[OtelSchemaTranslator] = [
    OpenInferenceTranslator(),
    GenAiTranslator(),
    TraceloopTranslator(),
    GoogleADKTranslator(),
    VercelAITranslator(),
    VoltAgentTranslator(),
]


def translate_span_when_storing(span: Span) -> dict[str, Any]:
    """
    Apply attributes translations to a span's dictionary when storing.

    Supported translations:
    - Token usage attributes from various OTEL schemas
    - Inputs and outputs attributes from various OTEL schemas
    - Message format for chat UI rendering

    These attributes translation need to happen when storing spans because we need
    to update TraceInfo accordingly.

    Args:
        span: Span object

    Returns:
        Translated span dictionary
    """
    span_dict = span.to_dict()
    attributes = sanitize_attributes(span_dict.get("attributes", {}))

    # Translate inputs and outputs
    if SpanAttributeKey.INPUTS not in attributes and (input_value := _get_input_value(attributes)):
        attributes[SpanAttributeKey.INPUTS] = input_value

    if SpanAttributeKey.OUTPUTS not in attributes and (
        output_value := _get_output_value(attributes)
    ):
        attributes[SpanAttributeKey.OUTPUTS] = output_value

    # Translate token usage
    if SpanAttributeKey.CHAT_USAGE not in attributes and (
        token_usage := _get_token_usage(attributes)
    ):
        attributes[SpanAttributeKey.CHAT_USAGE] = dump_span_attribute_value(token_usage)

    # Set message format for chat UI rendering
    if SpanAttributeKey.MESSAGE_FORMAT not in attributes and (
        message_format := _get_message_format(attributes)
    ):
        attributes[SpanAttributeKey.MESSAGE_FORMAT] = dump_span_attribute_value(message_format)

    span_dict["attributes"] = attributes
    return span_dict


def _get_token_usage(attributes: dict[str, Any]) -> dict[str, Any]:
    """
    Get token usage from various OTEL semantic conventions.
    """
    for translator in _TRANSLATORS:
        input_tokens = translator.get_input_tokens(attributes)
        output_tokens = translator.get_output_tokens(attributes)
        total_tokens = translator.get_total_tokens(attributes)

        # Calculate total tokens if not provided but input/output are available
        if input_tokens and output_tokens and (total_tokens is None):
            total_tokens = int(input_tokens) + int(output_tokens)

        if input_tokens and output_tokens and total_tokens:
            return {
                TokenUsageKey.INPUT_TOKENS: int(input_tokens),
                TokenUsageKey.OUTPUT_TOKENS: int(output_tokens),
                TokenUsageKey.TOTAL_TOKENS: int(total_tokens),
            }


def _get_input_value(attributes: dict[str, Any]) -> Any:
    """
    Get input value from various OTEL semantic conventions.

    Args:
        attributes: Dictionary of span attributes

    Returns:
        Input value or None if not found
    """
    for translator in _TRANSLATORS:
        if value := translator.get_input_value(attributes):
            return value


def _get_output_value(attributes: dict[str, Any]) -> Any:
    """
    Get output value from various OTEL semantic conventions.

    Args:
        attributes: Dictionary of span attributes

    Returns:
        Output value or None if not found
    """
    for translator in _TRANSLATORS:
        if value := translator.get_output_value(attributes):
            return value


def _get_message_format(attributes: dict[str, Any]) -> str | None:
    """
    Get message format from span attributes for chat UI rendering.

    Args:
        attributes: Dictionary of span attributes

    Returns:
        Message format string or None if not found
    """
    for translator in _TRANSLATORS:
        if message_format := translator.get_message_format(attributes):
            return message_format
    return None


def translate_span_type_from_otel(attributes: dict[str, Any]) -> str | None:
    """
    Translate OTEL span kind attributes to MLflow span type.

    This function checks for span kind attributes from various OTEL semantic
    conventions (OpenInference, Traceloop) and maps them to MLflow span types.

    Args:
        attributes: Dictionary of span attributes

    Returns:
        MLflow span type string or None if no mapping found
    """
    for translator in _TRANSLATORS:
        if mlflow_type := translator.translate_span_type(attributes):
            return mlflow_type


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

    try:
        if SpanAttributeKey.SPAN_TYPE not in attributes:
            if mlflow_type := translate_span_type_from_otel(attributes):
                # Serialize to match how MLflow stores attributes
                attributes[SpanAttributeKey.SPAN_TYPE] = dump_span_attribute_value(mlflow_type)
    except Exception:
        _logger.debug("Failed to translate span type", exc_info=True)

    span_dict["attributes"] = attributes
    return span_dict


def update_token_usage(
    current_token_usage: str | dict[str, Any], new_token_usage: str | dict[str, Any]
) -> str | dict[str, Any]:
    """
    Update current token usage in-place by adding the new token usage.

    Args:
        current_token_usage: Current token usage, dictionary or JSON string
        new_token_usage: New token usage, dictionary or JSON string

    Returns:
        Updated token usage dictionary or JSON string
    """
    try:
        if isinstance(current_token_usage, str):
            current_token_usage = json.loads(current_token_usage) or {}
        if isinstance(new_token_usage, str):
            new_token_usage = json.loads(new_token_usage) or {}
        if new_token_usage:
            for key in [
                TokenUsageKey.INPUT_TOKENS,
                TokenUsageKey.OUTPUT_TOKENS,
                TokenUsageKey.TOTAL_TOKENS,
            ]:
                current_token_usage[key] = current_token_usage.get(key, 0) + new_token_usage.get(
                    key, 0
                )
    except Exception:
        _logger.debug(
            f"Failed to update token usage with current_token_usage: {current_token_usage}, "
            f"new_token_usage: {new_token_usage}",
            exc_info=True,
        )

    return current_token_usage


def sanitize_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    """
    Sanitize attributes by removing duplicate dumped attributes.
    This is necessary because when spans are logged to sql store with otel_api, the
    span attributes are dumped twice (once in Span.from_otel_proto and once in span.to_dict).
    """
    updated_attributes = {}
    for key, value in attributes.items():
        try:
            result = json.loads(value)
            if isinstance(result, str):
                try:
                    # If the original value is a string or dict, we store it as
                    # a JSON-encoded string.  For other types, we store the original value directly.
                    # For string type, this is to avoid interpreting "1" as an int accidentally.
                    # For dictionary, we save the json-encoded-once string so that the UI can render
                    # it correctly after loading.
                    if isinstance(json.loads(result), (str, dict)):
                        updated_attributes[key] = result
                        continue
                except json.JSONDecodeError:
                    pass
        except (json.JSONDecodeError, TypeError):
            pass
        # if the value is not a json string, or it's only dumped once, we keep the original value
        updated_attributes[key] = value
    return updated_attributes
