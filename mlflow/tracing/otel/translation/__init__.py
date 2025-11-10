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
from mlflow.tracing.utils import dump_span_attribute_value

_logger = logging.getLogger(__name__)

_TRANSLATORS: list[OtelSchemaTranslator] = [
    OpenInferenceTranslator(),
    GenAiTranslator(),
    TraceloopTranslator(),
    GoogleADKTranslator(),
]


def translate_span_when_storing(span: Span) -> dict[str, Any]:
    """
    Apply attributes translations to a span's dictionary when storing.

    Supported translations:
    - Token usage attributes from various OTEL schemas
    - Inputs and outputs attributes from various OTEL schemas

    These attributes translation need to happen when storing spans because we need
    to update TraceInfo accordingly.

    Args:
        span: Span object

    Returns:
        Translated span dictionary
    """
    span_dict = span.to_dict()
    attributes = span_dict.get("attributes", {})

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
