"""
GenAI Schema Mapping Module for MLflow Tracing.

This module provides utilities to translate MLflow span attributes to OpenTelemetry
GenAI semantic conventions. This enables interoperability with OTEL-compatible
observability tools and follows the standardized GenAI semantic conventions.

Reference: https://opentelemetry.io/docs/specs/semconv/gen-ai/
"""

import json
import logging
from typing import Any

_logger = logging.getLogger(__name__)


# Mapping from MLflow internal span attribute keys to OpenTelemetry GenAI semantic convention keys.
# Reference: https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/
GENAI_MAPPING: dict[str, str] = {
    # Input/Output attributes
    "mlflow.spanInputs": "gen_ai.request.input",
    "mlflow.spanOutputs": "gen_ai.response.output",
    "mlflow.traceInputs": "gen_ai.request.input",
    "mlflow.traceOutputs": "gen_ai.response.output",
    # Span type mapping
    "mlflow.spanType": "gen_ai.operation.name",
    # Token usage attributes
    # MLflow stores token usage in a nested dict, but we also support flat keys
    "mlflow.chat.tokenUsage.input_tokens": "gen_ai.usage.input_tokens",
    "mlflow.chat.tokenUsage.output_tokens": "gen_ai.usage.output_tokens",
    "mlflow.chat.tokenUsage.total_tokens": "gen_ai.usage.total_tokens",
    # Model and provider attributes
    "mlflow.model.name": "gen_ai.request.model",
    "mlflow.model.provider": "gen_ai.system",
    "mlflow.modelId": "gen_ai.request.model",
    # Chat-specific attributes
    "mlflow.chat.tools": "gen_ai.request.tools",
    "mlflow.message.format": "gen_ai.request.format",
    # Temperature and other model parameters
    "mlflow.model.temperature": "gen_ai.request.temperature",
    "mlflow.model.maxTokens": "gen_ai.request.max_tokens",
    "mlflow.model.topP": "gen_ai.request.top_p",
    "mlflow.model.topK": "gen_ai.request.top_k",
    "mlflow.model.stopSequences": "gen_ai.request.stop_sequences",
    "mlflow.model.frequencyPenalty": "gen_ai.request.frequency_penalty",
    "mlflow.model.presencePenalty": "gen_ai.request.presence_penalty",
    # Response attributes
    "mlflow.response.finishReason": "gen_ai.response.finish_reasons",
    "mlflow.response.id": "gen_ai.response.id",
    # Function/tool calling
    "mlflow.spanFunctionName": "gen_ai.tool.name",
    # Trace metadata
    "mlflow.traceRequestId": "gen_ai.request.id",
    "mlflow.experimentId": "mlflow.experiment_id",  # Preserve MLflow-specific
    # Session and user context
    "mlflow.trace.user": "gen_ai.user.id",
    "mlflow.trace.session": "gen_ai.session.id",
}

# Mapping from MLflow span types to GenAI operation names
MLFLOW_SPAN_TYPE_TO_GENAI_OPERATION: dict[str, str] = {
    "LLM": "text_completion",
    "CHAT_MODEL": "chat",
    "EMBEDDING": "embeddings",
    "RETRIEVER": "retrieval",
    "TOOL": "execute_tool",
    "AGENT": "invoke_agent",
    "CHAIN": "chain",
    "RERANKER": "rerank",
    "PARSER": "parse",
    "GUARDRAIL": "guardrail",
    "EVALUATOR": "evaluate",
    "UNKNOWN": "unknown",
}

# Token usage keys in the nested mlflow.chat.tokenUsage attribute
TOKEN_USAGE_KEY_MAPPING: dict[str, str] = {
    "input_tokens": "gen_ai.usage.input_tokens",
    "output_tokens": "gen_ai.usage.output_tokens",
    "total_tokens": "gen_ai.usage.total_tokens",
}


def convert_to_genai_schema(attrs: dict[str, Any]) -> dict[str, Any]:
    """
    Convert MLflow span attributes to OpenTelemetry GenAI semantic conventions.

    This function takes a dictionary of span attributes with MLflow keys and returns
    a new dictionary with keys remapped to GenAI conventions. Unmapped attributes
    are preserved, and nested attributes (like token usage) are handled appropriately.

    Args:
        attrs: Dictionary of span attributes with MLflow keys.

    Returns:
        New dictionary with keys remapped to GenAI semantic conventions.
        Unmapped attributes are preserved with their original keys.

    Example:
        >>> attrs = {
        ...     "mlflow.spanInputs": "Hello, world!",
        ...     "mlflow.spanOutputs": "Hi there!",
        ...     "mlflow.spanType": "LLM",
        ...     "mlflow.chat.tokenUsage": {"input_tokens": 10, "output_tokens": 5},
        ...     "custom.attribute": "preserved",
        ... }
        >>> result = convert_to_genai_schema(attrs)
        >>> result["gen_ai.request.input"]
        'Hello, world!'
        >>> result["gen_ai.usage.input_tokens"]
        10
        >>> result["custom.attribute"]
        'preserved'
    """
    if not attrs:
        return {}

    result: dict[str, Any] = {}

    for key, value in attrs.items():
        # Handle nested token usage attribute specially
        if key == "mlflow.chat.tokenUsage":
            _convert_token_usage(value, result)
            continue

        # Handle span type conversion
        if key == "mlflow.spanType":
            genai_key = GENAI_MAPPING.get(key, key)
            converted_value = _convert_span_type(value)
            result[genai_key] = converted_value
            continue

        # Check for direct mapping
        if key in GENAI_MAPPING:
            genai_key = GENAI_MAPPING[key]
            result[genai_key] = _process_value(value)
        else:
            # Preserve unmapped attributes
            result[key] = _process_value(value)

    return result


def _convert_token_usage(token_usage: Any, result: dict[str, Any]) -> None:
    """
    Convert MLflow token usage attribute to GenAI convention attributes.

    The MLflow token usage is stored as a nested dict or JSON string:
    {"input_tokens": int, "output_tokens": int, "total_tokens": int}

    This function extracts these values and maps them to flat GenAI keys.

    Args:
        token_usage: Token usage value (dict or JSON string).
        result: Result dictionary to update with converted token usage.
    """
    if token_usage is None:
        return

    # Parse JSON string if necessary
    if isinstance(token_usage, str):
        try:
            token_usage = json.loads(token_usage)
        except (json.JSONDecodeError, TypeError):
            _logger.debug(f"Failed to parse token usage JSON: {token_usage}")
            return

    if not isinstance(token_usage, dict):
        _logger.debug(f"Token usage is not a dict: {type(token_usage)}")
        return

    # Map each token usage key to GenAI convention
    for mlflow_key, genai_key in TOKEN_USAGE_KEY_MAPPING.items():
        if mlflow_key in token_usage:
            result[genai_key] = token_usage[mlflow_key]


def _convert_span_type(span_type: Any) -> str:
    """
    Convert MLflow span type to GenAI operation name.

    Args:
        span_type: MLflow span type value (string or JSON-encoded string).

    Returns:
        GenAI operation name string.
    """
    # Parse JSON string if necessary
    if isinstance(span_type, str):
        try:
            parsed = json.loads(span_type)
            if isinstance(parsed, str):
                span_type = parsed
        except (json.JSONDecodeError, TypeError):
            pass  # Use the string value as-is

    # Map to GenAI operation name
    return MLFLOW_SPAN_TYPE_TO_GENAI_OPERATION.get(span_type, span_type)


def _process_value(value: Any) -> Any:
    """
    Process a value for inclusion in the result dictionary.

    Handles JSON-encoded strings by parsing them if they represent
    complex types (dicts, lists).

    Args:
        value: The value to process.

    Returns:
        Processed value (parsed JSON if applicable, otherwise original).
    """
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            # Return parsed value for complex types
            if isinstance(parsed, (dict, list)):
                return parsed
            # For primitive types wrapped in JSON, return as-is to preserve format
            return value
        except (json.JSONDecodeError, TypeError):
            pass
    return value


# Reverse mapping from GenAI keys to preferred MLflow keys.
# When multiple MLflow keys map to the same GenAI key, we prefer the span-level keys.
GENAI_TO_MLFLOW_MAPPING: dict[str, str] = {
    "gen_ai.request.input": "mlflow.spanInputs",
    "gen_ai.response.output": "mlflow.spanOutputs",
    "gen_ai.operation.name": "mlflow.spanType",
    "gen_ai.request.model": "mlflow.model.name",
    "gen_ai.system": "mlflow.model.provider",
    "gen_ai.request.tools": "mlflow.chat.tools",
    "gen_ai.request.format": "mlflow.message.format",
    "gen_ai.request.temperature": "mlflow.model.temperature",
    "gen_ai.request.max_tokens": "mlflow.model.maxTokens",
    "gen_ai.request.top_p": "mlflow.model.topP",
    "gen_ai.request.top_k": "mlflow.model.topK",
    "gen_ai.request.stop_sequences": "mlflow.model.stopSequences",
    "gen_ai.request.frequency_penalty": "mlflow.model.frequencyPenalty",
    "gen_ai.request.presence_penalty": "mlflow.model.presencePenalty",
    "gen_ai.response.finish_reasons": "mlflow.response.finishReason",
    "gen_ai.response.id": "mlflow.response.id",
    "gen_ai.tool.name": "mlflow.spanFunctionName",
    "gen_ai.request.id": "mlflow.traceRequestId",
    "mlflow.experiment_id": "mlflow.experimentId",
    "gen_ai.user.id": "mlflow.trace.user",
    "gen_ai.session.id": "mlflow.trace.session",
}


def convert_from_genai_schema(attrs: dict[str, Any]) -> dict[str, Any]:
    """
    Convert OpenTelemetry GenAI semantic convention attributes to MLflow format.

    This is the reverse operation of convert_to_genai_schema, useful for
    importing traces from OTEL-compatible sources.

    Args:
        attrs: Dictionary of span attributes with GenAI convention keys.

    Returns:
        New dictionary with keys remapped to MLflow format.
        Unmapped attributes are preserved with their original keys.

    Example:
        >>> attrs = {
        ...     "gen_ai.request.input": "Hello!",
        ...     "gen_ai.response.output": "Hi!",
        ...     "gen_ai.usage.input_tokens": 10,
        ...     "custom.attribute": "preserved",
        ... }
        >>> result = convert_from_genai_schema(attrs)
        >>> result["mlflow.spanInputs"]
        'Hello!'
    """
    if not attrs:
        return {}

    # Reverse token usage mapping for flat keys
    reverse_token_mapping = {v: k for k, v in TOKEN_USAGE_KEY_MAPPING.items()}

    result: dict[str, Any] = {}
    token_usage: dict[str, int] = {}

    for key, value in attrs.items():
        # Check if this is a token usage key
        if key in reverse_token_mapping:
            mlflow_key = reverse_token_mapping[key]
            token_usage[mlflow_key] = value
            continue

        # Check for direct reverse mapping using the explicit reverse mapping
        if key in GENAI_TO_MLFLOW_MAPPING:
            mlflow_key = GENAI_TO_MLFLOW_MAPPING[key]
            result[mlflow_key] = value
        else:
            # Preserve unmapped attributes
            result[key] = value

    # Add aggregated token usage if present
    if token_usage:
        result["mlflow.chat.tokenUsage"] = token_usage

    return result


def get_genai_attribute_keys() -> list[str]:
    """
    Get a list of all GenAI semantic convention attribute keys.

    Returns:
        List of GenAI attribute key strings.
    """
    return list(set(GENAI_MAPPING.values()) | set(TOKEN_USAGE_KEY_MAPPING.values()))


def get_mlflow_attribute_keys() -> list[str]:
    """
    Get a list of all MLflow attribute keys that can be mapped.

    Returns:
        List of MLflow attribute key strings.
    """
    return list(GENAI_MAPPING.keys())
