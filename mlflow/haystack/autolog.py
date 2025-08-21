"""Haystack autolog implementation following smolagents and crewai patterns."""

import inspect
import logging
from typing import Any, Optional

import mlflow
from mlflow.entities import SpanType
from mlflow.entities.span import LiveSpan, SpanAttributeKey
from mlflow.tracing.constant import TokenUsageKey
from mlflow.utils.autologging_utils.config import AutoLoggingConfig

_logger = logging.getLogger(__name__)


def patched_class_call(original, self, *args, **kwargs):
    config = AutoLoggingConfig.init(flavor_name=mlflow.haystack.FLAVOR_NAME)

    if not config.log_traces:
        return original(self, *args, **kwargs)

    fullname = f"{self.__class__.__name__}.{original.__name__}"
    span_type = _get_span_type(self)

    with mlflow.start_span(name=fullname, span_type=span_type) as span:
        inputs = _construct_full_inputs(original, self, *args, **kwargs)
        span.set_inputs(inputs)
        _set_span_attributes(span, self)

        result = original(self, *args, **kwargs)

        outputs = result.__dict__ if hasattr(result, "__dict__") else result
        if isinstance(outputs, dict):
            outputs = _format_outputs(outputs)

        span.set_outputs(outputs)

        if token_usage := _parse_token_usage(outputs):
            span.set_attribute(SpanAttributeKey.CHAT_USAGE, token_usage)

        if model := _extract_model_info(outputs):
            span.set_attribute("model", model)

        return result


async def patched_async_class_call(original, self, *args, **kwargs):
    """Async patch method for Haystack async methods."""
    config = AutoLoggingConfig.init(flavor_name=mlflow.haystack.FLAVOR_NAME)

    if not config.log_traces:
        return await original(self, *args, **kwargs)

    fullname = f"{self.__class__.__name__}.{original.__name__}"
    span_type = _get_span_type(self)

    with mlflow.start_span(name=fullname, span_type=span_type) as span:
        inputs = _construct_full_inputs(original, self, *args, **kwargs)
        span.set_inputs(inputs)
        _set_span_attributes(span, self)

        result = await original(self, *args, **kwargs)

        outputs = result.__dict__ if hasattr(result, "__dict__") else result
        if isinstance(outputs, dict):
            outputs = _format_outputs(outputs)

        span.set_outputs(outputs)

        if token_usage := _parse_token_usage(outputs):
            span.set_attribute(SpanAttributeKey.CHAT_USAGE, token_usage)

        if model := _extract_model_info(outputs):
            span.set_attribute("model", model)

        return result


def _get_span_type(instance: Any) -> str:
    """Determine the span type based on the instance type."""
    if instance is None:
        return SpanType.TOOL

    class_name = instance.__class__.__name__.lower()

    # Define span type mappings
    span_type_mapping = {
        "pipeline": SpanType.CHAIN,
        "asyncpipeline": SpanType.CHAIN,
        "agent": SpanType.AGENT,
        "generator": SpanType.CHAT_MODEL,
        "llm": SpanType.CHAT_MODEL,
        "chat": SpanType.CHAT_MODEL,
        "retriever": SpanType.RETRIEVER,
        "search": SpanType.RETRIEVER,
        "embed": SpanType.EMBEDDING,
        "toolinvoker": SpanType.TOOL,
    }

    # Check exact matches first
    if class_name in span_type_mapping:
        return span_type_mapping[class_name]

    # Check partial matches
    for key, span_type in span_type_mapping.items():
        if key in class_name:
            return span_type

    return SpanType.TOOL


def _construct_full_inputs(func, *args, **kwargs) -> dict[str, Any]:
    """Construct inputs for haystack components following smolagents/crewai pattern."""
    signature = inspect.signature(func)
    arguments = signature.bind_partial(*args, **kwargs).arguments

    if "self" in arguments:
        arguments.pop("self")

    # Special handling for Pipeline.run - extract meaningful inputs
    if "data" in arguments and isinstance(arguments.get("data"), dict):
        data = arguments["data"]
        # Extract question/query fields from any component
        for component_data in data.values():
            if isinstance(component_data, dict):
                for key in ["question", "query", "prompt", "text"]:
                    if key in component_data:
                        return {key: component_data[key]}
        return {}

    return {
        k: v.__dict__ if hasattr(v, "__dict__") else v
        for k, v in arguments.items()
        if v is not None
    }


def _format_outputs(outputs: Any) -> Any:
    """Format outputs for tracing."""
    if not isinstance(outputs, dict):
        return outputs

    formatted_outputs = {}
    for key, value in outputs.items():
        if isinstance(value, dict) and "replies" in value:
            formatted_component = {}
            replies = value["replies"]

            if isinstance(replies, list) and len(replies) == 1:
                formatted_component["replies"] = replies[0]
            else:
                formatted_component["replies"] = replies

            if meta := value.get("meta"):
                formatted_component["meta"] = meta

            for field_key, field_value in value.items():
                if field_key not in ["replies", "meta"]:
                    formatted_component[field_key] = field_value

            formatted_outputs[key] = formatted_component
        else:
            formatted_outputs[key] = value

    return formatted_outputs


def _set_span_attributes(span: LiveSpan, instance):
    """Set attributes on the span based on the instance type."""
    # Always set message format
    span.set_attribute(SpanAttributeKey.MESSAGE_FORMAT, "haystack")

    try:
        if hasattr(instance, "graph"):  # Pipeline
            attributes = _get_pipeline_attributes(instance)
        else:  # Component
            attributes = _get_component_attributes(instance)

        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, str(value) if isinstance(value, (list, dict)) else value)
    except Exception as e:
        _logger.debug(f"Failed to set span attributes: {e}")


def _get_pipeline_attributes(pipeline) -> dict[str, Any]:
    """Extract attributes for a Pipeline instance."""
    attributes = {}

    if hasattr(pipeline, "graph") and hasattr(pipeline.graph, "nodes"):
        nodes = pipeline.graph.nodes
        if nodes:
            attributes["components"] = list(nodes.keys())
            attributes["component_count"] = len(nodes)

    return attributes


def _get_component_attributes(instance) -> dict[str, Any]:
    """Extract attributes for a component instance."""
    attributes = {"type": instance.__class__.__name__}

    if hasattr(instance, "_init_parameters"):
        for key, value in instance._init_parameters.items():
            if key.lower() not in ["api_key", "token"]:
                attributes[key] = str(value) if value is not None else None

    if hasattr(instance, "__haystack_input__"):
        inputs = _extract_socket_names(instance.__haystack_input__)
        if inputs:
            attributes["input_types"] = inputs

    if hasattr(instance, "__haystack_output__"):
        outputs = _extract_socket_names(instance.__haystack_output__)
        if outputs:
            attributes["output_types"] = outputs

    return attributes


def _extract_socket_names(sockets) -> list[str]:
    """Extract socket names from a Haystack Sockets object."""
    try:
        if hasattr(sockets, "_sockets"):
            return list(sockets._sockets.keys())
        elif hasattr(sockets, "__dict__"):
            return [k for k in sockets.__dict__.keys() if not k.startswith("_")]
    except Exception:
        pass
    return []


def _parse_token_usage(outputs: dict[str, Any]) -> Optional[dict[str, int]]:
    """Parse token usage from outputs."""
    if not isinstance(outputs, dict):
        return None

    try:
        # Check for meta information in outputs
        meta = outputs.get("meta", {})
        if isinstance(meta, list) and meta:
            meta = meta[0]

        if isinstance(meta, dict) and "usage" in meta:
            usage = meta["usage"]
            if isinstance(usage, dict):
                return {
                    TokenUsageKey.INPUT_TOKENS: usage.get("prompt_tokens", 0),
                    TokenUsageKey.OUTPUT_TOKENS: usage.get("completion_tokens", 0),
                    TokenUsageKey.TOTAL_TOKENS: usage.get("total_tokens", 0),
                }
    except Exception as e:
        _logger.debug(f"Failed to parse token usage: {e}")

    return None


def _extract_model_info(outputs: dict[str, Any]) -> Optional[str]:
    """Extract model information from outputs."""
    if not isinstance(outputs, dict):
        return None

    try:
        meta = outputs.get("meta", {})
        if isinstance(meta, list) and meta:
            meta = meta[0]

        if isinstance(meta, dict) and "model" in meta:
            return meta["model"]
    except Exception as e:
        _logger.debug(f"Failed to extract model info: {e}")

    return None
