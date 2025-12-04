import inspect
import logging
from dataclasses import asdict
from typing import Any

import mlflow
from mlflow.entities import SpanType
from mlflow.entities.span import LiveSpan
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from mlflow.utils.autologging_utils.config import AutoLoggingConfig

_logger = logging.getLogger(__name__)

# Allowlists for safe attributes to extract from pydantic_ai objects.
# Using allowlists instead of denylists to avoid capturing client/provider
# references that can interfere with async cleanup (e.g., httpx client lifecycle).
_AGENT_SAFE_ATTRIBUTES = frozenset(
    {
        "name",
        "system_prompt",
        "retries",
        "result_type",
        "output_type",
        "deps_type",
        "end_strategy",
        "defer_model_check",
        "instrument",
    }
)

_MODEL_SAFE_ATTRIBUTES = frozenset(
    {
        "model_name",
        "name",
        "system",
    }
)

_TOOL_SAFE_ATTRIBUTES = frozenset(
    {
        "name",
        "description",
        "max_retries",
        "prepare",
    }
)

_MCP_SERVER_SAFE_ATTRIBUTES = frozenset(
    {
        "name",
        "url",
    }
)

_SAFE_ATTRIBUTE_TYPES = (str, int, float, bool, type(None), list, tuple)


def _is_safe_for_serialization(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, _SAFE_ATTRIBUTE_TYPES):
        return True
    if isinstance(value, dict):
        return all(_is_safe_for_serialization(v) for v in value.values())
    if hasattr(value, "__dataclass_fields__"):
        return True
    if isinstance(value, type):
        return True
    return False


def _safe_get_attribute(instance: Any, key: str) -> Any:
    try:
        value = getattr(instance, key, None)
        if value is None:
            return None
        if isinstance(value, type):
            return value.__name__
        if _is_safe_for_serialization(value):
            return value
        return None
    except Exception:
        return None


def _set_span_attributes(span: LiveSpan, instance):
    # 1) MCPServer attributes
    try:
        from pydantic_ai.mcp import MCPServer

        if isinstance(instance, MCPServer):
            mcp_attrs = _get_mcp_server_attributes(instance)
            span.set_attributes({k: v for k, v in mcp_attrs.items() if v is not None})
    except Exception as e:
        _logger.warning("Failed saving MCPServer attributes: %s", e)

    # 2) Agent attributes
    try:
        from pydantic_ai import Agent

        if isinstance(instance, Agent):
            agent_attrs = _get_agent_attributes(instance)
            span.set_attributes({k: v for k, v in agent_attrs.items() if v is not None})
    except Exception as e:
        _logger.warning("Failed saving Agent attributes: %s", e)

    # 3) InstrumentedModel attributes
    try:
        from pydantic_ai.models.instrumented import InstrumentedModel

        if isinstance(instance, InstrumentedModel):
            model_attrs = _get_model_attributes(instance)
            span.set_attributes({k: v for k, v in model_attrs.items() if v is not None})
    except Exception as e:
        _logger.warning("Failed saving InstrumentedModel attributes: %s", e)

    # 4) Tool attributes
    try:
        from pydantic_ai import Tool

        if isinstance(instance, Tool):
            tool_attrs = _get_tool_attributes(instance)
            span.set_attributes({k: v for k, v in tool_attrs.items() if v is not None})
    except Exception as e:
        _logger.warning("Failed saving Tool attributes: %s", e)


async def patched_async_class_call(original, self, *args, **kwargs):
    cfg = AutoLoggingConfig.init(flavor_name=mlflow.pydantic_ai.FLAVOR_NAME)
    if not cfg.log_traces:
        return await original(self, *args, **kwargs)

    fullname = f"{self.__class__.__name__}.{original.__name__}"
    span_type = _get_span_type(self)

    with mlflow.start_span(name=fullname, span_type=span_type) as span:
        inputs = _construct_full_inputs(original, self, *args, **kwargs)
        span.set_inputs(inputs)
        _set_span_attributes(span, self)

        result = await original(self, *args, **kwargs)
        outputs = _serialize_output(result)
        span.set_outputs(outputs)
        if usage_dict := _parse_usage(result):
            span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage_dict)
        return result


def patched_class_call(original, self, *args, **kwargs):
    cfg = AutoLoggingConfig.init(flavor_name=mlflow.pydantic_ai.FLAVOR_NAME)
    if not cfg.log_traces:
        return original(self, *args, **kwargs)

    fullname = f"{self.__class__.__name__}.{original.__name__}"
    span_type = _get_span_type(self)
    with mlflow.start_span(name=fullname, span_type=span_type) as span:
        inputs = _construct_full_inputs(original, self, *args, **kwargs)
        span.set_inputs(inputs)
        _set_span_attributes(span, self)

        result = original(self, *args, **kwargs)
        outputs = _serialize_output(result)
        span.set_outputs(outputs)
        if usage_dict := _parse_usage(result):
            span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage_dict)
        return result


def _get_span_type(instance) -> str:
    try:
        from pydantic_ai import Agent, Tool
        from pydantic_ai.mcp import MCPServer
        from pydantic_ai.models.instrumented import InstrumentedModel
    except ImportError:
        return SpanType.UNKNOWN

    if isinstance(instance, InstrumentedModel):
        return SpanType.LLM
    if isinstance(instance, Agent):
        return SpanType.AGENT
    if isinstance(instance, Tool):
        return SpanType.TOOL
    if isinstance(instance, MCPServer):
        return SpanType.TOOL

    try:
        from pydantic_ai._tool_manager import ToolManager

        if isinstance(instance, ToolManager):
            return SpanType.TOOL
    except ImportError:
        pass

    return SpanType.UNKNOWN


def _construct_full_inputs(func, *args, **kwargs) -> dict[str, Any]:
    sig = inspect.signature(func)
    bound = sig.bind_partial(*args, **kwargs).arguments
    bound.pop("self", None)
    bound.pop("deps", None)

    return {
        k: (v.__dict__ if hasattr(v, "__dict__") else v) for k, v in bound.items() if v is not None
    }


def _serialize_output(result: Any) -> Any:
    if result is None:
        return None

    if hasattr(result, "new_messages") and callable(result.new_messages):
        try:
            new_messages = result.new_messages()
            serialized_messages = [asdict(msg) for msg in new_messages]
            serialized_result = asdict(result)
            serialized_result["_new_messages_serialized"] = serialized_messages
            return serialized_result
        except Exception as e:
            _logger.debug(f"Failed to serialize new_messages: {e}")

    return result.__dict__ if hasattr(result, "__dict__") else result


def _get_agent_attributes(instance):
    attrs = {SpanAttributeKey.MESSAGE_FORMAT: "pydantic_ai"}
    for key in _AGENT_SAFE_ATTRIBUTES:
        value = _safe_get_attribute(instance, key)
        if value is not None:
            attrs[key] = value
    if hasattr(instance, "tools"):
        try:
            if tools_value := _parse_tools(instance.tools):
                attrs["tools"] = tools_value
        except Exception:
            pass
    return attrs


def _get_model_attributes(instance):
    attrs = {SpanAttributeKey.MESSAGE_FORMAT: "pydantic_ai"}
    for key in _MODEL_SAFE_ATTRIBUTES:
        value = _safe_get_attribute(instance, key)
        if value is not None:
            attrs[key] = value
    return attrs


def _get_tool_attributes(instance):
    attrs = {}
    for key in _TOOL_SAFE_ATTRIBUTES:
        value = _safe_get_attribute(instance, key)
        if value is not None:
            attrs[key] = value
    return attrs


def _get_mcp_server_attributes(instance):
    attrs = {}
    for key in _MCP_SERVER_SAFE_ATTRIBUTES:
        value = _safe_get_attribute(instance, key)
        if value is not None:
            attrs[key] = value
    if hasattr(instance, "tools"):
        try:
            if tools_value := _parse_tools(instance.tools):
                attrs["tools"] = tools_value
        except Exception:
            pass
    return attrs


def _parse_tools(tools):
    return [
        {"type": "function", "function": data}
        for tool in tools
        if (data := tool.model_dumps(exclude_none=True))
    ]


def _parse_usage(result: Any) -> dict[str, int] | None:
    try:
        if isinstance(result, tuple) and len(result) == 2:
            usage = result[1]
        else:
            usage = getattr(result, "usage", None)

        return {
            TokenUsageKey.INPUT_TOKENS: usage.request_tokens,
            TokenUsageKey.OUTPUT_TOKENS: usage.response_tokens,
            TokenUsageKey.TOTAL_TOKENS: usage.total_tokens,
        }
    except Exception as e:
        _logger.debug(f"Failed to parse token usage from output: {e}")
    return None
