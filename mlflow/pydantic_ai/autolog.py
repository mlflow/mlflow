import inspect
import logging
from typing import Any

import mlflow
from mlflow.entities import SpanType
from mlflow.entities.span import LiveSpan
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from mlflow.utils.autologging_utils.config import AutoLoggingConfig

_logger = logging.getLogger(__name__)


def _set_span_attributes(span: LiveSpan, instance):
    # 1) MCPServer attributes
    try:
        from pydantic_ai.mcp import MCPServer

        if isinstance(instance, MCPServer):
            for key, value in instance.__dict__.items():
                if value is None:
                    continue
                if key == "tools":
                    value = _parse_tools(value)
                span.set_attribute(key, value)
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
        outputs = result.__dict__ if hasattr(result, "__dict__") else result
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
        outputs = result.__dict__ if hasattr(result, "__dict__") else result
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


def _get_agent_attributes(instance):
    agent = {}
    for key, value in instance.__dict__.items():
        if key == "tools":
            value = _parse_tools(value)
        if value is None:
            continue
        agent[key] = value

    return agent


def _get_model_attributes(instance):
    model = {SpanAttributeKey.MESSAGE_FORMAT: "pydantic_ai"}
    for key, value in instance.__dict__.items():
        if value is None:
            continue
        elif key in ["callbacks", "api_key"]:
            # Skip sensitive information
            continue
        else:
            model[key] = value
    return model


def _get_tool_attributes(instance):
    tool = {}
    for key, value in instance.__dict__.items():
        if value is None:
            continue
        tool[key] = value
    return tool


def _parse_tools(tools):
    result = []
    for tool in tools:
        data = tool.model_dumps(exclude_none=True)

        if data:
            result.append(
                {
                    "type": "function",
                    "function": data,
                }
            )
    return result


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
