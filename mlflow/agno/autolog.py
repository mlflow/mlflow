import inspect
import logging
from typing import Any

import mlflow
from mlflow.entities import SpanType
from mlflow.entities.span import LiveSpan
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from mlflow.utils.autologging_utils.config import AutoLoggingConfig

FLAVOR_NAME = "agno"
_logger = logging.getLogger(__name__)


def _construct_full_inputs(func, *args, **kwargs) -> dict[str, Any]:
    sig = inspect.signature(func)
    bound = sig.bind_partial(*args, **kwargs).arguments
    return {
        k: (v.__dict__ if hasattr(v, "__dict__") else v) for k, v in bound.items() if v is not None
    }


def _compute_span_name(instance, original) -> str:
    try:
        from agno.tools.function import FunctionCall

        if isinstance(instance, FunctionCall):
            tool_name = None
            for attr in ["function_name", "name", "tool_name"]:
                val = getattr(instance, attr, None)
                if val:
                    return val
            if not tool_name and hasattr(instance, "function"):
                underlying_fn = getattr(instance, "function")
                for attr in ["name", "__name__", "function_name"]:
                    val = getattr(underlying_fn, attr, None)
                    if val:
                        return val
            if not tool_name:
                return "AgnoToolCall"

    except ImportError:
        pass

    return f"{instance.__class__.__name__}.{original.__name__}"


def _parse_tools(tools) -> list[dict[str, Any]]:
    result = []
    for tool in tools or []:
        try:
            data = tool.model_dumps(exclude_none=True)
            if data:
                result.append({"type": "function", "function": data})
        except Exception:
            # Fallback to string representation
            result.append({"name": str(tool)})
    return result


def _get_agent_attributes(instance) -> dict[str, Any]:
    agent_attr: dict[str, Any] = {}
    for key, value in instance.__dict__.items():
        if key == "tools":
            value = _parse_tools(value)
        if value is not None:
            agent_attr[key] = value
    return agent_attr


def _get_tools_attribute(instance) -> dict[str, Any]:
    return {
        f"tool_{key}": val
        for key, val in vars(instance.function).items()
        if not key.startswith("_") and val is not None
    }


def _set_span_attributes(span: LiveSpan, instance) -> None:
    try:
        from agno.agent import Agent
        from agno.team import Team

        if isinstance(instance, (Agent, Team)):
            span.set_attributes(_get_agent_attributes(instance))
    except Exception as exc:  # pragma: no cover
        _logger.debug("Unable to attach agent attributes: %s", exc)

    try:
        from agno.tools.function import FunctionCall

        if isinstance(instance, FunctionCall):
            span.set_attributes(_get_tools_attribute(instance))
    except Exception as exc:  # pragma: no cover
        _logger.debug("Unable to attach agent attributes: %s", exc)


def _get_span_type(instance) -> str:
    try:
        from agno.agent import Agent
        from agno.storage.base import Storage
        from agno.team import Team
        from agno.tools.function import FunctionCall

    except ImportError:
        return SpanType.UNKNOWN
    if isinstance(instance, (Agent, Team)):
        return SpanType.AGENT
    if isinstance(instance, FunctionCall):
        return SpanType.TOOL
    if isinstance(instance, Storage):
        return SpanType.RETRIEVER

    return SpanType.UNKNOWN


# Token usage metrics is given as list of integers.
def _coerce_to_int(value) -> int | None:
    if value is None:
        return None

    if isinstance(value, list):
        total = 0
        for item in value:
            coerced = _coerce_to_int(item)  # recurse for nested dicts etc.
            if coerced is None:
                return None  # bail if anything isn't numeric(recursive loop returns None)
            total += coerced
        return total


def _parse_usage(result) -> dict[str, int] | None:
    usage = getattr(result, "metrics", None) or getattr(result, "session_metrics", None)
    if not usage:
        return None

    parsed = {
        TokenUsageKey.INPUT_TOKENS: _coerce_to_int(usage.get("input_tokens")),
        TokenUsageKey.OUTPUT_TOKENS: _coerce_to_int(usage.get("output_tokens")),
        TokenUsageKey.TOTAL_TOKENS: _coerce_to_int(usage.get("total_tokens")),
    }
    return parsed if all(v is not None for v in parsed.values()) else None


async def patched_async_class_call(original, self, *args, **kwargs):
    cfg = AutoLoggingConfig.init(flavor_name=FLAVOR_NAME)
    if not cfg.log_traces:
        return await original(self, *args, **kwargs)

    span_name = _compute_span_name(self, original)
    span_type = _get_span_type(self)

    async with mlflow.start_span(name=span_name, span_type=span_type) as span:
        span.set_inputs(_construct_full_inputs(original, self, *args, **kwargs))
        _set_span_attributes(span, self)

        try:
            result = await original(self, *args, **kwargs)
            span.set_outputs(result)
            if usage := _parse_usage(result):
                span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage)
            return result
        except Exception as e:
            span.record_exception(e)
            raise e


def patched_class_call(original, self, *args, **kwargs):
    cfg = AutoLoggingConfig.init(flavor_name=FLAVOR_NAME)
    if not cfg.log_traces:
        return original(self, *args, **kwargs)

    span_name = _compute_span_name(self, original)
    span_type = _get_span_type(self)

    with mlflow.start_span(name=span_name, span_type=span_type) as span:
        span.set_inputs(_construct_full_inputs(original, self, *args, **kwargs))
        _set_span_attributes(span, self)

        try:
            result = original(self, *args, **kwargs)
            span.set_outputs(result)
            if usage := _parse_usage(result):
                span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage)
            return result
        except Exception as e:
            span.record_exception(e)
            raise e
