"""
Autologging logic for Agno V1 using MLflow's tracing API.
"""

import logging
from typing import Any

import mlflow
from mlflow.entities import SpanType
from mlflow.entities.span import LiveSpan
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from mlflow.tracing.utils import construct_full_inputs
from mlflow.utils.autologging_utils.config import AutoLoggingConfig

FLAVOR_NAME = "agno"
_logger = logging.getLogger(__name__)


def _compute_span_name(instance, original) -> str:
    try:
        from agno.tools.function import FunctionCall

        if isinstance(instance, FunctionCall):
            tool_name = None
            for attr in ["function_name", "name", "tool_name"]:
                if val := getattr(instance, attr, None):
                    return val
            if not tool_name and hasattr(instance, "function"):
                underlying_fn = getattr(instance, "function")
                for attr in ["name", "__name__", "function_name"]:
                    if val := getattr(underlying_fn, attr, None):
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
            if data := tool.model_dumps(exclude_none=True):
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
        key: val
        for key, val in vars(instance.function).items()
        if not key.startswith("_") and val is not None
    }


def _set_span_inputs_attributes(span: LiveSpan, instance: Any, raw_inputs: dict[str, Any]) -> None:
    try:
        from agno.agent import Agent
        from agno.team import Team

        if isinstance(instance, (Agent, Team)):
            span.set_attributes(_get_agent_attributes(instance))
            # Filter out None values from inputs because Agent/Team's
            # run method has so many optional arguments.
            span.set_inputs({k: v for k, v in raw_inputs.items() if v is not None})
            return
    except Exception as exc:  # pragma: no cover
        _logger.debug("Unable to attach agent attributes: %s", exc)

    try:
        from agno.tools.function import FunctionCall

        if isinstance(instance, FunctionCall):
            span.set_inputs(instance.arguments)
            if tool_data := _get_tools_attribute(instance):
                span.set_attributes(tool_data)
            return
    except Exception as exc:  # pragma: no cover
        _logger.debug("Unable to set function attrcalling inputs and attributes: %s", exc)

    try:
        from agno.models.message import Message

        if (
            (messages := raw_inputs.get("messages"))
            and isinstance(messages, list)
            and all(isinstance(m, Message) for m in messages)
        ):
            raw_inputs["messages"] = [m.to_dict() for m in messages]
            span.set_inputs(raw_inputs)
            return
    except Exception as exc:  # pragma: no cover
        _logger.debug("Unable to parse input message: %s", exc)

    span.set_inputs(raw_inputs)


def _get_span_type(instance) -> str:
    try:
        from agno.agent import Agent
        from agno.models.base import Model
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
        return SpanType.MEMORY
    if isinstance(instance, Model):
        return SpanType.LLM
    return SpanType.UNKNOWN


def _parse_usage(result) -> dict[str, int] | None:
    usage = getattr(result, "metrics", None) or getattr(result, "session_metrics", None)
    if not usage:
        return None

    return {
        TokenUsageKey.INPUT_TOKENS: sum(usage.get("input_tokens")),
        TokenUsageKey.OUTPUT_TOKENS: sum(usage.get("output_tokens")),
        TokenUsageKey.TOTAL_TOKENS: sum(usage.get("total_tokens")),
    }


def _set_span_outputs(span: LiveSpan, result: Any) -> None:
    from agno.run.response import RunResponse
    from agno.run.team import TeamRunResponse

    if isinstance(result, (RunResponse, TeamRunResponse)):
        span.set_outputs(result.to_dict())
    else:
        span.set_outputs(result)

    if usage := _parse_usage(result):
        span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage)


async def patched_async_class_call(original, self, *args, **kwargs):
    cfg = AutoLoggingConfig.init(flavor_name=FLAVOR_NAME)
    if not cfg.log_traces:
        return await original(self, *args, **kwargs)

    span_name = _compute_span_name(self, original)
    span_type = _get_span_type(self)

    with mlflow.start_span(name=span_name, span_type=span_type) as span:
        raw_inputs = construct_full_inputs(original, self, *args, **kwargs)
        _set_span_inputs_attributes(span, self, raw_inputs)

        result = await original(self, *args, **kwargs)

        _set_span_outputs(span, result)
        return result


def patched_class_call(original, self, *args, **kwargs):
    cfg = AutoLoggingConfig.init(flavor_name=FLAVOR_NAME)
    if not cfg.log_traces:
        return original(self, *args, **kwargs)

    span_name = _compute_span_name(self, original)
    span_type = _get_span_type(self)

    with mlflow.start_span(name=span_name, span_type=span_type) as span:
        raw_inputs = construct_full_inputs(original, self, *args, **kwargs)
        _set_span_inputs_attributes(span, self, raw_inputs)

        result = original(self, *args, **kwargs)

        _set_span_outputs(span, result)
        return result
