import importlib.metadata as _meta
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

# AGNO SDK doesn't provide version parameter from 1.7.1 onwards. Hence we capture the
# latest version manually

try:
    import agno

    if not hasattr(agno, "__version__"):
        try:
            agno.__version__ = _meta.version("agno")
        except _meta.PackageNotFoundError:
            agno.__version__ = "1.7.7"
except ImportError:
    pass


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
        from agno.team import Team
        from agno.tools.function import FunctionCall
    except ImportError:
        return SpanType.UNKNOWN

    storage_types = ()

    try:
        # Agno version below 2 uses storage.base
        from agno.storage.base import Storage
    except ImportError:
        try:
            # Agno version 2+ uses db.base
            import agno.db.base as db_base
        except ImportError:
            pass
        else:
            candidates: list[type[Any]] = []
            for attr in ("BaseDb", "AsyncBaseDb"):
                cls = getattr(db_base, attr, None)
                if cls is not None:
                    candidates.append(cls)
            storage_types = tuple(candidates)
    else:
        storage_types = (Storage,)

    if isinstance(instance, (Agent, Team)):
        return SpanType.AGENT
    if isinstance(instance, FunctionCall):
        return SpanType.TOOL
    if storage_types and isinstance(instance, storage_types):
        return SpanType.MEMORY
    if isinstance(instance, Model):
        return SpanType.LLM
    return SpanType.UNKNOWN


# Agno version >=2 uses Metrics object, but version <2 uses dictionary
def _parse_usage(result) -> dict[str, int] | None:
    usage = getattr(result, "metrics", None) or getattr(result, "session_metrics", None)
    if not usage:
        return None

    def _get_value(container, key):
        if isinstance(container, dict):
            return container.get(key)
        return getattr(container, key, None)

    def _coerce(value):
        if value is None:
            return 0
        if isinstance(value, (list, tuple)):
            return sum(value)
        return int(value)

    return {
        TokenUsageKey.INPUT_TOKENS: _coerce(_get_value(usage, "input_tokens")),
        TokenUsageKey.OUTPUT_TOKENS: _coerce(_get_value(usage, "output_tokens")),
        TokenUsageKey.TOTAL_TOKENS: _coerce(_get_value(usage, "total_tokens")),
    }


# Agno version>=2 uses run.agent.RunOutput and run.workflow.TeamRunOutput,
# but version <2 uses run.response.RunResponse and run.team.TeamRunResponse
def _set_span_outputs(span: LiveSpan, result: Any) -> None:
    try:
        from agno.run.response import RunResponse
        from agno.run.team import TeamRunResponse

        response_types = (RunResponse, TeamRunResponse)
    except ImportError:
        from agno.run.agent import RunOutput
        from agno.run.workflow import TeamRunOutput

        response_types = (RunOutput, TeamRunOutput)

    if response_types and isinstance(result, response_types):
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
