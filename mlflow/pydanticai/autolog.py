import inspect
import json
import logging
import warnings
from typing import Any

import mlflow
from mlflow.entities import SpanType
from mlflow.entities.span import LiveSpan
from mlflow.tracing.utils import TraceJSONEncoder
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import autologging_integration, safe_patch
from mlflow.utils.autologging_utils.config import AutoLoggingConfig

_logger = logging.getLogger(__name__)

FLAVOR_NAME = "pydanticai"


def _set_span_attributes(span: LiveSpan, instance):
    try:
        import pydantic_ai
        from pydantic_ai import Agent
        from pydantic_ai.mcp import MCPServer
        from pydantic_ai.models.instrumented import InstrumentedModel

        if isinstance(instance, MCPServer):
            for key, value in instance.__dict__.items():
                if value is not None:
                    if key == "tools":
                        value = _parse_tools(value)
                    span.set_attribute(key, str(value) if isinstance(value, list) else value)

        elif isinstance(instance, Agent):
            agent = _get_agent_attributes(instance)
            for key, value in agent.items():
                if value is not None:
                    span.set_attribute(key, str(value) if isinstance(value, list) else value)

        elif isinstance(instance, InstrumentedModel):
            model = _get_model_attributes(instance)
            for key, value in model.items():
                if value is not None:
                    span.set_attribute(key, str(value) if isinstance(value, list) else value)

        elif hasattr(pydantic_ai, "Tool") and isinstance(instance, pydantic_ai.Tool):
            tool = _get_tool_attributes(instance)
            for key, value in tool.items():
                if value is not None:
                    span.set_attribute(key, str(value) if isinstance(value, list) else value)

    except AttributeError as e:
        _logger.warn("An exception happens when saving span attributes. Exception: %s", e)


async def patched_async_class_call(original, self, *args, **kwargs):
    cfg = AutoLoggingConfig.init(flavor_name=FLAVOR_NAME)
    if not cfg.log_traces:
        return await original(self, *args, **kwargs)

    fullname = f"{self.__class__.__name__}.{original.__name__}"
    span_type = _get_span_type(self)
    span_cm = mlflow.start_span(name=fullname, span_type=span_type)
    span = span_cm.__enter__()
    try:
        inputs = _construct_full_inputs(original, self, *args, **kwargs)
        span.set_inputs(inputs)
        _set_span_attributes(span, self)

        result = await original(self, *args, **kwargs)

        outputs = result.__dict__ if hasattr(result, "__dict__") else result
        span.set_outputs(outputs)
        return result
    finally:
        try:
            span_cm.__exit__(None, None, None)
        except Exception:
            _logger.debug("Failed to exit span context cleanly", exc_info=True)


def patched_class_call(original, self, *args, **kwargs):
    cfg = AutoLoggingConfig.init(flavor_name=FLAVOR_NAME)
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
        return SpanType.CHAIN
    if isinstance(instance, Tool):
        return SpanType.TOOL
    if isinstance(instance, MCPServer):
        return SpanType.AGENT
    return SpanType.UNKNOWN


def _is_serializable(value: Any) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            json.dumps(value, cls=TraceJSONEncoder, ensure_ascii=False)
        return True
    except (TypeError, ValueError):
        return False


def _construct_full_inputs(func, *args, **kwargs) -> dict[str, Any]:
    sig = inspect.signature(func)
    bound = sig.bind_partial(*args, **kwargs).arguments
    bound.pop("self", None)

    return {
        k: (v.__dict__ if hasattr(v, "__dict__") else v)
        for k, v in bound.items()
        if v is not None and _is_serializable(v)
    }


def _get_agent_attributes(instance):
    agent = {}
    for key, value in instance.__dict__.items():
        if key == "tools":
            value = _parse_tools(value)
        if value is None:
            continue
        agent[key] = str(value)

    return agent


def _get_model_attributes(instance):
    model = {}
    for key, value in instance.__dict__.items():
        if value is None:
            continue
        elif key in ["callbacks", "api_key"]:
            # Skip sensitive information
            continue
        else:
            model[key] = str(value)
    return model


def _get_tool_attributes(instance):
    tool = {}
    for key, value in instance.__dict__.items():
        if value is None:
            continue
        tool[key] = str(value)
    return tool


def _parse_tools(tools):
    result = []
    for tool in tools:
        res = {}
        if hasattr(tool, "name") and tool.name is not None:
            res["name"] = tool.name
        if hasattr(tool, "description") and tool.description is not None:
            res["description"] = tool.description
        if hasattr(tool, "parameters") and tool.parameters is not None:
            res["parameters"] = tool.parameters
        if res:
            result.append(
                {
                    "type": "function",
                    "function": res,
                }
            )
    return result


def list_tools(instance):
    if hasattr(instance, "tools"):
        return _parse_tools(instance.tools)
    return []


def call_tool(instance, tool_name, **kwargs):
    if not hasattr(instance, "tools"):
        raise AttributeError(f"{instance.__class__.__name__} does not have tools")

    for tool in instance.tools:
        if tool.name == tool_name:
            return tool(**kwargs)

    raise ValueError(f"Tool '{tool_name}' not found")


@experimental
@autologging_integration(FLAVOR_NAME)
def autolog(log_traces: bool = True, disable: bool = False, silent: bool = False):
    """
    Enable (or disable) autologging for PydanticAI.

    Args:
        log_traces: If True, capture spans for agent + model calls.
        disable:   If True, disable the autologging patches.
        silent:    If True, suppress MLflow warnings/info.
    """
    class_map = {
        "pydantic_ai.Agent": ["run", "run_sync", "run_stream"],
        "pydantic_ai.models.instrumented.InstrumentedModel": ["request", "request_stream"],
        "pydantic_ai.Tool": ["run"],
        "pydantic_ai.mcp.MCPServer": ["call_tool", "list_tools"],
    }

    try:
        for cls_path, methods in class_map.items():
            module_name, class_name = cls_path.rsplit(".", 1)
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            for method in methods:
                orig = getattr(cls, method)
                wrapper = (
                    patched_async_class_call
                    if inspect.iscoroutinefunction(orig)
                    else patched_class_call
                )
                safe_patch(
                    FLAVOR_NAME,
                    cls,
                    method,
                    wrapper,
                )
    except (ImportError, AttributeError) as e:
        _logger.error("Error patching PydanticAI autolog: %s", e)
