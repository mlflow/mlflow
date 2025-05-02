import inspect
import logging

import mlflow
from mlflow.entities import SpanType
from mlflow.entities.span import LiveSpan
from mlflow.smolagents.chat import set_span_chat_attributes
from mlflow.utils.autologging_utils.config import AutoLoggingConfig

_logger = logging.getLogger(__name__)


def patched_class_call(original, self, *args, **kwargs):
    try:
        config = AutoLoggingConfig.init(flavor_name=mlflow.smolagents.FLAVOR_NAME)

        if config.log_traces:
            fullname = f"{self.__class__.__name__}.{original.__name__}"
            span_type = _get_span_type(self)
            with mlflow.start_span(name=fullname, span_type=span_type) as span:
                inputs = _construct_full_inputs(original, self, *args, **kwargs)
                span.set_inputs(inputs)
                _set_span_attributes(span=span, instance=self)
                result = original(self, *args, **kwargs)

                # Need to convert the response of smolagents API for better visualization
                outputs = result.__dict__ if hasattr(result, "__dict__") else result
                if span_type == SpanType.CHAT_MODEL:
                    set_span_chat_attributes(
                        span=span, messages=inputs.get("messages", []), output=outputs
                    )
                span.set_outputs(outputs)
                return result
    except Exception as e:
        _logger.error("the error occurred while patching")
        raise e


def _get_span_type(instance) -> str:
    from smolagents import CodeAgent, MultiStepAgent, Tool, ToolCallingAgent, models

    if isinstance(instance, (MultiStepAgent, CodeAgent, ToolCallingAgent)):
        return SpanType.AGENT
    elif isinstance(instance, Tool):
        return SpanType.TOOL
    elif isinstance(instance, models.Model):
        return SpanType.CHAT_MODEL

    return SpanType.UNKNOWN


def _construct_full_inputs(func, *args, **kwargs):
    signature = inspect.signature(func)
    # This does not create copy. So values should not be mutated directly
    arguments = signature.bind_partial(*args, **kwargs).arguments

    if "self" in arguments:
        arguments.pop("self")

    # Avoid non serializable objects and circular references
    return {
        k: v.__dict__ if hasattr(v, "__dict__") else v
        for k, v in arguments.items()
        if v is not None
    }


def _set_span_attributes(span: LiveSpan, instance):
    # Smolagents is available only python >= 3.10, so importing libraries inside methods.
    try:
        from smolagents import CodeAgent, MultiStepAgent, Tool, ToolCallingAgent, models

        if isinstance(instance, (MultiStepAgent, CodeAgent, ToolCallingAgent)):
            agent = _get_agent_attributes(instance)
            for key, value in agent.items():
                if value is not None:
                    span.set_attribute(key, str(value) if isinstance(value, list) else value)

        elif isinstance(instance, Tool):
            tool = _get_tool_attributes(instance)
            for key, value in tool.items():
                if value is not None:
                    span.set_attribute(key, str(value) if isinstance(value, list) else value)

        elif issubclass(type(instance), models.Model):
            model = _get_model_attributes(instance)
            for key, value in model.items():
                if value is not None:
                    span.set_attribute(key, str(value) if isinstance(value, list) else value)

    except Exception as e:
        _logger.warn("An exception happens when saving span attributes. Exception: %s", e)


def _get_agent_attributes(instance):
    agent = {}
    for key, value in instance.__dict__.items():
        if key == "tools":
            value = _parse_tools(value)
        if value is None:
            continue
        agent[key] = str(value)

    return agent


def _inner_get_tool_attributes(tool_dict):
    res = {}
    if hasattr(tool_dict, "name") and tool_dict.name is not None:
        res["name"] = tool_dict.name
    if hasattr(tool_dict, "description") and tool_dict.description is not None:
        res["description"] = tool_dict.description
    result = {}
    if res:
        result["type"] = "function"
        result["function"] = res
    return result


def _get_tool_attributes(instance):
    instance_dict = instance.__dict__
    return _inner_get_tool_attributes(instance_dict)


def _parse_tools(tools):
    result = []
    for tool in tools:
        res = _inner_get_tool_attributes(tool)
        result.append(res)
    return result


def _get_model_attributes(instance):
    model = {}
    for key, value in instance.__dict__.items():
        if value is None or key == "api_key":
            continue
        model[key] = str(value)
    return model
