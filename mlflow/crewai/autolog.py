import inspect
import json
import logging
import warnings
from contextlib import contextmanager, nullcontext
from typing import Any

from packaging.version import Version

import mlflow
from mlflow.entities import SpanType
from mlflow.entities.span import LiveSpan
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from mlflow.tracing.utils import TraceJSONEncoder
from mlflow.utils.autologging_utils.config import AutoLoggingConfig

_logger = logging.getLogger(__name__)


def patched_standalone_call(original, *args, **kwargs):
    config = AutoLoggingConfig.init(flavor_name=mlflow.crewai.FLAVOR_NAME)

    if not config.log_traces:
        return original(*args, **kwargs)

    fullname, span_type = _resolve_standalone_span(original, kwargs)
    if fullname is None or span_type is None:
        _logger.debug(f"Could not resolve span name or type for {original}")
        return original(*args, **kwargs)

    with mlflow.start_span(name=fullname, span_type=span_type) as span:
        inputs = _construct_full_inputs(original, *args, **kwargs)
        span.set_inputs(inputs)

        result = original(*args, **kwargs)

        # Need to convert the response of generate_content for better visualization
        outputs = result.__dict__ if hasattr(result, "__dict__") else result
        span.set_outputs(outputs)

        return result


def patched_class_call(original, self, *args, **kwargs):
    config = AutoLoggingConfig.init(flavor_name=mlflow.crewai.FLAVOR_NAME)

    if not config.log_traces:
        return original(self, *args, **kwargs)

    default_name = f"{self.__class__.__name__}.{original.__name__}"
    fullname = _get_span_name(self) or default_name
    span_type = _get_span_type(self)
    with mlflow.start_span(name=fullname, span_type=span_type) as span:
        inputs = _construct_full_inputs(original, self, *args, **kwargs)
        span.set_inputs(inputs)
        _set_span_attributes(span=span, instance=self)

        # CrewAI reports only crew-level usage totals.
        # This patch hooks LiteLLM's `completion` to capture each response
        # so per-call LLM usage can be logged.
        capture_context = (
            _capture_llm_response(self) if span_type == SpanType.LLM else nullcontext()
        )
        with capture_context:
            result = original(self, *args, **kwargs)

        # Need to convert the response of generate_content for better visualization
        outputs = result.__dict__ if hasattr(result, "__dict__") else result

        if span_type == SpanType.LLM and (usage_dict := _parse_usage(self)):
            span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage_dict)
        span.set_outputs(outputs)

        return result


def _capture_llm_response(instance):
    @contextmanager
    def _patched_completion():
        import litellm

        original_completion = litellm.completion

        def _capture_completion(*args, **kwargs):
            response = original_completion(*args, **kwargs)
            setattr(instance, "_mlflow_last_response", response)
            return response

        litellm.completion = _capture_completion
        try:
            yield
        finally:
            litellm.completion = original_completion

    return _patched_completion()


def _parse_usage(instance: Any) -> dict[str, int] | None:
    usage = instance.__dict__.get("_mlflow_last_response", {}).get("usage", {})
    if not usage:
        return None

    return {
        TokenUsageKey.INPUT_TOKENS: usage.prompt_tokens,
        TokenUsageKey.OUTPUT_TOKENS: usage.completion_tokens,
        TokenUsageKey.TOTAL_TOKENS: usage.total_tokens,
    }


def _resolve_standalone_span(original, kwargs) -> tuple[str, SpanType]:
    name = original.__name__
    if name == "execute_tool_and_check_finality":
        # default_tool_name should not be hit in normal runs; may append if crewai bugs
        default_tool_name = "ToolExecution"
        fullname = kwargs["agent_action"].tool if "agent_action" in kwargs else None
        fullname = fullname or default_tool_name
        return fullname, SpanType.TOOL

    return None, None


def _get_span_type(instance) -> str:
    import crewai
    from crewai import LLM, Agent, Crew, Task
    from crewai.flow.flow import Flow

    try:
        if isinstance(instance, (Flow, Crew, Task)):
            return SpanType.CHAIN
        elif isinstance(instance, Agent):
            return SpanType.AGENT
        elif isinstance(instance, LLM):
            return SpanType.LLM
        elif isinstance(instance, Flow):
            return SpanType.CHAIN
        elif isinstance(
            instance, crewai.agents.agent_builder.base_agent_executor_mixin.CrewAgentExecutorMixin
        ):
            return SpanType.MEMORY

        CREWAI_VERSION = Version(crewai.__version__)
        # Knowledge and Memory are not available before 0.83.0
        if CREWAI_VERSION >= Version("0.83.0"):
            memory_classes = (
                crewai.memory.ShortTermMemory,
                crewai.memory.LongTermMemory,
                crewai.memory.EntityMemory,
            )
            # UserMemory was removed in 0.157.0:
            # https://github.com/crewAIInc/crewAI/pull/3225
            if CREWAI_VERSION < Version("0.157.0"):
                memory_classes = (*memory_classes, crewai.memory.UserMemory)

            if isinstance(instance, memory_classes):
                return SpanType.MEMORY

            if isinstance(instance, crewai.Knowledge):
                return SpanType.RETRIEVER
    except AttributeError as e:
        _logger.warn("An exception happens when resolving the span type. Exception: %s", e)

    return SpanType.UNKNOWN


def _get_span_name(instance) -> str | None:
    try:
        from crewai import LLM, Agent, Crew, Task

        if isinstance(instance, Crew):
            default_name = Crew.model_fields["name"].default
            return instance.name if instance.name != default_name else None
        elif isinstance(instance, Task):
            return instance.name
        elif isinstance(instance, Agent):
            return instance.role
        elif isinstance(instance, LLM):
            return instance.model

    except AttributeError as e:
        _logger.debug("An exception happens when resolving the span name. Exception: %s", e)

    return None


def _is_serializable(value):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # There is type mismatch in some crewai class, suppress warning here
            json.dumps(value, cls=TraceJSONEncoder, ensure_ascii=False)
        return True
    except (TypeError, ValueError):
        return False


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
        if v is not None and _is_serializable(v)
    }


def _set_span_attributes(span: LiveSpan, instance):
    # Crewai is available only python >=3.10, so importing libraries inside methods.
    try:
        import crewai
        from crewai import LLM, Agent, Crew, Task
        from crewai.flow.flow import Flow

        ## Memory class does not have helpful attributes
        if isinstance(instance, Crew):
            for key, value in instance.__dict__.items():
                if value is not None:
                    if key == "tasks":
                        value = _parse_tasks(value)
                    elif key == "agents":
                        value = _parse_agents(value)
                    elif key == "embedder":
                        value = _sanitize_value(value)
                    span.set_attribute(key, str(value) if isinstance(value, list) else value)

        elif isinstance(instance, Agent):
            agent = _get_agent_attributes(instance)
            for key, value in agent.items():
                if value is not None:
                    span.set_attribute(key, str(value) if isinstance(value, list) else value)

        elif isinstance(instance, Task):
            task = _get_task_attributes(instance)
            for key, value in task.items():
                if value is not None:
                    span.set_attribute(key, str(value) if isinstance(value, list) else value)

        elif isinstance(instance, LLM):
            llm = _get_llm_attributes(instance)
            for key, value in llm.items():
                if value is not None:
                    span.set_attribute(key, str(value) if isinstance(value, list) else value)

        elif isinstance(instance, Flow):
            for key, value in instance.__dict__.items():
                if value is not None:
                    span.set_attribute(key, str(value) if isinstance(value, list) else value)

        elif Version(crewai.__version__) >= Version("0.83.0"):
            if isinstance(instance, crewai.Knowledge):
                for key, value in instance.__dict__.items():
                    if value is not None and key != "storage":
                        span.set_attribute(key, str(value) if isinstance(value, list) else value)

    except AttributeError as e:
        _logger.warn("An exception happens when saving span attributes. Exception: %s", e)


def _get_agent_attributes(instance):
    agent = {}
    for key, value in instance.__dict__.items():
        if key == "tools":
            value = _parse_tools(value)
        elif key == "embedder":
            value = _sanitize_value(value)
        if value is None:
            continue
        agent[key] = str(value)

    return agent


def _get_task_attributes(instance):
    task = {}
    for key, value in instance.__dict__.items():
        if value is None:
            continue
        if key == "tools":
            value = _parse_tools(value)
            task[key] = value
        elif key == "agent":
            task[key] = value.role
        else:
            task[key] = str(value)
    return task


def _get_llm_attributes(instance):
    llm = {SpanAttributeKey.MESSAGE_FORMAT: "crewai"}
    for key, value in instance.__dict__.items():
        if value is None:
            continue
        elif key in ["callbacks", "api_key"]:
            # Skip callbacks until how they should be logged are decided
            continue
        else:
            llm[key] = str(value)
    return llm


def _parse_agents(agents):
    attributes = []
    for agent in agents:
        model = None
        if agent.llm is not None:
            if hasattr(agent.llm, "model"):
                model = agent.llm.model
            elif hasattr(agent.llm, "model_name"):
                model = agent.llm.model_name
        attributes.append(
            {
                "id": str(agent.id),
                "role": agent.role,
                "goal": agent.goal,
                "backstory": agent.backstory,
                "cache": agent.cache,
                "config": agent.config,
                "verbose": agent.verbose,
                "allow_delegation": agent.allow_delegation,
                "tools": agent.tools,
                "max_iter": agent.max_iter,
                "llm": str(model if model is not None else ""),
            }
        )
    return attributes


def _parse_tasks(tasks):
    return [
        {
            "agent": task.agent.role,
            "description": task.description,
            "async_execution": task.async_execution,
            "expected_output": task.expected_output,
            "human_input": task.human_input,
            "tools": task.tools,
            "output_file": task.output_file,
        }
        for task in tasks
    ]


def _parse_tools(tools):
    result = []
    for tool in tools:
        res = {}
        if hasattr(tool, "name") and tool.name is not None:
            res["name"] = tool.name
        if hasattr(tool, "description") and tool.description is not None:
            res["description"] = tool.description
        if res:
            result.append(
                {
                    "type": "function",
                    "function": res,
                }
            )
    return result


def _sanitize_value(val):
    """
    Sanitize a value to remove sensitive information.

    Args:
        val: The value to sanitize. Can be None, a dict, a list, or other types.

    Returns:
        The sanitized value.
    """
    if val is None:
        return None

    sensitive_keys = ["api_key", "secret", "password", "token"]

    if isinstance(val, dict):
        sanitized = {}
        for k, v in val.items():
            if any(sensitive in k.lower() for sensitive in sensitive_keys):
                continue
            sanitized[k] = _sanitize_value(v)
        return sanitized

    elif isinstance(val, list):
        return [_sanitize_value(item) for item in val]

    return val
