import inspect
import json
import logging
import warnings

from packaging.version import Version

import mlflow
from mlflow.entities import SpanType
from mlflow.entities.span import LiveSpan
from mlflow.tracing.utils import TraceJSONEncoder
from mlflow.utils.autologging_utils.config import AutoLoggingConfig

_logger = logging.getLogger(__name__)


def patched_class_call(original, self, *args, **kwargs):
    config = AutoLoggingConfig.init(flavor_name=mlflow.gemini.FLAVOR_NAME)

    if config.log_traces:
        fullname = f"{self.__class__.__name__}.{original.__name__}"
        with mlflow.start_span(
            name=fullname,
            span_type=_get_span_type(self),
        ) as span:
            inputs = _construct_full_inputs(original, self, *args, **kwargs)
            span.set_inputs(inputs)
            _set_span_attributes(span=span, instance=self)
            result = original(self, *args, **kwargs)
            # Need to convert the response of generate_content for better visualization
            outputs = result.__dict__ if hasattr(result, "__dict__") else result
            span.set_outputs(outputs)

            return result


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
            return SpanType.RETRIEVER

        # Knowledge and Memory are not available before 0.83.0
        if Version(crewai.__version__) >= Version("0.83.0"):
            if isinstance(
                instance,
                (
                    crewai.memory.ShortTermMemory,
                    crewai.memory.LongTermMemory,
                    crewai.memory.UserMemory,
                    crewai.memory.EntityMemory,
                    crewai.Knowledge,
                ),
            ):
                return SpanType.RETRIEVER
    except AttributeError as e:
        _logger.warn("An exception happens when resolving the span type. Exception: %s", e)

    return SpanType.UNKNOWN


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
    llm = {}
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
