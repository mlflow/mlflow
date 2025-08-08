import logging
from typing import Any

from pydantic import BaseModel

import mlflow
from mlflow.autogen.chat import log_tools
from mlflow.entities import SpanType
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from mlflow.tracing.utils import construct_full_inputs
from mlflow.utils.autologging_utils import (
    autologging_integration,
    get_autologging_config,
    safe_patch,
)

_logger = logging.getLogger(__name__)
FLAVOR_NAME = "autogen"


@autologging_integration(FLAVOR_NAME)
def autolog(
    log_traces: bool = True,
    disable: bool = False,
    silent: bool = False,
):
    """
    Enables (or disables) and configures autologging for AutoGen flavor.
    Due to its patch design, this method needs to be called after importing AutoGen classes.

    Args:
        log_traces: If ``True``, traces are logged for AutoGen models.
            If ``False``, no traces are collected during inference. Default to ``True``.
        disable: If ``True``, disables the AutoGen autologging. Default to ``False``.
        silent: If ``True``, suppress all event logs and warnings from MLflow during AutoGen
            autologging. If ``False``, show all events and warnings.

    Example:

    .. code-block:: python
        :caption: Example

        import mlflow
        from autogen_agentchat.agents import AssistantAgent
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        mlflow.autogen.autolog()
        agent = AssistantAgent("assistant", OpenAIChatCompletionClient(model="gpt-4o-mini"))
        result = await agent.run(task="Say 'Hello World!'")
        print(result)
    """
    from autogen_agentchat.agents import BaseChatAgent
    from autogen_core.models import ChatCompletionClient

    async def patched_completion(original, self, *args, **kwargs):
        if not get_autologging_config(FLAVOR_NAME, "log_traces"):
            return await original(self, *args, **kwargs)
        else:
            name = f"{self.__class__.__name__}.{original.__name__}"
            with mlflow.start_span(name, span_type=SpanType.LLM) as span:
                inputs = construct_full_inputs(original, self, *args, **kwargs)
                span.set_inputs(
                    {key: _convert_value_to_dict(value) for key, value in inputs.items()}
                )
                span.set_attribute(SpanAttributeKey.MESSAGE_FORMAT, "autogen")

                if tools := inputs.get("tools"):
                    log_tools(span, tools)

                outputs = await original(self, *args, **kwargs)

                if usage := _parse_usage(outputs):
                    span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage)

                span.set_outputs(_convert_value_to_dict(outputs))

                return outputs

    async def patched_agent(original, self, *args, **kwargs):
        if not get_autologging_config(FLAVOR_NAME, "log_traces"):
            return await original(self, *args, **kwargs)
        else:
            agent_name = getattr(self, "name", self.__class__.__name__)
            name = f"{agent_name}.{original.__name__}"
            with mlflow.start_span(name, span_type=SpanType.AGENT) as span:
                inputs = construct_full_inputs(original, self, *args, **kwargs)
                span.set_inputs(
                    {key: _convert_value_to_dict(value) for key, value in inputs.items()}
                )

                if tools := getattr(self, "_tools", None):
                    log_tools(span, tools)

                outputs = await original(self, *args, **kwargs)

                span.set_outputs(_convert_value_to_dict(outputs))

                return outputs

    for cls in BaseChatAgent.__subclasses__():
        safe_patch(FLAVOR_NAME, cls, "run", patched_agent)
        safe_patch(FLAVOR_NAME, cls, "on_messages", patched_agent)

    for cls in _get_all_subclasses(ChatCompletionClient):
        safe_patch(FLAVOR_NAME, cls, "create", patched_completion)


def _convert_value_to_dict(value):
    # BaseChatMessage does not contain content and type attributes
    return value.model_dump(serialize_as_any=True) if isinstance(value, BaseModel) else value


def _get_all_subclasses(cls):
    """Get all subclasses recursively"""
    all_subclasses = []

    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(_get_all_subclasses(subclass))

    return all_subclasses


def _parse_usage(output: Any) -> dict[str, int] | None:
    try:
        usage = getattr(output, "usage", None)
        if usage:
            return {
                TokenUsageKey.INPUT_TOKENS: usage.prompt_tokens,
                TokenUsageKey.OUTPUT_TOKENS: usage.completion_tokens,
                TokenUsageKey.TOTAL_TOKENS: usage.prompt_tokens + usage.completion_tokens,
            }
    except Exception as e:
        _logger.debug(f"Failed to parse token usage from output: {e}")
    return None
