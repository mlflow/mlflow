import logging

import mlflow
from mlflow.autogen.chat import (
    convert_assistant_message_to_chat_message,
    log_chat_messages,
    log_tools,
)
from mlflow.entities import SpanType
from mlflow.tracing.utils import construct_full_inputs, set_span_chat_messages
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import (
    autologging_integration,
    get_autologging_config,
    safe_patch,
)

FLAVOR_NAME = "autogen"

_logger = logging.getLogger(__name__)


@experimental
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

    async def patched_run(original, self, *args, **kwargs):
        if not get_autologging_config(FLAVOR_NAME, "log_traces"):
            return await original(self, *args, **kwargs)
        else:
            span_type = SpanType.AGENT if isinstance(self, BaseChatAgent) else SpanType.LLM
            with mlflow.start_span(original.__name__, span_type=span_type) as span:
                inputs = construct_full_inputs(original, self, *args, **kwargs)
                span.set_inputs(inputs)

                if tools := getattr(self, "_tools", None):
                    log_tools(span, tools)

                if isinstance(self, ChatCompletionClient) and (messages := inputs.get("messages")):
                    log_chat_messages(span, messages)

                outputs = await original(self, *args, **kwargs)

                if isinstance(self, ChatCompletionClient) and (
                    content := getattr(outputs, "content", None)
                ):
                    if chat_message := convert_assistant_message_to_chat_message(content):
                        set_span_chat_messages(span, [chat_message], append=True)

                span.set_outputs(outputs)

                return outputs

    for cls in BaseChatAgent.__subclasses__():
        safe_patch(FLAVOR_NAME, cls, "run", patched_run)
        safe_patch(FLAVOR_NAME, cls, "on_messages", patched_run)

    for cls in _get_all_subclasses(ChatCompletionClient):
        safe_patch(FLAVOR_NAME, cls, "create", patched_run)


def _get_all_subclasses(cls):
    """Get all subclasses recursively"""
    all_subclasses = []

    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(_get_all_subclasses(subclass))

    return all_subclasses
