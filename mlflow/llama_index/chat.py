from llama_index.core.base.llms.types import ChatMessage as LLamaChatMessage
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.events.llm import (
    LLMChatEndEvent,
    LLMChatStartEvent,
    LLMCompletionEndEvent,
    LLMCompletionStartEvent,
)

# llama-index includes llama-index-llms-openai in its requirements
# https://github.com/run-llama/llama_index/blob/663e1700f58c2414e549b9f5005abe87a275dd77/pyproject.toml#L52
from llama_index.llms.openai.utils import to_openai_message_dict

from mlflow.types.chat import ChatMessage
from mlflow.utils.pydantic_utils import model_dump_compat


def get_chat_messages_from_event(event: BaseEvent) -> list[ChatMessage]:
    """
    Extract chat messages from the LlamaIndex callback event.
    """
    if isinstance(event, LLMCompletionStartEvent):
        return [ChatMessage(role="user", content=event.prompt)]
    elif isinstance(event, LLMCompletionEndEvent):
        return [ChatMessage(role="assistant", content=event.response.text)]
    elif isinstance(event, LLMChatStartEvent):
        return [_convert_message_to_mlflow_chat(msg) for msg in event.messages]
    elif isinstance(event, LLMChatEndEvent):
        message = event.response.message
        return [_convert_message_to_mlflow_chat(message)]

    raise ValueError(f"Unsupported event type for chat attribute extraction: {type(event)}")


def _convert_message_to_mlflow_chat(message: LLamaChatMessage) -> ChatMessage:
    """Convert a message object from LlamaIndex to MLflow's standard format."""
    message = to_openai_message_dict(message, drop_none=False)

    # tool calls are pydantic models in llama-index
    if tool_calls := message.get("tool_calls"):
        message["tool_calls"] = [model_dump_compat(tool) for tool in tool_calls]

    return ChatMessage.validate_compat(message)
