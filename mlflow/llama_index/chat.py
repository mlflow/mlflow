import logging
from typing import Union

from llama_index.core.base.llms.types import ChatMessage as LLamaChatMessage
from llama_index.core.base.llms.types import ImageBlock, TextBlock
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.events.llm import (
    LLMChatEndEvent,
    LLMChatStartEvent,
    LLMCompletionEndEvent,
    LLMCompletionStartEvent,
)

from mlflow.types.chat import ChatMessage, ImageContentPart, ImageUrl, TextContentPart

_logger = logging.getLogger(__name__)


def get_chat_messages_from_event(event: BaseEvent) -> list[ChatMessage]:
    """
    Extract chat messages from the LlamaIndex callback event.
    """
    if isinstance(event, LLMCompletionStartEvent):
        return [ChatMessage(role="user", content=event.prompt)]
    elif isinstance(event, LLMCompletionEndEvent):
        return [ChatMessage(role="assistant", content=event.response.text)]
    elif isinstance(event, LLMChatStartEvent):
        # TODO: Parse tool calls
        return [_convert_message_to_mlflow_chat(msg) for msg in event.messages]
    elif isinstance(event, LLMChatEndEvent):
        message = event.response.message
        return [_convert_message_to_mlflow_chat(message)]
    else:
        ValueError(f"Unsupported event type for chat attribute extraction: {type(event)}")


def _convert_message_to_mlflow_chat(message: LLamaChatMessage) -> ChatMessage:
    """Convert a message object from LlamaIndex to MLflow's standard format."""
    content = [_parse_content_block(cb) for cb in _get_content(message)]
    mlflow_message = ChatMessage(role=message.role.value, content=content)

    if tool_calls := message.additional_kwargs.get("tool_calls"):
        mlflow_message.tool_calls = tool_calls

    if tool_call_id := message.additional_kwargs.get("tool_call_id"):
        mlflow_message.tool_call_id = tool_call_id

    return mlflow_message


def _parse_content_block(
    content_block: Union[TextBlock, ImageBlock, str],
) -> Union[ImageContentPart, TextContentPart]:
    if isinstance(content_block, str):
        return TextContentPart(text=content_block, type="text")
    elif isinstance(content_block, TextBlock):
        return TextContentPart(text=content_block.text, type="text")
    elif isinstance(content_block, ImageBlock):
        # https://github.com/run-llama/llama_index/blob/b449940dfad14afbc5721dcd37744d4b0ddac15e/llama-index-core/llama_index/core/base/llms/types.py#L51
        if content_block.url or content_block.path:
            return ImageContentPart(
                type="image_url",
                image_url=ImageUrl(
                    # LlamaIndex support using local path for image but OpenAI does not,
                    # therefore we save the local path as URL.
                    url=str(content_block.url) if content_block.url else str(content_block.path),
                    detail=content_block.detail,
                ),
            )

        if isinstance(content_block.image, bytes):
            image_base64 = str(content_block.image, "utf-8")
        else:
            image_base64 = content_block.image

        return ImageContentPart(
            type="image_url",
            image_url=ImageUrl(
                url=f"data:{content_block.image_mimetype};base64,{image_base64}",
                detail=content_block.detail,
            ),
        )
    else:
        _logger.debug(f"Unsupported content block type, skipping: {type(content_block)}")


def _get_content(message: LLamaChatMessage) -> Union[TextBlock, ImageBlock, str]:
    """
    Get the content blocks from the ChatMessage object in LlamaIndex.

    The `block` field was added in llama-index 0.12.2. Before that, the message stores
    a single string in the `content` field.
    """
    return getattr(message, "blocks", []) or message.content
