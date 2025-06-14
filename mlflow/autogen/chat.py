import logging
from typing import TYPE_CHECKING, Optional, Union

from opentelemetry.sdk.trace import Span

from mlflow.tracing.utils import set_span_chat_messages, set_span_chat_tools
from mlflow.types.chat import (
    ChatMessage,
    ChatTool,
    Function,
    TextContentPart,
    ToolCall,
)

if TYPE_CHECKING:
    from autogen_core import FunctionCall
    from autogen_core.models import LLMMessage
    from autogen_core.tools import BaseTool, ToolSchema

_logger = logging.getLogger(__name__)


def log_tools(span: Span, tools: list[Union["BaseTool", "ToolSchema"]]):
    """
    Log Autogen tool definitions into the passed in span.

    Ref: https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/components/tools.html

    Args:
        span: The span to log the tools into.
        tools: A list of Autogen BaseTool.
    """
    from autogen_core.tools import BaseTool

    try:
        tools = [
            ChatTool(
                type="function",
                function=tool.schema if isinstance(tool, BaseTool) else tool,
            )
            for tool in tools
        ]
        set_span_chat_tools(span, tools)
    except Exception:
        _logger.debug(f"Failed to log tools to Span {span}.", exc_info=True)


def convert_assistant_message_to_chat_message(
    content: Union[str, list["FunctionCall"]],
) -> Optional[ChatMessage]:
    """
    Convert an Autogen assistant message to a ChatMessage.
    The content of assistant message is a str or a list of tool calls.

    Args:
        content: The Autogen assistant message content to convert.

    Returns:
        A ChatMessage object
    """
    from autogen_core import FunctionCall

    if isinstance(content, str):
        return ChatMessage(role="assistant", content=content)
    elif isinstance(content, list) and all(isinstance(f, FunctionCall) for f in content):
        return ChatMessage(
            role="assistant",
            tool_calls=[
                ToolCall(
                    id=f.id, type="function", function=Function(name=f.name, arguments=f.arguments)
                )
                for f in content
            ],
        )
    else:
        _logger.debug(f"Unsupported message type: {type(content)}. Skipping conversion.")


def log_chat_messages(span: Span, messages: list["LLMMessage"]):
    """
    Log Autogen chat messages into the passed in span.

    Args:
        span: The span to log the tools into.
        messages: A list of Autogen chat messages.
    """
    from autogen_core.models import (
        AssistantMessage,
        FunctionExecutionResultMessage,
        SystemMessage,
        UserMessage,
    )

    chat_messages = []
    for message in messages:
        if isinstance(message, SystemMessage):
            chat_messages.append(
                ChatMessage(
                    role="system",
                    content=message.content,
                )
            )
        elif isinstance(message, UserMessage):
            content = message.content
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, str):
                        parts.append(
                            TextContentPart(
                                type="text",
                                text=part,
                            )
                        )
                    else:
                        # The content type of UserMessage is text or image
                        parts.append(part.to_openai_format())
                content = parts
            chat_messages.append(
                ChatMessage(
                    role="user",
                    content=content,
                )
            )
        elif isinstance(message, AssistantMessage):
            content = message.content
            if chat_message := convert_assistant_message_to_chat_message(content):
                chat_messages.append(chat_message)
        elif isinstance(message, FunctionExecutionResultMessage):
            chat_messages.append(
                ChatMessage(
                    role="user",
                    content=message.model_dump(),
                )
            )
        else:
            _logger.debug(f"Unsupported message type: {type(message)}. Skipping logging.")

    try:
        set_span_chat_messages(span, chat_messages)
    except Exception:
        _logger.debug(f"Failed to log chat messages to Span {span}.", exc_info=True)
