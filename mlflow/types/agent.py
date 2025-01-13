from typing import Any, Optional
from uuid import uuid4

from pydantic import Field

from mlflow.types.chat import BaseModel, ChatUsage, ToolCall
from mlflow.types.llm import (
    _custom_inputs_col_spec,
    _custom_outputs_col_spec,
    _token_usage_stats_col_spec,
)
from mlflow.types.schema import (
    Array,
    ColSpec,
    DataType,
    Map,
    Object,
    Property,
    Schema,
)
from mlflow.utils import IS_PYDANTIC_V2_OR_NEWER

if IS_PYDANTIC_V2_OR_NEWER:
    from pydantic import model_validator
else:
    from pydantic import root_validator


class ChatAgentMessage(BaseModel):
    """
    A message in a ChatAgent model request or response.

    Args:
        role (str): The role of the entity that sent the message (e.g. ``"user"``, ``"system"``,
            ``"assistant"``, ``"tool"``).
        content (str): The content of the message.
            **Optional** Can be ``None`` if refusal or tool_calls are provided.
        name (str): The name of the entity that sent the message. **Optional** defaults to ``None``
        id (str): The ID of the message. **Optional** defaults to a random UUID
        tool_calls (List[:py:class:`ToolCallPydantic`]): A list of tool calls made by the model.
            **Optional** defaults to ``None``
        tool_call_id (str): The ID of the tool call that this message is a response to.
            **Optional** defaults to ``None``
        attachments (Dict[str, str]): A dictionary of attachments. **Optional** defaults to ``None``
        finish_reason (str): The reason why generation stopped. **Optional** defaults to ``None``
    """

    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    id: Optional[str] = Field(default_factory=lambda: str(uuid4()))
    tool_calls: Optional[list[ToolCall]] = None
    tool_call_id: Optional[str] = None
    # TODO make this a pydantic class with subtypes once we have more details on usage
    attachments: Optional[dict[str, str]] = None
    finish_reason: Optional[str] = None
    # TODO: add finish_reason_metadata once we have a plan for usage

    if IS_PYDANTIC_V2_OR_NEWER:

        @model_validator(mode="after")
        def check_content_and_tool_calls(cls, chat_agent_msg):
            """
            Ensure at least one of 'content' or 'tool_calls' is set.
            """
            if not chat_agent_msg.content and not chat_agent_msg.tool_calls:
                raise ValueError("Either 'content' or 'tool_calls' must be provided.")
            return chat_agent_msg
    else:

        @root_validator
        def check_content_and_tool_calls(cls, values):
            """
            Ensure at least one of 'content' or 'tool_calls' is set.
            """
            content = values.get("content")
            tool_calls = values.get("tool_calls")
            if not content and not tool_calls:
                raise ValueError("Either 'content' or 'tool_calls' must be provided.")
            return values


class Context(BaseModel):
    """
    Context to be used in a ChatAgent endpoint.

    Args:
        conversation_id (str): The ID of the conversation. **Optional** defaults to ``None``
        user_id (str): The ID of the user. **Optional** defaults to ``None``
    """

    conversation_id: Optional[str] = None
    user_id: Optional[str] = None


class ChatAgentParams(BaseModel):
    """
    Common parameters used for the ChatAgent interface.

    Args:
        context (:py:class:`Context`): The context to be used in the chat endpoint. Includes
            conversation_id and user_id. **Optional** defaults to ``None``
        custom_inputs (Dict[str, Any]): An optional param to provide arbitrary additional context
            to the model. The dictionary values must be JSON-serializable.
            **Optional** defaults to ``None``
        stream (bool): Whether to stream back responses as they are generated.
            **Optional**, defaults to ``False``
    """

    context: Optional[Context] = None
    custom_inputs: Optional[dict[str, Any]] = None
    stream: Optional[bool] = False


class ChatAgentRequest(ChatAgentParams):
    """
    Format of a ChatAgent interface request.

    Args:
        messages: A list of :py:class:`ChatAgentMessage` that will be passed to the model.
            **Optional**, defaults to empty list (``[]``)
        context (:py:class:`Context`): The context to be used in the chat endpoint. Includes
            conversation_id and user_id. **Optional** defaults to ``None``
        custom_inputs (Dict[str, Any]): An optional param to provide arbitrary additional context
            to the model. The dictionary values must be JSON-serializable.
            **Optional** defaults to ``None``
        stream (bool): Whether to stream back responses as they are generated.
            **Optional**, defaults to ``False``
    """

    messages: list[ChatAgentMessage] = Field(default_factory=list)


class ChatAgentResponse(BaseModel):
    messages: list[ChatAgentMessage]
    custom_outputs: Optional[dict[str, Any]] = None
    usage: Optional[ChatUsage] = None


# fmt: off
_chat_agent_messages_col_spec = ColSpec(
    name="messages",
    type=Array(
        Object(
            [
                Property("role", DataType.string),
                Property("content", DataType.string, False),
                Property("name", DataType.string, False),
                Property("id", DataType.string, False),
                Property("tool_calls", Array(Object([
                    Property("id", DataType.string),
                    Property("function", Object([
                        Property("name", DataType.string),
                        Property("arguments", DataType.string),
                    ])),
                    Property("type", DataType.string),
                ])), False),
                Property("tool_call_id", DataType.string, False),
                Property("attachments", Map(DataType.string), False),
                Property("finish_reason", DataType.string, False),
            ]
        )
    ),
)

# TODO: move out all params to a ParamSchema when Map(AnyType()) is supported by ParamSpec
CHAT_AGENT_INPUT_SCHEMA = Schema(
    [
        _chat_agent_messages_col_spec,
        ColSpec(name="context", type=Object([
            Property("conversation_id", DataType.string, False),
            Property("user_id", DataType.string, False),
        ]), required=False),
        _custom_inputs_col_spec,
        ColSpec(name="stream", type=DataType.boolean, required=False),
    ]
)

CHAT_AGENT_OUTPUT_SCHEMA = Schema(
    [
        _chat_agent_messages_col_spec,
        _custom_outputs_col_spec,
        _token_usage_stats_col_spec,
    ]
)

CHAT_AGENT_INPUT_EXAMPLE = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
    "stream": False,
}
