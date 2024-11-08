import json
import logging
import time
from typing import Literal, Optional

import pydantic
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain.schema import ChatMessage as LangChainChatMessage
from packaging.version import Version

from mlflow.exceptions import MlflowException
from mlflow.types.schema import Array, ColSpec, DataType, Schema

_logger = logging.getLogger(__name__)

IS_PYDANTIC_V1 = Version(pydantic.__version__).major < 2


# NB: Even though _ChatMessage is only referenced in one method within this module
# (as of 12/27/2023), it must be defined at the module level for compatibility with
# pydantic < 2
class _ChatMessage(pydantic.BaseModel, extra="forbid"):
    role: str
    content: str

    def to_langchain_message(self) -> LangChainChatMessage:
        if self.role == "system":
            return SystemMessage(content=self.content)
        elif self.role == "assistant":
            return AIMessage(content=self.content)
        elif self.role == "user":
            return HumanMessage(content=self.content)
        else:
            raise MlflowException.invalid_parameter_value(
                f"Unrecognized chat message role: {self.role}"
            )

    @staticmethod
    def get_schema():
        return Schema([ColSpec(DataType.string, "role"), ColSpec(DataType.string, "content")])


class _ChatDeltaMessage(pydantic.BaseModel):
    role: str
    content: str


class _ChatRequest(pydantic.BaseModel, extra="forbid"):
    messages: list[_ChatMessage]


class _ChatChoice(pydantic.BaseModel, extra="forbid"):
    index: int
    message: Optional[_ChatMessage] = None
    finish_reason: Optional[str] = None

    @staticmethod
    def get_schema():
        return Schema(
            [
                ColSpec(DataType.integer, "index"),
                ColSpec(_ChatMessage.get_schema(), "message", required=False),
                ColSpec(DataType.string, "finish_reason", required=False),
            ]
        )


class _ChatChoiceDelta(pydantic.BaseModel):
    index: int
    finish_reason: Optional[str] = None
    delta: _ChatDeltaMessage


class _ChatUsage(pydantic.BaseModel, extra="forbid"):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    @staticmethod
    def get_schema():
        return Schema(
            [
                ColSpec(DataType.integer, "prompt_tokens", required=False),
                ColSpec(DataType.integer, "completion_tokens", required=False),
                ColSpec(DataType.integer, "total_tokens", required=False),
            ]
        )


class _ChatResponse(pydantic.BaseModel, extra="forbid"):
    id: Optional[str] = None
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    # Make the model field optional since we may not be able to get a stable model identifier
    # for an arbitrary LangChain model
    model: Optional[str] = None
    choices: list[_ChatChoice]
    usage: _ChatUsage

    @staticmethod
    def get_schema():
        return Schema(
            [
                ColSpec(DataType.string, "id", required=False),
                ColSpec(DataType.string, "object"),
                ColSpec(DataType.integer, "created"),
                ColSpec(DataType.string, "model", required=False),
                ColSpec(Array(_ChatChoice.get_schema()), "choices"),
                ColSpec(_ChatUsage.get_schema(), "usage"),
            ]
        )


class _ChatChunkResponse(pydantic.BaseModel):
    id: Optional[str] = None
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    # Make the model field optional since we may not be able to get a stable model identifier
    # for an arbitrary LangChain model
    model: Optional[str] = None
    choices: list[_ChatChoiceDelta]


def try_transform_response_to_chat_format(response):
    if isinstance(response, str):
        message_content = response
        message_id = None
    elif isinstance(response, AIMessage):
        message_content = response.content
        message_id = getattr(response, "id", None)
    else:
        return response

    transformed_response = _ChatResponse(
        id=message_id,
        created=int(time.time()),
        model=None,
        choices=[
            _ChatChoice(
                index=0,
                message=_ChatMessage(
                    role="assistant",
                    content=message_content,
                ),
                finish_reason=None,
            )
        ],
        usage=_ChatUsage(
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
        ),
    )

    if IS_PYDANTIC_V1:
        return json.loads(transformed_response.json())
    else:
        return transformed_response.model_dump(mode="json")


def try_transform_response_iter_to_chat_format(chunk_iter):
    from langchain_core.messages.ai import AIMessageChunk

    def _gen_converted_chunk(message_content, message_id, finish_reason):
        transformed_response = _ChatChunkResponse(
            id=message_id,
            created=int(time.time()),
            model=None,
            choices=[
                _ChatChoiceDelta(
                    index=0,
                    delta=_ChatDeltaMessage(
                        role="assistant",
                        content=message_content,
                    ),
                    finish_reason=finish_reason,
                )
            ],
        )

        if IS_PYDANTIC_V1:
            return json.loads(transformed_response.json())
        else:
            return transformed_response.model_dump(mode="json")

    def _convert(chunk):
        if isinstance(chunk, str):
            message_content = chunk
            message_id = None
            finish_reason = None
        elif isinstance(chunk, AIMessageChunk):
            message_content = chunk.content
            message_id = getattr(chunk, "id", None)

            if response_metadata := getattr(chunk, "response_metadata", None):
                finish_reason = response_metadata.get("finish_reason")
            else:
                finish_reason = None
        elif isinstance(chunk, AIMessage):
            # The langchain chat model does not support stream
            # so `model.stream` returns the whole result.
            message_content = chunk.content
            message_id = getattr(chunk, "id", None)
            finish_reason = "stop"
        else:
            return chunk
        return _gen_converted_chunk(
            message_content,
            message_id=message_id,
            finish_reason=finish_reason,
        )

    return map(_convert, chunk_iter)


def _convert_chat_request_or_throw(chat_request: dict) -> list[BaseMessage]:
    try:
        from langchain_core.messages.utils import convert_to_messages

        return convert_to_messages(chat_request["messages"])
    except ImportError:
        pass

    # it's safe to drop below when langchain >= 0.1.20
    # TODO: drop this once we updated minimum supported langchain version
    if IS_PYDANTIC_V1:
        model = _ChatRequest.parse_obj(chat_request)
    else:
        model = _ChatRequest.model_validate(chat_request)

    return [message.to_langchain_message() for message in model.messages]


def json_dict_might_be_chat_request(json_message: dict):
    return (
        isinstance(json_message, dict)
        and "messages" in json_message
        and
        # Additional keys can't be specified when calling LangChain invoke() / batch()
        # with chat messages
        len(json_message) == 1
        # messages field should be a list
        and isinstance(json_message["messages"], list)
    )
