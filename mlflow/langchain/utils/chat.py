import json
import logging
import time
from typing import Literal, Optional, Union

import pydantic
from langchain.agents import AgentExecutor
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.schema import ChatMessage as LangChainChatMessage
from packaging.version import Version

from mlflow.environment_variables import (
    MLFLOW_CONVERT_MESSAGES_DICT_FOR_LANGCHAIN,
)
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


def _convert_chat_request_or_throw(chat_request: dict):
    if IS_PYDANTIC_V1:
        model = _ChatRequest.parse_obj(chat_request)
    else:
        model = _ChatRequest.model_validate(chat_request)

    return [message.to_langchain_message() for message in model.messages]


def convert_chat_request(chat_request: Union[dict, list[dict]]):
    if isinstance(chat_request, list):
        return [_convert_chat_request_or_throw(request) for request in chat_request]
    else:
        return _convert_chat_request_or_throw(chat_request)


def _get_lc_model_input_fields(lc_model) -> set[str]:
    try:
        if hasattr(lc_model, "input_schema"):
            return set(lc_model.input_schema.__fields__)
    except Exception as e:
        _logger.debug(
            f"Unexpected exception while checking LangChain input schema for"
            f" request transformation: {e}"
        )

    return set()


def should_transform_requst_json_for_chat(lc_model):
    # Avoid converting the request to LangChain's Message format if the chain
    # is an AgentExecutor, as LangChainChatMessage might not be accepted by the chain
    if isinstance(lc_model, AgentExecutor):
        return False

    input_fields = _get_lc_model_input_fields(lc_model)
    if "messages" in input_fields:
        # If the chain accepts a "messages" field directly, don't attempt to convert
        # the request to LangChain's Message format automatically. Assume that the chain
        # is handling the "messages" field by itself
        return False

    return True


def transform_request_json_for_chat_if_necessary(request_json, lc_model):
    """
    Convert the input request JSON to LangChain's Message format if the LangChain model
    accepts ChatMessage objects (e.g. AIMessage, HumanMessage, SystemMessage) as input.

    Args:
        request_json: The input request JSON.
        lc_model: The LangChain model.

    Returns:
        A 2-element tuple containing:

            1. The new request.
            2. A boolean indicating whether or not the request was transformed from the OpenAI
            chat format.
    """

    def json_dict_might_be_chat_request(json_message):
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

    def is_list_of_chat_messages(json_message: list[dict]):
        return isinstance(json_message, list) and all(
            json_dict_might_be_chat_request(message) for message in json_message
        )

    should_convert = MLFLOW_CONVERT_MESSAGES_DICT_FOR_LANGCHAIN.get()
    if should_convert is None:
        should_convert = should_transform_requst_json_for_chat(lc_model) and (
            json_dict_might_be_chat_request(request_json) or is_list_of_chat_messages(request_json)
        )
        if should_convert:
            _logger.debug(
                "Converting the request JSON to LangChain's Message format. "
                "To disable this conversion, set the environment variable "
                f"`{MLFLOW_CONVERT_MESSAGES_DICT_FOR_LANGCHAIN}` to 'false'."
            )

    if should_convert:
        try:
            return convert_chat_request(request_json), True
        except pydantic.ValidationError:
            _logger.debug(
                "Failed to convert the request JSON to LangChain's Message format. "
                "The request will be passed to the LangChain model as-is. ",
                exc_info=True,
            )
            return request_json, False
    else:
        return request_json, False
