"""
This module defines the schemas for the MLflow AI Gateway's chat endpoint.

The schemas must be compatible with OpenAI's Chat Completion API.
https://platform.openai.com/docs/api-reference/chat

NB: These Pydantic models just alias the models defined in mlflow.types.chat to avoid code
    duplication, but with the addition of RequestModel and ResponseModel base classes.
"""

from typing import Literal, Optional

from pydantic import Field

from mlflow.gateway.base_models import RequestModel, ResponseModel
from mlflow.types.chat import (
    ChatChoice,
    ChatChoiceDelta,
    ChatChunkChoice,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    FunctionToolDefinition,
)
from mlflow.types.chat import ChatCompletionRequest as _Function
from mlflow.types.chat import ChatUsage as _ChatUsage
from mlflow.types.chat import ToolCall as _ToolCall
from mlflow.utils import IS_PYDANTIC_V2_OR_NEWER


class RequestMessage(ChatMessage, RequestModel):
    pass


class UnityCatalogFunctionToolDefinition(RequestModel):
    name: str


class ChatToolWithUC(RequestModel):
    type: Literal["function", "uc_function"]
    function: Optional[FunctionToolDefinition] = None
    uc_function: Optional[UnityCatalogFunctionToolDefinition] = None


_REQUEST_PAYLOAD_EXTRA_SCHEMA = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
    "temperature": 0.0,
    "max_tokens": 64,
    "stop": ["END"],
    "n": 1,
    "stream": False,
}


# NB: RequestPayload is mostly OpenAI's ChatCompletion API spec, except for the `tools` field.
#     The field accepts a tool with a special type 'uc_function' for Unity Catalog integration.
#     https://mlflow.org/docs/latest/llms/deployments/uc_integration.html
class RequestPayload(ChatCompletionRequest, RequestModel):
    tools: Optional[list[ChatToolWithUC]] = None

    class Config:
        if IS_PYDANTIC_V2_OR_NEWER:
            json_schema_extra = _REQUEST_PAYLOAD_EXTRA_SCHEMA
        else:
            schema_extra = _REQUEST_PAYLOAD_EXTRA_SCHEMA


_RESPONSE_PAYLOAD_EXTRA_SCHEMA = {
    "example": {
        "id": "3cdb958c-e4cc-4834-b52b-1d1a7f324714",
        "object": "chat.completion",
        "created": 1700173217,
        "model": "llama-2-70b-chat-hf",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello! I am an AI assistant"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
    }
}


class Function(_Function, ResponseModel):
    pass


class ToolCall(_ToolCall, ResponseModel):
    pass


class ResponseMessage(ChatMessage, ResponseModel):
    # Override the `tool_call_id` field to be excluded from the response.
    # This is a band-aid solution to avoid exposing the tool_call_id in the response,
    # while we use the same ChatMessage model for both request and response.
    tool_call_id: Optional[str] = Field(None, exclude=True)


class Choice(ChatChoice, ResponseModel):
    # Override the `message` field to use the ResponseMessage model.
    message: ResponseMessage


class ChatUsage(_ChatUsage, ResponseModel):
    pass


class ResponsePayload(ChatCompletionResponse, ResponseModel):
    # Override the `choices` field to use the Choice model
    choices: list[Choice]

    class Config:
        if IS_PYDANTIC_V2_OR_NEWER:
            json_schema_extra = _RESPONSE_PAYLOAD_EXTRA_SCHEMA
        else:
            schema_extra = _RESPONSE_PAYLOAD_EXTRA_SCHEMA


class StreamDelta(ChatChoiceDelta, ResponseModel):
    pass


class StreamChoice(ChatChunkChoice, ResponseModel):
    pass


_STREAM_RESPONSE_PAYLOAD_EXTRA_SCHEMA = {
    "example": {
        "id": "3cdb958c-e4cc-4834-b52b-1d1a7f324714",
        "object": "chat.completion",
        "created": 1700173217,
        "model": "llama-2-70b-chat-hf",
        "choices": [
            {
                "index": 6,
                "finish_reason": "stop",
                "delta": {"role": "assistant", "content": "you?"},
            }
        ],
    }
}


class StreamResponsePayload(ChatCompletionChunk, ResponseModel):
    class Config:
        if IS_PYDANTIC_V2_OR_NEWER:
            json_schema_extra = _STREAM_RESPONSE_PAYLOAD_EXTRA_SCHEMA
        else:
            schema_extra = _STREAM_RESPONSE_PAYLOAD_EXTRA_SCHEMA
