"""
This module defines the schemas for the MLflow AI Gateway's chat endpoint.

The schemas must be compatible with OpenAI's Chat Completion API.
https://platform.openai.com/docs/api-reference/chat

NB: These Pydantic models just alias the models defined in mlflow.types.chat to avoid code
    duplication, but with the addition of RequestModel and ResponseModel base classes.
"""

from typing import Literal, Optional

from mlflow.gateway.base_models import RequestModel, ResponseModel

# Import marked with noqa is for backward compatibility
from mlflow.types.chat import (
    ChatChoice,
    ChatChoiceDelta,
    ChatChunkChoice,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatUsage,  # noqa F401
    Function,  # noqa F401
    FunctionToolDefinition,
    ToolCall,  # noqa F401
)
from mlflow.utils import IS_PYDANTIC_V2_OR_NEWER

# NB: `import x as y` does not work and will cause a Pydantic error.
Choice = ChatChoice
StreamDelta = ChatChoiceDelta
StreamChoice = ChatChunkChoice
RequestMessage = ChatMessage
ResponseMessage = ChatMessage


class UnityCatalogFunctionToolDefinition(RequestModel):
    name: str


class ChatToolWithUC(RequestModel):
    """
    A tool definition for the chat endpoint with Unity Catalog integration.
    The Gateway request accepts a special tool type 'uc_function' for Unity Catalog integration.
    https://mlflow.org/docs/latest/llms/deployments/uc_integration.html
    """

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


class ResponsePayload(ChatCompletionResponse, ResponseModel):
    class Config:
        if IS_PYDANTIC_V2_OR_NEWER:
            json_schema_extra = _RESPONSE_PAYLOAD_EXTRA_SCHEMA
        else:
            schema_extra = _RESPONSE_PAYLOAD_EXTRA_SCHEMA


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
