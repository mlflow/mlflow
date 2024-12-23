from mlflow.gateway.base_models import RequestModel, ResponseModel
from mlflow.types.chat import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from mlflow.utils import IS_PYDANTIC_V2_OR_NEWER

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
