from typing import Optional

from mlflow.gateway.base_models import RequestModel, ResponseModel
from mlflow.types.chat import BaseRequestPayload
from mlflow.utils import IS_PYDANTIC_V2_OR_NEWER

_REQUEST_PAYLOAD_EXTRA_SCHEMA = {
    "example": {
        "prompt": "hello",
        "temperature": 0.0,
        "max_tokens": 64,
        "stop": ["END"],
        "n": 1,
    }
}


class RequestPayload(BaseRequestPayload, RequestModel):
    prompt: str
    model: Optional[str] = None

    class Config:
        if IS_PYDANTIC_V2_OR_NEWER:
            json_schema_extra = _REQUEST_PAYLOAD_EXTRA_SCHEMA
        else:
            schema_extra = _REQUEST_PAYLOAD_EXTRA_SCHEMA


class Choice(ResponseModel):
    index: int
    text: str
    finish_reason: Optional[str] = None


class CompletionsUsage(ResponseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


_RESPONSE_PAYLOAD_EXTRA_SCHEMA = {
    "example": {
        "id": "cmpl-123",
        "object": "text_completion",
        "created": 1589478378,
        "model": "gpt-4",
        "choices": [
            {"text": "Hello! I am an AI Assistant!", "index": 0, "finish_reason": "length"}
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }
}


class ResponsePayload(ResponseModel):
    id: Optional[str] = None
    object: str = "text_completion"
    created: int
    model: str
    choices: list[Choice]
    usage: CompletionsUsage

    class Config:
        if IS_PYDANTIC_V2_OR_NEWER:
            json_schema_extra = _RESPONSE_PAYLOAD_EXTRA_SCHEMA
        else:
            schema_extra = _RESPONSE_PAYLOAD_EXTRA_SCHEMA


class StreamDelta(ResponseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class StreamChoice(ResponseModel):
    index: int
    finish_reason: Optional[str] = None
    delta: StreamDelta


_STREAM_RESPONSE_PAYLOAD_EXTRA_SCHEMA = {
    "example": {
        "id": "cmpl-123",
        "object": "text_completion",
        "created": 1589478378,
        "model": "gpt-4",
        "choices": [
            {
                "index": 6,
                "finish_reason": "stop",
                "delta": {"role": "assistant", "content": "you?"},
            }
        ],
    }
}


class StreamResponsePayload(ResponseModel):
    id: Optional[str] = None
    object: str = "text_completion_chunk"
    created: int
    model: str
    choices: list[StreamChoice]

    class Config:
        if IS_PYDANTIC_V2_OR_NEWER:
            json_schema_extra = _STREAM_RESPONSE_PAYLOAD_EXTRA_SCHEMA
        else:
            schema_extra = _STREAM_RESPONSE_PAYLOAD_EXTRA_SCHEMA
