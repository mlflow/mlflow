from typing import List, Literal, Optional

from mlflow.gateway.base_models import ResponseModel
from mlflow.gateway.config import IS_PYDANTIC_V2
from mlflow.gateway.schemas.chat import BaseRequestPayload

_REQUEST_PAYLOAD_EXTRA_SCHEMA = {
    "example": {
        "prompt": "hello",
        "temperature": 0.0,
        "max_tokens": 64,
        "stop": ["END"],
        "n": 1,
    }
}


class RequestPayload(BaseRequestPayload):
    prompt: str

    class Config:
        if IS_PYDANTIC_V2:
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
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: List[Choice]
    usage: CompletionsUsage

    class Config:
        if IS_PYDANTIC_V2:
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
    object: Literal["text_completion_chunk"] = "text_completion_chunk"
    created: int
    model: str
    choices: List[StreamChoice]

    class Config:
        if IS_PYDANTIC_V2:
            json_schema_extra = _STREAM_RESPONSE_PAYLOAD_EXTRA_SCHEMA
        else:
            schema_extra = _STREAM_RESPONSE_PAYLOAD_EXTRA_SCHEMA
