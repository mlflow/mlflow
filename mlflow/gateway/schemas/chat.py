from typing import List, Optional

from pydantic import Field

from mlflow.gateway.base_models import RequestModel, ResponseModel


class RequestMessage(RequestModel):
    role: str
    content: str


class BaseRequestPayload(RequestModel):
    temperature: float = Field(0.0, ge=0, le=2)
    n: int = Field(1, ge=1)
    stop: Optional[List[str]] = Field(None, min_items=1)
    max_tokens: Optional[int] = Field(None, ge=1)


class RequestPayload(BaseRequestPayload):
    messages: List[RequestMessage] = Field(..., min_items=1)

    class Config:
        schema_extra = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
            "temperature": 0.0,
            "max_tokens": 64,
            "stop": ["END"],
            "n": 1,
        }


class ResponseMessage(ResponseModel):
    role: str
    content: str


class Choice(ResponseModel):
    index: int
    message: ResponseMessage
    finish_reason: Optional[str] = None


class ChatUsage(ResponseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class ResponsePayload(ResponseModel):
    id: Optional[str] = None
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: ChatUsage

    class Config:
        schema_extra = {
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
