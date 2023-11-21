from typing import List, Literal, Optional

from mlflow.gateway.base_models import ResponseModel
from mlflow.gateway.schemas.chat import BaseRequestPayload


class RequestPayload(BaseRequestPayload):
    prompt: str

    class Config:
        schema_extra = {
            "example": {
                "prompt": "hello",
                "temperature": 0.0,
                "max_tokens": 64,
                "stop": ["END"],
                "candidate_count": 1,
            }
        }


class Choice(ResponseModel):
    index: int
    text: str
    finish_reason: Optional[str] = None


class CompletionsUsage(ResponseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class ResponsePayload(ResponseModel):
    id: Optional[str] = None
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: List[Choice]
    usage: CompletionsUsage

    class Config:
        schema_extra = {
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
