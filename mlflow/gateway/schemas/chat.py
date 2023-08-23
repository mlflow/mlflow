from enum import Enum
from typing import List, Optional

from pydantic import Field

from mlflow.gateway.base_models import RequestModel, ResponseModel
from mlflow.gateway.config import RouteType


class RequestMessage(RequestModel):
    role: str
    content: str


class BaseRequestPayload(RequestModel):
    temperature: float = Field(0.0, ge=0, le=1)
    candidate_count: int = Field(1, ge=1, le=5)
    stop: Optional[List[str]] = Field(None, min_items=1)
    max_tokens: Optional[int] = Field(None, ge=1)


class RequestPayload(BaseRequestPayload):
    messages: List[RequestMessage] = Field(..., min_items=1)

    class Config:
        schema_extra = {
            "example": {
                "messages": [
                    {
                        "role": "user",
                        "content": "hello world",
                    }
                ],
                "temperature": 0.0,
                "max_tokens": 64,
                "stop": ["END"],
                "candidate_count": 1,
            }
        }


class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"


class ResponseMessage(ResponseModel):
    role: str
    content: str


class CandidateMetadata(ResponseModel, extra="allow"):
    finish_reason: Optional[FinishReason] = None


class Candidate(ResponseModel):
    message: ResponseMessage
    metadata: CandidateMetadata


class Metadata(ResponseModel):
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    model: str
    route_type: RouteType


class ResponsePayload(ResponseModel):
    candidates: List[Candidate]
    metadata: Metadata

    class Config:
        schema_extra = {
            "example": {
                "candidates": [
                    {
                        "message": {
                            "role": "user",
                            "content": "hello world",
                        },
                        "metadata": {
                            "finish_reason": "stop",
                        },
                    }
                ],
                "metadata": {
                    "input_tokens": 1,
                    "output_tokens": 2,
                    "total_tokens": 3,
                    "model": "gpt-3.5-turbo",
                    "route_type": "llm/v1/completions",
                },
            }
        }
