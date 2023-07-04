from typing import List, Optional

from ..base_models import ResponseModel
from .chat import BaseRequestPayload, FinishReason
from ..config import RouteType


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


class CandidateMetadata(ResponseModel):
    finish_reason: Optional[FinishReason]


class Candidate(ResponseModel):
    text: str
    metadata: CandidateMetadata


class Metadata(ResponseModel):
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    total_tokens: Optional[int]
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
                        "text": "hello world",
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
