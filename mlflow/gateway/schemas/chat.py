from typing import List, Optional

from pydantic import BaseModel, Extra, Field

from ..config import RouteType


class Message(BaseModel):
    role: str
    content: str


class BaseRequestPayload(BaseModel, extra=Extra.allow):
    temperature: int = Field(0, ge=0, le=1)
    stop: List[str] = Field([])
    max_tokens: Optional[int] = Field(None, ge=0)
    candidate_count: Optional[int] = Field(None, ge=1, le=5)


class RequestPayload(BaseRequestPayload):
    messages: List[Message] = Field(..., min_items=1)


class CandidateMetadata(BaseModel, extra=Extra.forbid):
    finish_reason: Optional[str]


class Candidate(BaseModel):
    message: Message
    metadata: CandidateMetadata


class Metadata(BaseModel, extra=Extra.forbid):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model: str
    route_type: RouteType


class ResponsePayload(BaseModel, extra=Extra.forbid):
    candidates: List[Candidate]
    metadata: Metadata
