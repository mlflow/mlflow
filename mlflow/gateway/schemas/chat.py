from typing import List, Optional

from pydantic import BaseModel, Extra, Field


class Message(BaseModel):
    role: str
    content: str


class RequestPayload(BaseModel, extra=Extra.allow):
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
    route_type: str


class ResponsePayload(BaseModel, extra=Extra.forbid):
    candidates: List[Candidate]
    metadata: Metadata
