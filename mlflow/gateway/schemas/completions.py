from typing import List, Optional

from pydantic import BaseModel, Extra


class RequestPayload(BaseModel, extra=Extra.allow):
    prompt: str


class CandidateMetadata(BaseModel):
    finish_reason: Optional[str]


class Candidate(BaseModel, extra=Extra.allow):
    text: str
    metadata: CandidateMetadata


class Metadata(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model: str
    route_type: str


class ResponsePayload(BaseModel, extra=Extra.allow):
    candidates: List[Candidate]
    metadata: Metadata
