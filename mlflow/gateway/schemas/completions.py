from typing import List, Optional

from pydantic import BaseModel, Extra

from .chat import BaseRequestPayload, FinishReason
from ..config import RouteType


class RequestPayload(BaseRequestPayload):
    prompt: str


class CandidateMetadata(BaseModel, extra=Extra.forbid):
    finish_reason: Optional[FinishReason]


class Candidate(BaseModel, extra=Extra.forbid):
    text: str
    metadata: CandidateMetadata


class Metadata(BaseModel, extra=Extra.forbid):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model: str
    route_type: RouteType


class ResponsePayload(BaseModel, extra=Extra.allow):
    candidates: List[Candidate]
    metadata: Metadata
