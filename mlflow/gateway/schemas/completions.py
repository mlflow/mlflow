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
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    total_tokens: Optional[int]
    model: str
    route_type: RouteType


class ResponsePayload(BaseModel, extra=Extra.allow):
    candidates: List[Candidate]
    metadata: Metadata
