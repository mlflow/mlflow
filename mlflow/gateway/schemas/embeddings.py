from typing import List

from pydantic import BaseModel, Extra


class RequestPayload(BaseModel, extra=Extra.allow):
    text: str


class Metadata(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model: str
    route_type: str


class ResponsePayload(BaseModel, extra=Extra.allow):
    embeddings: List[float]
    metadata: Metadata
