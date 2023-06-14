from typing import List, Union

from pydantic import BaseModel, Extra

from ..config import RouteType


class RequestPayload(BaseModel, extra=Extra.allow):
    text: Union[str, List[str]]


class Metadata(BaseModel, extra=Extra.forbid):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model: str
    route_type: RouteType


class ResponsePayload(BaseModel, extra=Extra.forbid):
    embeddings: List[float]
    metadata: Metadata
