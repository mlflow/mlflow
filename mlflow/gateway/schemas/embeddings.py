from typing import List, Union

from pydantic import BaseModel, Extra

from ..config import RouteType


class RequestPayload(BaseModel):
    text: Union[str, List[str]]

    class Config:
        extra = Extra.allow
        schema_extra = {
            "example": {
                "text": ["hello", "world"],
            }
        }


class Metadata(BaseModel, extra=Extra.forbid):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model: str
    route_type: RouteType


class ResponsePayload(BaseModel, extra=Extra.forbid):
    embeddings: List[List[float]]
    metadata: Metadata
