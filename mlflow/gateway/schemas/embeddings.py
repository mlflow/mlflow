from typing import List, Union, Optional

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
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    total_tokens: Optional[int]
    model: str
    route_type: RouteType


class ResponsePayload(BaseModel):
    embeddings: List[List[float]]
    metadata: Metadata

    class Config:
        extra = Extra.forbid
        schema_extra = {
            "example": {
                "embeddings": [
                    [
                        0.1,
                        0.2,
                        0.3,
                    ],
                    [
                        0.4,
                        0.5,
                        0.6,
                    ],
                ],
                "metadata": {
                    "input_tokens": 1,
                    "output_tokens": 0,
                    "total_tokens": 1,
                    "model": "gpt-3.5-turbo",
                    "route_type": "llm/v1/embeddings",
                },
            }
        }
