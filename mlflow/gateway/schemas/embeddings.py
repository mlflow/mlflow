from typing import List, Optional, Union

from mlflow.gateway.base_models import RequestModel, ResponseModel
from mlflow.gateway.config import RouteType


class RequestPayload(RequestModel):
    text: Union[str, List[str]]

    class Config:
        schema_extra = {
            "example": {
                "text": ["hello", "world"],
            }
        }


class Metadata(ResponseModel):
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    model: str
    route_type: RouteType


class ResponsePayload(ResponseModel):
    embeddings: List[List[float]]
    metadata: Metadata

    class Config:
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
