from typing import List, Literal, Optional, Union

from mlflow.gateway.base_models import RequestModel, ResponseModel
from mlflow.gateway.config import IS_PYDANTIC_V2

_REQUEST_PAYLOAD_EXTRA_SCHEMA = {
    "example": {
        "input": ["hello", "world"],
    }
}


class RequestPayload(RequestModel):
    input: Union[str, List[str]]

    class Config:
        if IS_PYDANTIC_V2:
            json_schema_extra = _REQUEST_PAYLOAD_EXTRA_SCHEMA
        else:
            schema_extra = _REQUEST_PAYLOAD_EXTRA_SCHEMA


class EmbeddingObject(ResponseModel):
    object: Literal["embedding"] = "embedding"
    embedding: List[float]
    index: int


class EmbeddingsUsage(ResponseModel):
    prompt_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class ResponsePayload(ResponseModel):
    object: Literal["list"] = "list"
    data: List[EmbeddingObject]
    model: str
    usage: EmbeddingsUsage

    class Config:
        schema_extra = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [
                        0.017291732,
                        -0.017291732,
                        0.014577783,
                        -0.02902633,
                        -0.037271563,
                        0.019333655,
                        -0.023055641,
                        -0.007359971,
                        -0.015818445,
                        -0.030654699,
                        0.008348623,
                        0.018312693,
                        -0.017149571,
                        -0.0044424757,
                        -0.011165961,
                        0.01018377,
                    ],
                },
                {
                    "object": "embedding",
                    "index": 1,
                    "embedding": [
                        0.0060126893,
                        -0.008691099,
                        -0.0040095365,
                        0.019889368,
                        0.036211833,
                        -0.0013270887,
                        0.013401738,
                        -0.0036735237,
                        -0.0049594184,
                        0.035229642,
                        -0.03435084,
                        0.019798903,
                        -0.0006110424,
                        0.0073793563,
                        0.005657291,
                        0.022487005,
                    ],
                },
            ],
            "model": "text-embedding-ada-002-v2",
            "usage": {"prompt_tokens": 400, "total_tokens": 400},
        }
