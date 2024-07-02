from typing import Optional

from mlflow.gateway.base_models import ResponseModel
from mlflow.gateway.config import (
    Limit,
    RouteConfig,  # noqa: F401
    RouteModelInfo,
    _resolve_api_key_from_input,  # noqa: F401
)


class Endpoint(ResponseModel):
    name: str
    endpoint_type: str
    model: RouteModelInfo
    endpoint_url: str
    limit: Optional[Limit]

    class Config:
        schema_extra = {
            "example": {
                "name": "openai-completions",
                "endpoint_type": "llm/v1/completions",
                "model": {
                    "name": "gpt-3.5-turbo",
                    "provider": "openai",
                },
                "endpoint_url": "/endpoints/completions/invocations",
                "limit": {"calls": 1, "key": None, "renewal_period": "minute"},
            }
        }
