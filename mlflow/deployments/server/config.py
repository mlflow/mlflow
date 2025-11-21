from pydantic import ConfigDict

from mlflow.gateway.base_models import ResponseModel
from mlflow.gateway.config import EndpointModelInfo, Limit


class Endpoint(ResponseModel):
    name: str
    endpoint_type: str
    model: EndpointModelInfo
    endpoint_url: str
    limit: Limit | None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "openai-completions",
                "endpoint_type": "llm/v1/completions",
                "model": {
                    "name": "gpt-4o-mini",
                    "provider": "openai",
                },
                "endpoint_url": "/endpoints/completions/invocations",
                "limit": {"calls": 1, "key": None, "renewal_period": "minute"},
            }
        }
    )
