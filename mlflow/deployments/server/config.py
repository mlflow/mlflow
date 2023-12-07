from mlflow.gateway.base_models import ResponseModel
from mlflow.gateway.config import RouteModelInfo


class Endpoint(ResponseModel):
    name: str
    endpoint_type: str
    model: RouteModelInfo
    endpoint_url: str

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
            }
        }
