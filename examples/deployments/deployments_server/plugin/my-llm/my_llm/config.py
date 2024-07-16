from pydantic import validator

from mlflow.gateway.base_models import ConfigModel
from mlflow.gateway.config import _resolve_api_key_from_input


class MyLLMConfig(ConfigModel):
    my_llm_api_key: str

    @validator("my_llm_api_key", pre=True)
    def validate_my_llm_api_key(cls, value):
        return _resolve_api_key_from_input(value)
