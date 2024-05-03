from pydantic import validator

from mlflow.gateway.base_models import ProviderConfigModel
from mlflow.gateway.config import _resolve_api_key_from_input


class FooConfig(ProviderConfigModel):
    foo_api_key: str

    @validator("foo_api_key", pre=True)
    def validate_foo_api_key(cls, value):
        return _resolve_api_key_from_input(value)
