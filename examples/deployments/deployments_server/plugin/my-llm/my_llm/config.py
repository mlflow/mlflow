import os

from pydantic import field_validator

from mlflow.gateway.base_models import ConfigModel


class MyLLMConfig(ConfigModel):
    my_llm_api_key: str

    @field_validator("my_llm_api_key", mode="before")
    def validate_my_llm_api_key(cls, value):
        if value.startswith("$"):
            # This resolves the API key from an environment variable
            env_var_name = value[1:]
            if env_var := os.getenv(env_var_name):
                return env_var
            else:
                raise ValueError(f"Environment variable {env_var_name!r} is not set")
        return value
