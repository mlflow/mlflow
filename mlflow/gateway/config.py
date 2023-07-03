import pathlib
from enum import Enum
import json
import logging
import os
from pathlib import Path
from pydantic import BaseModel, validator, root_validator, parse_obj_as, Extra, ValidationError
from pydantic.json import pydantic_encoder
from typing import Optional, Union, List
import yaml

from mlflow.exceptions import MlflowException
from mlflow.gateway.utils import is_valid_endpoint_name, check_configuration_route_name_collisions


_logger = logging.getLogger(__name__)


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    # Note: Databricks Model Serving is only supported on Databricks
    DATABRICKS_MODEL_SERVING = "databricks-model-serving"
    COHERE = "cohere"

    @classmethod
    def values(cls):
        return {p.value for p in cls}


class RouteType(str, Enum):
    LLM_V1_COMPLETIONS = "llm/v1/completions"
    LLM_V1_CHAT = "llm/v1/chat"
    LLM_V1_EMBEDDINGS = "llm/v1/embeddings"


class CohereConfig(BaseModel, extra=Extra.allow):
    api_key: str
    api_base: str = "https://api.cohere.ai/v1"

    # pylint: disable=no-self-argument
    @validator("api_key", pre=True)
    def validate_api_key(cls, value):
        return _resolve_api_key_from_input(value)


class OpenAIAPIType(str, Enum):
    OPENAI = "openai"
    AZURE = "azure"
    AZUREAD = "azuread"


class OpenAIConfig(BaseModel, extra=Extra.allow):
    openai_api_key: str
    openai_api_type: str = OpenAIAPIType.OPENAI
    openai_api_base: Optional[str] = None
    openai_api_version: Optional[str] = None
    openai_deployment_name: Optional[str] = None
    openai_organization: Optional[str] = None

    # pylint: disable=no-self-argument
    @validator("openai_api_key", pre=True)
    def validate_openai_api_key(cls, value):
        return _resolve_api_key_from_input(value)

    # pylint: disable=no-self-argument
    @validator("openai_api_type", pre=True)
    def validate_openai_api_type(cls, value):
        api_type = value.lower()
        if api_type not in [api_type.value for api_type in OpenAIAPIType]:
            raise MlflowException.invalid_parameter_value(f"Invalid OpenAI API type '{api_type}'")
        return api_type

    @root_validator(pre=False)
    def validate_field_compatibility(cls, config: "OpenAIConfig"):
        api_type = config["openai_api_type"]
        if api_type == OpenAIAPIType.OPENAI:
            if config.get("openai_deployment_name") is not None:
                raise MlflowException.invalid_parameter_value(
                    f"OpenAI route configuration can only specify a value for "
                    f"'openai_deployment_name' if 'openai_api_type' is '{OpenAIAPIType.AZURE}' "
                    f"or '{OpenAIAPIType.AZUREAD}'. Found type: '{api_type}'"
                )
            if config.get("openai_api_base") is None:
                config["openai_api_base"] = "https://api.openai.com/v1"
        elif api_type in (OpenAIAPIType.AZURE, OpenAIAPIType.AZUREAD):
            if config.get("openai_organization") is not None:
                raise MlflowException.invalid_parameter_value(
                    f"OpenAI route configuration can only specify a value for "
                    f"'openai_organization' if 'openai_api_type' is '{OpenAIAPIType.OPENAI}'"
                )
            base_url = config.get("openai_api_base")
            deployment_name = config.get("openai_deployment_name")
            api_version = config.get("openai_api_version")
            if (base_url, deployment_name, api_version).count(None) > 0:
                raise MlflowException.invalid_parameter_value(
                    f"OpenAI route configuration must specify 'openai_api_base', "
                    f"'openai_deployment_name', and 'openai_api_version' if 'openai_api_type' is "
                    f"'{OpenAIAPIType.AZURE}' or '{OpenAIAPIType.AZUREAD}'."
                )
        else:
            raise MlflowException.invalid_parameter_value(f"Invalid OpenAI API type '{api_type}'")

        return config


class AnthropicConfig(BaseModel, extra=Extra.allow):
    anthropic_api_key: str
    anthropic_api_base: str = "https://api.anthropic.com/"

    # pylint: disable=no-self-argument
    @validator("anthropic_api_key", pre=True)
    def validate_anthropic_api_key(cls, value):
        return _resolve_api_key_from_input(value)


config_types = {
    Provider.COHERE: CohereConfig,
    Provider.OPENAI: OpenAIConfig,
    Provider.ANTHROPIC: AnthropicConfig,
}


class ModelInfo(BaseModel, extra=Extra.allow):
    name: Optional[str] = None
    provider: Provider


def _resolve_api_key_from_input(api_key_input):
    """
    Resolves the provided API key.

    Input formats accepted:

    - Path to a file as a string which will have the key loaded from it
    - environment variable name that stores the api key
    - the api key itself
    """

    if not isinstance(api_key_input, str):
        raise MlflowException.invalid_parameter_value(
            "The api key provided is not a string. Please provide either an environment "
            "variable key, a path to a file containing the api key, or the api key itself"
        )

    # try reading as an environment variable
    if api_key_input.startswith("$"):
        env_var_name = api_key_input[1:]
        if env_var := os.getenv(env_var_name):
            return env_var
        else:
            raise MlflowException.invalid_parameter_value(
                f"Environment variable {env_var_name!r} is not set"
            )

    # try reading from a local path
    file = pathlib.Path(api_key_input)
    if file.is_file():
        return file.read_text()

    # if the key itself is passed, return
    return api_key_input


# pylint: disable=no-self-argument
class Model(BaseModel, extra=Extra.allow):
    name: Optional[str] = None
    provider: Union[str, Provider]
    config: Optional[
        Union[
            CohereConfig,
            OpenAIConfig,
            AnthropicConfig,
        ]
    ] = None

    @validator("provider", pre=True)
    def validate_provider(cls, value):
        if isinstance(value, Provider):
            return value
        if value.upper() in Provider.__members__:
            return Provider[value.upper()]
        raise MlflowException.invalid_parameter_value(f"The provider '{value}' is not supported.")

    @validator("config", pre=True)
    def validate_config(cls, config, values):
        if provider := values.get("provider"):
            config_type = config_types[provider]
            return config_type(**config)

        raise MlflowException.invalid_parameter_value(
            "A provider must be provided for each gateway route."
        )


# pylint: disable=no-self-argument
class RouteConfig(BaseModel, extra=Extra.allow):
    name: str
    route_type: RouteType
    model: Model

    @validator("name")
    def validate_endpoint_name(cls, route_name):
        if not is_valid_endpoint_name(route_name):
            raise MlflowException.invalid_parameter_value(
                "The route name provided contains disallowed characters for a url endpoint. "
                f"'{route_name}' is invalid. Names cannot contain spaces or any non "
                "alphanumeric characters other than hyphen and underscore."
            )
        return route_name

    @validator("model", pre=True)
    def validate_model(cls, model):
        if model:
            model_instance = Model(**model)
            if model_instance.provider in Provider.values() and model_instance.config is None:
                raise MlflowException.invalid_parameter_value(
                    "A config must be supplied when setting a provider. The provider entry for "
                    f"{model_instance.provider} is incorrect."
                )
        return model

    @validator("route_type", pre=True)
    def validate_route_type(cls, value):
        if value in RouteType._value2member_map_:
            return value
        raise MlflowException.invalid_parameter_value(f"The route_type '{value}' is not supported.")

    def to_route(self) -> "Route":
        return Route(
            name=self.name,
            route_type=self.route_type,
            model=ModelInfo(
                name=self.model.name,
                provider=self.model.provider,
            ),
        )


class Route(BaseModel, extra=Extra.allow):
    name: str
    route_type: RouteType
    model: ModelInfo
    route_url: Optional[str] = None


class GatewayConfig(BaseModel, extra=Extra.allow):
    routes: List[RouteConfig]


def _load_route_config(path: Union[str, Path]) -> GatewayConfig:
    """
    Reads the gateway configuration yaml file from the storage location and returns an instance
    of the configuration RouteConfig class
    """
    if isinstance(path, str):
        path = Path(path)
    try:
        configuration = yaml.safe_load(path.read_text())
    except Exception as e:
        raise MlflowException.invalid_parameter_value(
            f"The file at {path} is not a valid yaml file"
        ) from e
    check_configuration_route_name_collisions(configuration)
    try:
        return parse_obj_as(GatewayConfig, configuration)
    except ValidationError as e:
        raise MlflowException.invalid_parameter_value(
            f"The gateway configuration is invalid: {e}"
        ) from e


def _save_route_config(config: GatewayConfig, path: Union[str, Path]) -> None:
    if isinstance(path, str):
        path = Path(path)
    path.write_text(yaml.safe_dump(json.loads(json.dumps(config.dict(), default=pydantic_encoder))))


def _validate_config(config_path: str) -> GatewayConfig:
    if not os.path.exists(config_path):
        raise MlflowException.invalid_parameter_value(f"{config_path} does not exist")

    try:
        return _load_route_config(config_path)
    except ValidationError as e:
        raise MlflowException.invalid_parameter_value(f"Invalid gateway configuration: {e}") from e
