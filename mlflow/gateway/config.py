import json
import logging
import os
import pathlib
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import pydantic
import yaml
from packaging.version import Version
from pydantic import ConfigDict, Field, ValidationError
from pydantic.json import pydantic_encoder

from mlflow.exceptions import MlflowException
from mlflow.gateway.base_models import ConfigModel, LimitModel, ResponseModel
from mlflow.gateway.constants import (
    MLFLOW_AI_GATEWAY_MOSAICML_CHAT_SUPPORTED_MODEL_PREFIXES,
    MLFLOW_GATEWAY_ROUTE_BASE,
    MLFLOW_QUERY_SUFFIX,
)
from mlflow.gateway.utils import (
    check_configuration_deprecated_fields,
    check_configuration_route_name_collisions,
    is_valid_ai21labs_model,
    is_valid_endpoint_name,
    is_valid_mosiacml_chat_model,
)
from mlflow.utils.pydantic_utils import IS_PYDANTIC_V2_OR_NEWER, field_validator, model_validator

_logger = logging.getLogger(__name__)

if IS_PYDANTIC_V2_OR_NEWER:
    from pydantic import SerializeAsAny


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    AI21LABS = "ai21labs"
    MLFLOW_MODEL_SERVING = "mlflow-model-serving"
    MOSAICML = "mosaicml"
    HUGGINGFACE_TEXT_GENERATION_INFERENCE = "huggingface-text-generation-inference"
    PALM = "palm"
    GEMINI = "gemini"
    BEDROCK = "bedrock"
    AMAZON_BEDROCK = "amazon-bedrock"  # an alias for bedrock
    # Note: The following providers are only supported on Databricks
    DATABRICKS_MODEL_SERVING = "databricks-model-serving"
    DATABRICKS = "databricks"
    MISTRAL = "mistral"
    TOGETHERAI = "togetherai"

    @classmethod
    def values(cls):
        return {p.value for p in cls}


class TogetherAIConfig(ConfigModel):
    togetherai_api_key: str

    @field_validator("togetherai_api_key", mode="before")
    def validate_togetherai_api_key(cls, value):
        return _resolve_api_key_from_input(value)


class RouteType(str, Enum):
    LLM_V1_COMPLETIONS = "llm/v1/completions"
    LLM_V1_CHAT = "llm/v1/chat"
    LLM_V1_EMBEDDINGS = "llm/v1/embeddings"


class CohereConfig(ConfigModel):
    cohere_api_key: str

    @field_validator("cohere_api_key", mode="before")
    def validate_cohere_api_key(cls, value):
        return _resolve_api_key_from_input(value)


class AI21LabsConfig(ConfigModel):
    ai21labs_api_key: str

    @field_validator("ai21labs_api_key", mode="before")
    def validate_ai21labs_api_key(cls, value):
        return _resolve_api_key_from_input(value)


class MosaicMLConfig(ConfigModel):
    mosaicml_api_key: str
    mosaicml_api_base: Optional[str] = None

    @field_validator("mosaicml_api_key", mode="before")
    def validate_mosaicml_api_key(cls, value):
        return _resolve_api_key_from_input(value)


class OpenAIAPIType(str, Enum):
    OPENAI = "openai"
    AZURE = "azure"
    AZUREAD = "azuread"

    @classmethod
    def _missing_(cls, value):
        """
        Implements case-insensitive matching of API type strings
        """
        for api_type in cls:
            if api_type.value == value.lower():
                return api_type

        raise MlflowException.invalid_parameter_value(f"Invalid OpenAI API type '{value}'")


class OpenAIConfig(ConfigModel):
    openai_api_key: str
    openai_api_type: OpenAIAPIType = OpenAIAPIType.OPENAI
    openai_api_base: Optional[str] = None
    openai_api_version: Optional[str] = None
    openai_deployment_name: Optional[str] = None
    openai_organization: Optional[str] = None

    @field_validator("openai_api_key", mode="before")
    def validate_openai_api_key(cls, value):
        return _resolve_api_key_from_input(value)

    @classmethod
    def _validate_field_compatibility(cls, info: dict[str, Any]):
        if not isinstance(info, dict):
            return info
        api_type = (info.get("openai_api_type") or OpenAIAPIType.OPENAI).lower()
        if api_type == OpenAIAPIType.OPENAI:
            if info.get("openai_deployment_name") is not None:
                raise MlflowException.invalid_parameter_value(
                    f"OpenAI route configuration can only specify a value for "
                    f"'openai_deployment_name' if 'openai_api_type' is '{OpenAIAPIType.AZURE}' "
                    f"or '{OpenAIAPIType.AZUREAD}'. Found type: '{api_type}'"
                )
            if info.get("openai_api_base") is None:
                info["openai_api_base"] = "https://api.openai.com/v1"
        elif api_type in (OpenAIAPIType.AZURE, OpenAIAPIType.AZUREAD):
            if info.get("openai_organization") is not None:
                raise MlflowException.invalid_parameter_value(
                    f"OpenAI route configuration can only specify a value for "
                    f"'openai_organization' if 'openai_api_type' is '{OpenAIAPIType.OPENAI}'"
                )
            base_url = info.get("openai_api_base")
            deployment_name = info.get("openai_deployment_name")
            api_version = info.get("openai_api_version")
            if (base_url, deployment_name, api_version).count(None) > 0:
                raise MlflowException.invalid_parameter_value(
                    f"OpenAI route configuration must specify 'openai_api_base', "
                    f"'openai_deployment_name', and 'openai_api_version' if 'openai_api_type' is "
                    f"'{OpenAIAPIType.AZURE}' or '{OpenAIAPIType.AZUREAD}'."
                )
        else:
            raise MlflowException.invalid_parameter_value(f"Invalid OpenAI API type '{api_type}'")

        return info

    @model_validator(mode="before")
    def validate_field_compatibility(cls, info: dict[str, Any]):
        return cls._validate_field_compatibility(info)


class AnthropicConfig(ConfigModel):
    anthropic_api_key: str
    anthropic_version: str = "2023-06-01"

    @field_validator("anthropic_api_key", mode="before")
    def validate_anthropic_api_key(cls, value):
        return _resolve_api_key_from_input(value)


class PaLMConfig(ConfigModel):
    palm_api_key: str

    @field_validator("palm_api_key", mode="before")
    def validate_palm_api_key(cls, value):
        return _resolve_api_key_from_input(value)


class GeminiConfig(ConfigModel):
    gemini_api_key: str

    @field_validator("gemini_api_key", mode="before")
    def validate_gemini_api_key(cls, value):
        return _resolve_api_key_from_input(value)


class MlflowModelServingConfig(ConfigModel):
    model_server_url: str

    # Workaround to suppress warning that Pydantic raises when a field name starts with "model_".
    # https://github.com/mlflow/mlflow/issues/10335
    model_config = pydantic.ConfigDict(protected_namespaces=())


class HuggingFaceTextGenerationInferenceConfig(ConfigModel):
    hf_server_url: str


class AWSBaseConfig(pydantic.BaseModel):
    aws_region: Optional[str] = None


class AWSRole(AWSBaseConfig):
    aws_role_arn: str
    session_length_seconds: int = 15 * 60


class AWSIdAndKey(AWSBaseConfig):
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_session_token: Optional[str] = None


class AmazonBedrockConfig(ConfigModel):
    # order here is important, at least for pydantic<2
    aws_config: Union[AWSRole, AWSIdAndKey, AWSBaseConfig]


class MistralConfig(ConfigModel):
    mistral_api_key: str

    @field_validator("mistral_api_key", mode="before")
    def validate_mistral_api_key(cls, value):
        return _resolve_api_key_from_input(value)


class ModelInfo(ResponseModel):
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
    try:
        if file.is_file():
            return file.read_text()
    except OSError:
        # `is_file` throws an OSError if `api_key_input` exceeds the maximum filename length
        # (e.g., 255 characters on Unix).
        pass

    # if the key itself is passed, return
    return api_key_input


class Model(ConfigModel):
    name: Optional[str] = None
    provider: Union[str, Provider]
    if IS_PYDANTIC_V2_OR_NEWER:
        config: Optional[SerializeAsAny[ConfigModel]] = None
    else:
        config: Optional[ConfigModel] = None

    @field_validator("provider", mode="before")
    def validate_provider(cls, value):
        from mlflow.gateway.provider_registry import provider_registry

        if isinstance(value, Provider):
            return value
        formatted_value = value.replace("-", "_").upper()
        if formatted_value in Provider.__members__:
            return Provider[formatted_value]
        if value in provider_registry.keys():
            return value
        raise MlflowException.invalid_parameter_value(f"The provider '{value}' is not supported.")

    @classmethod
    def _validate_config(cls, val, context):
        from mlflow.gateway.provider_registry import provider_registry

        # For Pydantic v2: 'context' is a ValidationInfo object with a 'data' attribute.
        # For Pydantic v1: 'context' is dict-like 'values'.
        if IS_PYDANTIC_V2_OR_NEWER:
            provider = context.data.get("provider")
        else:
            provider = context.get("provider") if context else None

        if provider:
            config_type = provider_registry.get(provider).CONFIG_TYPE
            return config_type(**val) if isinstance(val, dict) else val
        raise MlflowException.invalid_parameter_value(
            "A provider must be provided for each gateway route."
        )

    @field_validator("config", mode="before")
    def validate_config(cls, info, values):
        return cls._validate_config(info, values)


class AliasedConfigModel(ConfigModel):
    """
    Enables use of field aliases in a configuration model for backwards compatibility
    """

    if Version(pydantic.__version__) >= Version("2.0"):
        model_config = ConfigDict(populate_by_name=True)
    else:

        class Config:
            allow_population_by_field_name = True


class Limit(LimitModel):
    calls: int
    key: Optional[str] = None
    renewal_period: str


class LimitsConfig(ConfigModel):
    limits: Optional[list[Limit]] = []


class RouteConfig(AliasedConfigModel):
    name: str
    route_type: RouteType = Field(alias="endpoint_type")
    model: Model
    limit: Optional[Limit] = None

    @field_validator("name")
    def validate_endpoint_name(cls, route_name):
        if not is_valid_endpoint_name(route_name):
            raise MlflowException.invalid_parameter_value(
                "The route name provided contains disallowed characters for a url endpoint. "
                f"'{route_name}' is invalid. Names cannot contain spaces or any non "
                "alphanumeric characters other than hyphen and underscore."
            )
        return route_name

    @field_validator("model", mode="before")
    def validate_model(cls, model):
        if model:
            model_instance = Model(**model)
            if model_instance.provider in Provider.values() and model_instance.config is None:
                raise MlflowException.invalid_parameter_value(
                    "A config must be supplied when setting a provider. The provider entry for "
                    f"{model_instance.provider} is incorrect."
                )
        return model

    @model_validator(mode="after", skip_on_failure=True)
    def validate_route_type_and_model_name(cls, values):
        if IS_PYDANTIC_V2_OR_NEWER:
            route_type = values.route_type
            model = values.model
        else:
            route_type = values.get("route_type")
            model = values.get("model")
        if (
            model
            and model.provider == "mosaicml"
            and route_type == RouteType.LLM_V1_CHAT
            and not is_valid_mosiacml_chat_model(model.name)
        ):
            raise MlflowException.invalid_parameter_value(
                f"An invalid model has been specified for the chat route. '{model.name}'. "
                f"Ensure the model selected starts with one of: "
                f"{MLFLOW_AI_GATEWAY_MOSAICML_CHAT_SUPPORTED_MODEL_PREFIXES}"
            )
        if model and model.provider == "ai21labs" and not is_valid_ai21labs_model(model.name):
            raise MlflowException.invalid_parameter_value(
                f"An Unsupported AI21Labs model has been specified: '{model.name}'. "
                f"Please see documentation for supported models."
            )
        return values

    @field_validator("route_type", mode="before")
    def validate_route_type(cls, value):
        if value in RouteType._value2member_map_:
            return value
        raise MlflowException.invalid_parameter_value(f"The route_type '{value}' is not supported.")

    @field_validator("limit", mode="before")
    def validate_limit(cls, value):
        from limits import parse

        if value:
            limit = Limit(**value)
            try:
                parse(f"{limit.calls}/{limit.renewal_period}")
            except ValueError:
                raise MlflowException.invalid_parameter_value(
                    "Failed to parse the rate limit configuration."
                    "Please make sure limit.calls is a positive number and"
                    "limit.renewal_period is a right granularity"
                )

        return value

    def to_route(self) -> "Route":
        return Route(
            name=self.name,
            route_type=self.route_type,
            model=RouteModelInfo(
                name=self.model.name,
                provider=self.model.provider,
            ),
            route_url=f"{MLFLOW_GATEWAY_ROUTE_BASE}{self.name}{MLFLOW_QUERY_SUFFIX}",
            limit=self.limit,
        )


class RouteModelInfo(ResponseModel):
    name: Optional[str] = None
    # Use `str` instead of `Provider` enum to allow gateway backends such as Databricks to
    # support new providers without breaking the gateway client.
    provider: str


_ROUTE_EXTRA_SCHEMA = {
    "example": {
        "name": "openai-completions",
        "route_type": "llm/v1/completions",
        "model": {
            "name": "gpt-4o-mini",
            "provider": "openai",
        },
        "route_url": "/gateway/routes/completions/invocations",
    }
}


class Route(ConfigModel):
    name: str
    route_type: str
    model: RouteModelInfo
    route_url: str
    limit: Optional[Limit] = None

    class Config:
        if IS_PYDANTIC_V2_OR_NEWER:
            json_schema_extra = _ROUTE_EXTRA_SCHEMA
        else:
            schema_extra = _ROUTE_EXTRA_SCHEMA

    def to_endpoint(self):
        from mlflow.deployments.server.config import Endpoint

        return Endpoint(
            name=self.name,
            endpoint_type=self.route_type,
            model=self.model,
            endpoint_url=self.route_url,
            limit=self.limit,
        )


class GatewayConfig(AliasedConfigModel):
    routes: list[RouteConfig] = Field(alias="endpoints")


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
    check_configuration_deprecated_fields(configuration)
    check_configuration_route_name_collisions(configuration)
    try:
        return GatewayConfig(**configuration)
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
