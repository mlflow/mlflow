from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field, validator, parse_obj_as
from pydantic.json import pydantic_encoder
from typing import Optional, Union, List
import yaml
import json


class Provider(str, Enum):
    UNSPECIFIED_PROVIDER = "unspecified"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    BARD = "bard"


class RouteType(str, Enum):
    UNSPECIFIED = "unspecified"
    LLM_V1_INSTRUCT = "llm/v1/instruct"
    LLM_V1_CHAT = "llm/v1/chat"


class OpenAIConfig(BaseModel):
    openai_api_key: str = Field(..., alias="openai_api_key")
    openai_api_type: Optional[str] = Field(None, alias="openai_api_type")
    openai_api_base: Optional[str] = Field("https://api.openai.com/", alias="openai_api_base")
    openai_api_version: Optional[str] = Field(None, alias="openai_api_version")


class AnthropicConfig(BaseModel):
    anthropic_api_key: str = Field(..., alias="anthropic_api_key")
    anthropic_api_base: Optional[str] = Field(
        "https://api.anthropic.com/", alias="anthropic_api_base"
    )


class BardConfig(BaseModel):
    bard_api_key: str = Field(..., alias="bard_api_key")
    bard_api_base: str = Field("https://bard.google.com/", alias="bard_api_base")


class ModelInfo(BaseModel):
    name: Optional[str] = None
    provider: Provider = Provider.UNSPECIFIED_PROVIDER


class Model(BaseModel):
    name: Optional[str] = None
    provider: Provider = Provider.UNSPECIFIED_PROVIDER
    config: Optional[Union[OpenAIConfig, AnthropicConfig, BardConfig]] = None

    @validator("provider", pre=True)
    def validate_provider(cls, value):
        if value in Provider._value2member_map_:
            return value
        return Provider.UNSPECIFIED_PROVIDER.value


class RouteConfig(BaseModel):
    name: Optional[str] = None
    type: RouteType = RouteType.UNSPECIFIED
    model: Optional[Model] = None

    @validator("type", pre=True)
    def validate_route_type(cls, value):
        if value in RouteType._value2member_map_:
            return value
        return RouteType.UNSPECIFIED.value


class Route(BaseModel):
    name: str
    type: RouteType
    model: ModelInfo


def _load_gateway_config(path: Union[str, Path]) -> List[RouteConfig]:
    """
    Reads the gateway configuration yaml file from the storage location and returns an instance
    of the configuration _GatewayConfig class
    """
    if isinstance(path, str):
        path = Path(path)
    configuration = yaml.safe_load(path.read_text())
    return parse_obj_as(List[RouteConfig], configuration)


def _save_gateway_config(config: List[RouteConfig], path: Union[str, Path]):
    if isinstance(path, str):
        path = Path(path)
    serialized = [
        json.loads(json.dumps(route.dict(), default=pydantic_encoder)) for route in config
    ]
    path.write_text(yaml.safe_dump(serialized))


def _route_config_to_route(route: RouteConfig) -> Route:
    return Route(
        name=route.name,
        type=route.type,
        model=ModelInfo(
            name=route.model.name,
            provider=route.model.provider,
        ),
    )


def _convert_route_config_to_route(route_config: List[RouteConfig]) -> List[Route]:
    return [_route_config_to_route(route) for route in route_config]
