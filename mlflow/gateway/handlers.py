from copy import deepcopy
from dataclasses import dataclass
from dataclasses_json import DataClassJsonMixin, dataclass_json
import json
from typing import List, Optional
import yaml


class ObfuscatedDataClassJsonMixin(DataClassJsonMixin):
    def copy(self):
        # Make a copy of the dataclass
        return deepcopy(self)

    def to_json(self, *args, **kwargs):
        # Make a copy of the dataclass with obfuscated api_key
        copy = self.copy()

        # Serialize the copy
        return json.dumps(copy.to_dict(), *args, **kwargs)

    def to_dict(self, *args, **kwargs):
        # Make a copy of the dataclass with obfuscated api_key
        copy = self.copy()
        return DataClassJsonMixin.to_dict(copy, *args, **kwargs)


@dataclass
class _ResourceConfig(ObfuscatedDataClassJsonMixin):
    api_key: Optional[str] = None
    api_type: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None

    def copy(self):
        return _ResourceConfig(
            api_key="***REDACTED***" if self.api_key else None,
            api_type=self.api_type,
            api_base=self.api_base,
            api_version=self.api_version,
        )


@dataclass
class _Resource(ObfuscatedDataClassJsonMixin):
    provider: str
    config: _ResourceConfig
    models: List[str]

    def copy(self):
        return _Resource(
            provider=self.provider,
            config=self.config.copy(),
            models=list(self.models),
        )


@dataclass
class _Route(ObfuscatedDataClassJsonMixin):
    type: str
    resources: List[_Resource]

    def copy(self):
        return _Route(
            type=self.type,
            resources=[resource.copy() for resource in self.resources],
        )


@dataclass
class _GatewayConfig(ObfuscatedDataClassJsonMixin):
    routes: List[_Route]

    def copy(self):
        return _GatewayConfig(
            routes=[route.copy() for route in self.routes],
        )


@dataclass_json
@dataclass
class _ResourceConfig:
    """
    Payload return object for GetRoute and SearchRoutes APIs
    """

    name: str
    type: Optional[str] = None
    provider: Optional[str] = None


def _load_gateway_config(path: str):
    """
    Reads the gateway configuration yaml file from the storage location and returns an instance
    of the configuration _GatewayConfig class
    """
    with open(path, "r") as f:
        configuration = yaml.safe_load(f)
    return _GatewayConfig.from_dict(configuration)
