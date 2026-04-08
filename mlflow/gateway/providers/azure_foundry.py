"""
Azure AI Foundry provider for MLflow AI Gateway.

Azure AI Foundry (formerly Azure AI Studio) provides serverless API endpoints
for various models that are OpenAI-compatible. This provider handles the
Azure-specific authentication (api-key header) and URL patterns.
"""

from typing import Any

from mlflow.gateway.config import AzureFoundryConfig, EndpointConfig
from mlflow.gateway.providers.base import ProviderAdapter
from mlflow.gateway.providers.openai_compatible import (
    OpenAICompatibleAdapter,
    OpenAICompatibleProvider,
)


class AzureFoundryAdapter(OpenAICompatibleAdapter):
    @classmethod
    def chat_to_model(cls, payload: dict[str, Any], config: EndpointConfig) -> dict[str, Any]:
        # Azure AI Foundry endpoints determine the model from the URL,
        # so we don't inject the model name into the payload.
        return payload

    @classmethod
    def embeddings_to_model(cls, payload: dict[str, Any], config: EndpointConfig) -> dict[str, Any]:
        return payload


class AzureFoundryProvider(OpenAICompatibleProvider):
    DISPLAY_NAME = "Azure AI Foundry"
    CONFIG_TYPE = AzureFoundryConfig

    def __init__(self, config: EndpointConfig, enable_tracing: bool = False) -> None:
        super().__init__(config, enable_tracing=enable_tracing)
        self._foundry_config: AzureFoundryConfig = config.model.config

    def get_provider_name(self) -> str:
        return "azure_foundry"

    @property
    def _api_base(self) -> str:
        return self._foundry_config.azure_api_base

    @property
    def _api_key(self) -> str:
        return self._foundry_config.azure_api_key

    @property
    def headers(self) -> dict[str, str]:
        return {"api-key": self._api_key}

    @property
    def adapter_class(self) -> type[ProviderAdapter]:
        return AzureFoundryAdapter
