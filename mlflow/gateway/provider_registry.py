from typing import Type

from pydantic import BaseModel

from mlflow import MlflowException
from mlflow.gateway.base_models import ConfigModel
from mlflow.gateway.providers import BaseProvider
from mlflow.utils.os import get_entry_points


class ProviderEntry(BaseModel):
    provider: Type[BaseProvider]
    config: Type[ConfigModel]


class ProviderRegistry:
    def __init__(self):
        self._providers = {}

    def register(self, name: str, provider: Type[BaseProvider], config: Type[ConfigModel]):
        if name in self._providers:
            raise MlflowException.invalid_parameter_value(f"Provider {name} already registered")
        self._providers[name] = ProviderEntry(provider=provider, config=config)

    def get(self, name: str) -> ProviderEntry:
        if name not in self._providers:
            raise MlflowException.invalid_parameter_value(f"Provider {name} not found")
        return self._providers[name]


def _register_default_providers(registry: ProviderRegistry):
    from mlflow.gateway.config import (
        AI21LabsConfig,
        AmazonBedrockConfig,
        AnthropicConfig,
        CohereConfig,
        HuggingFaceTextGenerationInferenceConfig,
        MistralConfig,
        MlflowModelServingConfig,
        MosaicMLConfig,
        OpenAIConfig,
        PaLMConfig,
        Provider,
        TogetherAIConfig,
    )
    from mlflow.gateway.providers.ai21labs import AI21LabsProvider
    from mlflow.gateway.providers.anthropic import AnthropicProvider
    from mlflow.gateway.providers.bedrock import AmazonBedrockProvider
    from mlflow.gateway.providers.cohere import CohereProvider
    from mlflow.gateway.providers.huggingface import (
        HFTextGenerationInferenceServerProvider,
    )
    from mlflow.gateway.providers.mistral import MistralProvider
    from mlflow.gateway.providers.mlflow import MlflowModelServingProvider
    from mlflow.gateway.providers.mosaicml import MosaicMLProvider
    from mlflow.gateway.providers.openai import OpenAIProvider
    from mlflow.gateway.providers.palm import PaLMProvider
    from mlflow.gateway.providers.togetherai import TogetherAIProvider

    registry.register(Provider.OPENAI, OpenAIProvider, OpenAIConfig)
    registry.register(Provider.ANTHROPIC, AnthropicProvider, AnthropicConfig)
    registry.register(Provider.COHERE, CohereProvider, CohereConfig)
    registry.register(Provider.AI21LABS, AI21LabsProvider, AI21LabsConfig)
    registry.register(Provider.MOSAICML, MosaicMLProvider, MosaicMLConfig)
    registry.register(Provider.PALM, PaLMProvider, PaLMConfig)
    registry.register(
        Provider.MLFLOW_MODEL_SERVING,
        MlflowModelServingProvider,
        MlflowModelServingConfig,
    )
    registry.register(Provider.BEDROCK, AmazonBedrockProvider, AmazonBedrockConfig)
    registry.register(Provider.AMAZON_BEDROCK, AmazonBedrockProvider, AmazonBedrockConfig)
    registry.register(
        Provider.HUGGINGFACE_TEXT_GENERATION_INFERENCE,
        HFTextGenerationInferenceServerProvider,
        HuggingFaceTextGenerationInferenceConfig,
    )
    registry.register(Provider.MISTRAL, MistralProvider, MistralConfig)
    registry.register(Provider.TOGETHERAI, TogetherAIProvider, TogetherAIConfig)


def _register_plugin_providers(registry: ProviderRegistry):
    providers = get_entry_points("mlflow.gateway.providers")
    for p in providers:
        cls = p.load()
        # todo
        from mlflow.gateway.config import OpenAIConfig

        registry.register(p.name, cls, OpenAIConfig)


provider_registry = ProviderRegistry()
_register_default_providers(provider_registry)
_register_plugin_providers(provider_registry)
