from typing import Union

from mlflow import MlflowException
from mlflow.gateway.config import Provider
from mlflow.gateway.providers import BaseProvider
from mlflow.utils.plugins import get_entry_points


class ProviderRegistry:
    def __init__(self):
        self._providers: dict[Union[str, Provider], type[BaseProvider]] = {}

    def register(self, name: str, provider: type[BaseProvider]):
        if name in self._providers:
            raise MlflowException.invalid_parameter_value(
                f"Provider {name} is already registered: {self._providers[name]}"
            )
        self._providers[name] = provider

    def get(self, name: str) -> type[BaseProvider]:
        if name not in self._providers:
            raise MlflowException.invalid_parameter_value(f"Provider {name} not found")
        return self._providers[name]

    def keys(self):
        return list(self._providers.keys())


def _register_default_providers(registry: ProviderRegistry):
    from mlflow.gateway.providers.ai21labs import AI21LabsProvider
    from mlflow.gateway.providers.anthropic import AnthropicProvider
    from mlflow.gateway.providers.bedrock import AmazonBedrockProvider
    from mlflow.gateway.providers.cohere import CohereProvider
    from mlflow.gateway.providers.gemini import GeminiProvider
    from mlflow.gateway.providers.huggingface import HFTextGenerationInferenceServerProvider
    from mlflow.gateway.providers.mistral import MistralProvider
    from mlflow.gateway.providers.mlflow import MlflowModelServingProvider
    from mlflow.gateway.providers.mosaicml import MosaicMLProvider
    from mlflow.gateway.providers.openai import OpenAIProvider
    from mlflow.gateway.providers.palm import PaLMProvider
    from mlflow.gateway.providers.togetherai import TogetherAIProvider

    registry.register(Provider.OPENAI, OpenAIProvider)
    registry.register(Provider.ANTHROPIC, AnthropicProvider)
    registry.register(Provider.COHERE, CohereProvider)
    registry.register(Provider.AI21LABS, AI21LabsProvider)
    registry.register(Provider.MOSAICML, MosaicMLProvider)
    registry.register(Provider.PALM, PaLMProvider)
    registry.register(Provider.GEMINI, GeminiProvider)
    registry.register(Provider.MLFLOW_MODEL_SERVING, MlflowModelServingProvider)
    registry.register(Provider.BEDROCK, AmazonBedrockProvider)
    registry.register(Provider.AMAZON_BEDROCK, AmazonBedrockProvider)
    registry.register(
        Provider.HUGGINGFACE_TEXT_GENERATION_INFERENCE, HFTextGenerationInferenceServerProvider
    )
    registry.register(Provider.MISTRAL, MistralProvider)
    registry.register(Provider.TOGETHERAI, TogetherAIProvider)


def _register_plugin_providers(registry: ProviderRegistry):
    providers = get_entry_points("mlflow.gateway.providers")
    for p in providers:
        cls = p.load()
        registry.register(p.name, cls)


def is_supported_provider(name: str) -> bool:
    return name in provider_registry.keys()


provider_registry = ProviderRegistry()
_register_default_providers(provider_registry)
_register_plugin_providers(provider_registry)
