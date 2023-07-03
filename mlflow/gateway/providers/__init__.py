from mlflow.exceptions import MlflowException
from .base import BaseProvider
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from .cohere import CohereProvider
from ..config import Provider


def get_provider(provider: Provider) -> BaseProvider:
    provider_to_class = {
        Provider.OPENAI: OpenAIProvider,
        Provider.ANTHROPIC: AnthropicProvider,
        Provider.COHERE: CohereProvider,
    }
    if prov := provider_to_class.get(provider):
        return prov

    raise MlflowException.invalid_parameter_value(f"Provider {provider} not found")
