from ..config import Provider

from mlflow.exceptions import MlflowException
from .base import BaseProvider
from .openai import OpenAIProvider


def get_provider(provider: Provider) -> BaseProvider:
    provider_to_class = {
        Provider.OPENAI: OpenAIProvider,
    }
    if prov := provider_to_class.get(provider):
        return prov

    raise MlflowException.invalid_parameter_value(f"Provider {provider} not found")
