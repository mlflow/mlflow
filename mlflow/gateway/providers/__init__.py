from typing import Type

from mlflow.gateway.config import Provider
from mlflow.gateway.providers.base import BaseProvider


def get_provider(provider: Provider) -> Type[BaseProvider]:
    from mlflow.gateway.provider_registry import provider_registry

    return provider_registry.get(provider)
