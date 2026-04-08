import pytest

from mlflow.exceptions import MlflowException
from mlflow.gateway.provider_registry import provider_registry


def test_registry_keys_returns_all_providers_by_default():
    keys = provider_registry.keys()
    assert len(keys) > 0
    assert "openai" in keys
    assert "anthropic" in keys


def test_registry_get_returns_provider_by_default():
    provider_class = provider_registry.get("openai")
    assert provider_class is not None


def test_registry_get_rejects_provider_not_in_allowed_list(monkeypatch):
    monkeypatch.setenv("MLFLOW_GATEWAY_ALLOWED_PROVIDERS", "openai")
    with pytest.raises(MlflowException, match="not allowed"):
        provider_registry.get("litellm")


def test_registry_get_allows_provider_in_allowed_list(monkeypatch):
    monkeypatch.setenv("MLFLOW_GATEWAY_ALLOWED_PROVIDERS", "openai")
    provider_class = provider_registry.get("openai")
    assert provider_class is not None


def test_keys_unfiltered_even_with_allowed_list(monkeypatch):
    monkeypatch.setenv("MLFLOW_GATEWAY_ALLOWED_PROVIDERS", "anthropic")
    keys = provider_registry.keys()
    assert "openai" in keys
    assert "anthropic" in keys
    assert "litellm" in keys
