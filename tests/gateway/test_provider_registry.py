import pytest

from mlflow.exceptions import MlflowException
from mlflow.gateway.provider_registry import is_supported_provider, provider_registry


def test_registry_keys_returns_all_providers_by_default():
    keys = provider_registry.keys()
    assert len(keys) > 0
    assert "openai" in keys
    assert "anthropic" in keys


def test_registry_get_returns_provider_by_default():
    provider_class = provider_registry.get("openai")
    assert provider_class is not None


def test_registry_keys_filters_with_allowed_list(monkeypatch):
    monkeypatch.setenv("MLFLOW_GATEWAY_ALLOWED_PROVIDERS", "openai,anthropic")
    keys = provider_registry.keys()
    assert "openai" in keys
    assert "anthropic" in keys
    assert "litellm" not in keys
    assert "bedrock" not in keys


def test_registry_keys_filters_with_blocked_list(monkeypatch):
    monkeypatch.setenv("MLFLOW_GATEWAY_BLOCKED_PROVIDERS", "litellm,bedrock")
    keys = provider_registry.keys()
    assert "openai" in keys
    assert "anthropic" in keys
    assert "litellm" not in keys
    assert "bedrock" not in keys


def test_registry_get_rejects_blocked_provider(monkeypatch):
    monkeypatch.setenv("MLFLOW_GATEWAY_BLOCKED_PROVIDERS", "litellm")
    with pytest.raises(MlflowException, match="not allowed"):
        provider_registry.get("litellm")


def test_registry_get_rejects_provider_not_in_allowed_list(monkeypatch):
    monkeypatch.setenv("MLFLOW_GATEWAY_ALLOWED_PROVIDERS", "openai")
    with pytest.raises(MlflowException, match="not allowed"):
        provider_registry.get("litellm")


def test_registry_get_allows_provider_in_allowed_list(monkeypatch):
    monkeypatch.setenv("MLFLOW_GATEWAY_ALLOWED_PROVIDERS", "openai")
    provider_class = provider_registry.get("openai")
    assert provider_class is not None


def test_is_supported_provider_respects_blocked_list(monkeypatch):
    assert is_supported_provider("openai") is True
    monkeypatch.setenv("MLFLOW_GATEWAY_BLOCKED_PROVIDERS", "openai")
    assert is_supported_provider("openai") is False
    assert is_supported_provider("anthropic") is True


def test_is_supported_provider_respects_allowed_list(monkeypatch):
    monkeypatch.setenv("MLFLOW_GATEWAY_ALLOWED_PROVIDERS", "openai")
    assert is_supported_provider("openai") is True
    assert is_supported_provider("litellm") is False
