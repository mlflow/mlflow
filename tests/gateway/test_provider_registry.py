import pytest

from mlflow.exceptions import MlflowException
from mlflow.gateway.config import Provider
from mlflow.gateway.provider_registry import provider_registry
from mlflow.utils.provider_filter import _PROVIDER_ALIASES


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


def test_provider_canonical_resolves_alias():
    assert Provider.AMAZON_BEDROCK._canonical() == Provider.BEDROCK
    assert Provider.BEDROCK._canonical() == Provider.BEDROCK
    assert Provider.DATABRICKS_MODEL_SERVING._canonical() == Provider.DATABRICKS
    assert Provider.DATABRICKS._canonical() == Provider.DATABRICKS
    assert Provider.OPENAI._canonical() == Provider.OPENAI


def test_provider_aliases_matches_canonical():
    assert _PROVIDER_ALIASES == {
        "amazon-bedrock": "bedrock",
        "databricks-model-serving": "databricks",
    }
