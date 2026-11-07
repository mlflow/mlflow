import subprocess
import sys

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


def test_native_provider_import_guard_without_optional_transports():
    subprocess.check_call([
        sys.executable,
        "-c",
        """
import importlib.abc
import sys


class BlockOptionalTransports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"aiohttp", "litellm"}:
            raise ModuleNotFoundError(f"Optional transport blocked: {fullname}", name=fullname)


sys.meta_path.insert(0, BlockOptionalTransports())

from mlflow.gateway.provider_registry import provider_registry

assert "openai" in provider_registry.keys()
assert "aiohttp" not in sys.modules
assert "litellm" not in sys.modules
""",
    ])
