import asyncio
import subprocess
import sys

import pytest
from fastapi import HTTPException

from mlflow.gateway.providers.litellm_utils import prepare_litellm_kwargs
from mlflow.gateway.ssrf import upstream_ssrf_protection


def _prepare(payload, auth_config, protected):
    token = upstream_ssrf_protection.set(protected)
    try:
        return asyncio.run(prepare_litellm_kwargs(payload, auth_config))
    finally:
        upstream_ssrf_protection.reset(token)


@pytest.fixture(autouse=True)
def private_upstreams_disabled(monkeypatch):
    monkeypatch.delenv("MLFLOW_GATEWAY_API_BASE_ALLOW_PRIVATE_IPS", raising=False)


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


@pytest.mark.parametrize("protected", [False, True])
def test_destination_request_guard(protected):
    with pytest.raises(HTTPException, match="The request must not set 'api_base'") as exc:
        _prepare({"api_base": "https://api.example.com"}, None, protected)
    assert exc.value.status_code == 400


def test_private_upstream_guard():
    with pytest.raises(HTTPException, match="not a public IP address") as exc:
        _prepare({}, {"api_base": "https://127.0.0.1"}, protected=True)
    assert exc.value.status_code == 502


def test_public_base_url_guard_preserves_normalization():
    auth_config = {"base_url": "https://8.8.8.8", "api_key": "configured-key"}
    assert _prepare({}, auth_config, protected=True) == {
        "api_base": "https://8.8.8.8",
        "api_key": "configured-key",
    }
    assert auth_config == {"base_url": "https://8.8.8.8", "api_key": "configured-key"}


@pytest.mark.parametrize("protected", [False, True])
@pytest.mark.parametrize("allow_private_ips", [False, True])
def test_oauth_request_guard(protected, allow_private_ips, monkeypatch):
    monkeypatch.setenv("MLFLOW_GATEWAY_API_BASE_ALLOW_PRIVATE_IPS", str(allow_private_ips))
    with pytest.raises(HTTPException, match="The request must not set 'gigachat_auth_url'") as exc:
        _prepare(
            {"gigachat_auth_url": "https://oauth.example.com/token"},
            {"api_key": "configured-key"},
            protected,
        )
    assert exc.value.status_code == 400


def test_oauth_stored_destination_guard():
    with pytest.raises(HTTPException, match="secret sets 'gigachat_auth_url'") as exc:
        _prepare(
            {},
            {"gigachat_auth_url": "https://oauth.example.com/token", "api_key": "configured-key"},
            protected=True,
        )
    assert exc.value.status_code == 502


@pytest.mark.parametrize("protected", [False, True])
def test_oauth_guard_preserves_credentials_and_inference_parameters(protected):
    payload = {"messages": [], "temperature": 0.5, "top_k": 2, "api_key": "caller-key"}
    auth_config = {"api_key": "configured-key", "gigachat_scope": "configured-scope"}
    assert _prepare(payload, auth_config, protected) == payload | auth_config
    assert payload["api_key"] == "caller-key"
    assert auth_config == {"api_key": "configured-key", "gigachat_scope": "configured-scope"}


def test_oauth_guard_preserves_operator_owned_standalone_configuration():
    auth_config = {
        "api_key": "configured-key",
        "gigachat_auth_url": "https://oauth.example.com/token",
    }
    assert _prepare({}, auth_config, protected=False) == auth_config
