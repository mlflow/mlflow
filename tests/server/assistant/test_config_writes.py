import base64
import sys
import types

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import mlflow.assistant.config as config_module
from mlflow.assistant.config import AssistantConfig, ProjectConfig, set_config_user
from mlflow.assistant.providers.base import clear_config_cache
from mlflow.server.assistant.api import assistant_router

CONFIG_URL = "/ajax-api/3.0/mlflow/assistant/config"


def _auth(username: str) -> dict[str, str]:
    return {"Authorization": f"Basic {base64.b64encode(f'{username}:pw'.encode()).decode()}"}


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    home = tmp_path / ".mlflow" / "assistant"
    monkeypatch.setattr(config_module, "MLFLOW_ASSISTANT_HOME", home)
    monkeypatch.setattr(config_module, "CONFIG_PATH", home / "config.json")
    monkeypatch.setenv("MLFLOW_ENABLE_REMOTE_ASSISTANT", "true")
    set_config_user(None)
    clear_config_cache()
    yield
    set_config_user(None)
    clear_config_cache()


@pytest.fixture
def auth_enabled(monkeypatch):
    def _authenticate(request):
        header = request.headers.get("Authorization")
        if not header:
            return None
        username = base64.b64decode(header.split(" ", 1)[1]).decode().partition(":")[0]
        return types.SimpleNamespace(username=username, id=username)

    module = types.ModuleType("mlflow.server.auth")
    module.is_auth_enabled = lambda: True
    module.authenticate_fastapi_request_user = _authenticate
    monkeypatch.setitem(sys.modules, "mlflow.server.auth", module)


def _client(monkeypatch, localhost: bool) -> TestClient:
    monkeypatch.setattr("mlflow.server.assistant.api._is_localhost", lambda request: localhost)
    app = FastAPI()
    app.include_router(assistant_router)
    return TestClient(app)


def test_remote_authenticated_user_writes_own_provider_config(auth_enabled, monkeypatch):
    client = _client(monkeypatch, localhost=False)
    response = client.put(
        CONFIG_URL,
        json={"providers": {"mlflow_gateway": {"model": "gpt-x", "selected": True}}},
        headers=_auth("alice"),
    )
    assert response.status_code == 200

    set_config_user("alice")
    assert AssistantConfig.load().providers["mlflow_gateway"].model == "gpt-x"
    # A different user does not inherit alice's selection.
    set_config_user("bob")
    assert "mlflow_gateway" not in AssistantConfig.load().providers


def test_remote_config_write_denied_on_no_auth_server(monkeypatch):
    # No auth plugin -> AUTHENTICATED policy refuses the remote write (no identity to attribute).
    monkeypatch.delitem(sys.modules, "mlflow.server.auth", raising=False)
    client = _client(monkeypatch, localhost=False)
    response = client.put(CONFIG_URL, json={"providers": {"mlflow_gateway": {"model": "gpt-x"}}})
    assert response.status_code == 403


def test_remote_user_cannot_configure_projects(auth_enabled, monkeypatch):
    client = _client(monkeypatch, localhost=False)
    response = client.put(
        CONFIG_URL,
        json={"projects": {"exp1": {"location": "/srv/proj"}}},
        headers=_auth("alice"),
    )
    assert response.status_code == 403


def test_remote_user_cannot_set_api_key(auth_enabled, monkeypatch):
    client = _client(monkeypatch, localhost=False)
    response = client.put(
        CONFIG_URL,
        json={
            "providers": {"mlflow_gateway": {"api_key": "sk-secret", "gateway_vendor": "openai"}}
        },
        headers=_auth("alice"),
    )
    assert response.status_code == 403


def test_remote_user_cannot_grant_full_access(auth_enabled, monkeypatch):
    # Full access bypasses all permission checks, so it is host-only like projects and API keys.
    client = _client(monkeypatch, localhost=False)
    response = client.put(
        CONFIG_URL,
        json={"providers": {"claude_code": {"permissions": {"full_access": True}}}},
        headers=_auth("alice"),
    )
    assert response.status_code == 403

    # The legitimate frontend payload always sends full_access=false, which is accepted.
    response = client.put(
        CONFIG_URL,
        json={"providers": {"claude_code": {"permissions": {"full_access": False}}}},
        headers=_auth("alice"),
    )
    assert response.status_code == 200


def test_remote_write_requires_credentials_when_auth_enabled(auth_enabled, monkeypatch):
    # Auth plugin active + remote caller + no credentials -> 401 with a basic-auth challenge, so
    # the Assistant cannot be driven anonymously on an authenticated deployment.
    client = _client(monkeypatch, localhost=False)
    response = client.put(CONFIG_URL, json={"providers": {"mlflow_gateway": {"model": "gpt-x"}}})
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"].startswith("Basic")


def test_localhost_can_configure_projects(tmp_path, monkeypatch):
    # Server-level project registration stays available from the server host (no-auth here).
    proj = tmp_path / "proj"
    proj.mkdir()
    client = _client(monkeypatch, localhost=True)
    response = client.put(CONFIG_URL, json={"projects": {"exp1": {"location": str(proj)}}})
    assert response.status_code == 200
    assert AssistantConfig.load().projects["exp1"] == ProjectConfig(location=str(proj))
