import base64
import sys
import types

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import mlflow.assistant.config as config_module
from mlflow.assistant.config import AssistantConfig, ProjectConfig, set_config_user
from mlflow.assistant.providers.base import clear_config_cache
from mlflow.exceptions import MlflowException
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
    # Pinned off so results do not depend on whether Docker is installed; see ``sandbox_on``.
    monkeypatch.setenv("MLFLOW_ENABLE_ASSISTANT_SANDBOX", "false")
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
    module.store = types.SimpleNamespace(
        get_user=lambda username: types.SimpleNamespace(is_admin=username == "admin")
    )
    monkeypatch.setitem(sys.modules, "mlflow.server.auth", module)


@pytest.fixture
def sandbox_on(monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_ASSISTANT_SANDBOX", "true")


def _client(monkeypatch, localhost: bool) -> TestClient:
    monkeypatch.setattr("mlflow.server.assistant.api._is_localhost", lambda request: localhost)
    app = FastAPI()
    app.include_router(assistant_router)
    return TestClient(app)


def test_remote_authenticated_user_writes_own_provider_config(
    auth_enabled, sandbox_on, monkeypatch
):
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


_HOST_ONLY_WRITES = [
    pytest.param(
        {"projects": {"exp1": {"location": "/srv/proj"}}}, "Project directories", id="projects"
    ),
    pytest.param({"projects": {"exp1": None}}, "Project directories", id="remove-project"),
    pytest.param(
        {"providers": {"mlflow_gateway": {"api_key": "sk-secret", "gateway_vendor": "openai"}}},
        "Gateway connections",
        id="api-key",
    ),
    pytest.param(
        {"providers": {"claude_code": {"permissions": {"full_access": True}}}},
        "Full access",
        id="full-access",
    ),
]


@pytest.mark.parametrize("username", ["alice", "admin"])
@pytest.mark.parametrize(("payload", "setting"), _HOST_ONLY_WRITES)
def test_remote_caller_cannot_change_server_wide_settings(
    auth_enabled, monkeypatch, username, payload, setting
):
    # Even an admin: these settings can only be changed from the MLflow server host.
    client = _client(monkeypatch, localhost=False)
    response = client.put(CONFIG_URL, json=payload, headers=_auth(username))
    assert response.status_code == 403
    assert response.json()["detail"].startswith(setting)
    assert "from the MLflow server host" in response.json()["detail"]


def test_remote_caller_can_save_full_access_off(auth_enabled, monkeypatch):
    # The frontend always sends full_access=false for a caller who cannot enable it.
    client = _client(monkeypatch, localhost=False)
    response = client.put(
        CONFIG_URL,
        json={"providers": {"claude_code": {"permissions": {"full_access": False}}}},
        headers=_auth("alice"),
    )
    assert response.status_code == 200


def test_remote_caller_cannot_select_a_host_only_provider(auth_enabled, sandbox_on, monkeypatch):
    # Once selected, the UI would only show that the Assistant is unavailable, with no way back.
    client = _client(monkeypatch, localhost=False)
    response = client.put(
        CONFIG_URL,
        json={"providers": {"claude_code": {"selected": True}}},
        headers=_auth("alice"),
    )
    assert response.status_code == 403
    assert "cannot be used from a remote client" in response.json()["detail"]


def test_remote_config_write_denied_when_remote_access_is_off(auth_enabled, monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_REMOTE_ASSISTANT", "false")
    client = _client(monkeypatch, localhost=False)
    response = client.put(
        CONFIG_URL,
        json={"providers": {"mlflow_gateway": {"model": "gpt-x"}}},
        headers=_auth("alice"),
    )
    assert response.status_code == 403


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


_SERVER_WIDE_WRITES = [
    {"projects": {"exp1": {"location": "/srv/proj"}}},
    {"projects": {"exp1": None}},
    {"providers": {"mlflow_gateway": {"api_key": "sk-secret", "gateway_vendor": "openai"}}},
    {"providers": {"claude_code": {"permissions": {"full_access": True}}}},
]


@pytest.mark.parametrize("payload", _SERVER_WIDE_WRITES)
def test_localhost_non_admin_cannot_change_server_wide_settings(
    auth_enabled, sandbox_on, monkeypatch, payload
):
    # On a sandboxed server with auth, reaching the host does not make a caller the operator: any
    # authenticated user can, so server-wide settings also require an admin.
    client = _client(monkeypatch, localhost=True)
    response = client.put(CONFIG_URL, json=payload, headers=_auth("alice"))
    assert response.status_code == 403
    assert "by an administrator" in response.json()["detail"]


# Without the sandbox, local users' tools run on the host anyway, so they keep full control of
# server-wide settings, as before the sandbox existed.
def test_localhost_non_admin_sets_a_project_without_the_sandbox(
    auth_enabled, tmp_path, monkeypatch
):
    client = _client(monkeypatch, localhost=True)
    response = client.put(
        CONFIG_URL,
        json={"projects": {"exp1": {"location": str(tmp_path)}}},
        headers=_auth("alice"),
    )
    assert response.status_code == 200
    assert response.json()["can_edit_server_settings"] is True
    assert AssistantConfig.load().projects["exp1"] == ProjectConfig(location=str(tmp_path))


def test_localhost_non_admin_sets_an_api_key_without_the_sandbox(auth_enabled, monkeypatch):
    monkeypatch.setattr(
        "mlflow.server.assistant.api.ensure_gateway_connection",
        lambda vendor, api_key: "mlflow-assistant-openai",
    )
    client = _client(monkeypatch, localhost=True)
    response = client.put(
        CONFIG_URL,
        json={"providers": {"mlflow_gateway": {"api_key": "sk-x", "gateway_vendor": "openai"}}},
        headers=_auth("alice"),
    )
    assert response.status_code == 200
    set_config_user("alice")
    assert AssistantConfig.load().providers["mlflow_gateway"].model == "mlflow-assistant-openai"


def test_localhost_non_admin_enables_full_access_without_the_sandbox(auth_enabled, monkeypatch):
    client = _client(monkeypatch, localhost=True)
    response = client.put(
        CONFIG_URL,
        json={"providers": {"claude_code": {"permissions": {"full_access": True}}}},
        headers=_auth("alice"),
    )
    assert response.status_code == 200
    set_config_user("alice")
    assert AssistantConfig.load().providers["claude_code"].permissions.full_access is True


def test_localhost_non_admin_can_install_skills_without_the_sandbox(auth_enabled, monkeypatch):
    client = _client(monkeypatch, localhost=True)
    response = client.post(
        "/ajax-api/3.0/mlflow/assistant/skills/install",
        json={"type": "global"},
        headers=_auth("alice"),
    )
    # Past the server-settings check; refused only because no provider is selected yet.
    assert response.status_code == 412


def test_localhost_admin_can_configure_projects(auth_enabled, sandbox_on, tmp_path, monkeypatch):
    proj = tmp_path / "proj"
    proj.mkdir()
    client = _client(monkeypatch, localhost=True)
    response = client.put(
        CONFIG_URL,
        json={"projects": {"exp1": {"location": str(proj)}}},
        headers=_auth("admin"),
    )
    assert response.status_code == 200
    assert AssistantConfig.load().projects["exp1"] == ProjectConfig(location=str(proj))


@pytest.mark.parametrize(
    ("username", "sandbox", "can_edit"),
    [("alice", True, False), ("admin", True, True), ("alice", False, True)],
)
def test_localhost_get_config_reports_who_can_edit_server_settings(
    auth_enabled, tmp_path, monkeypatch, username, sandbox, can_edit
):
    monkeypatch.setenv("MLFLOW_ENABLE_ASSISTANT_SANDBOX", str(sandbox).lower())
    AssistantConfig(projects={"exp1": ProjectConfig(location=str(tmp_path))}).save()
    client = _client(monkeypatch, localhost=True)
    response = client.get(CONFIG_URL, headers=_auth(username))
    assert response.status_code == 200
    assert response.json()["can_edit_server_settings"] is can_edit
    # Project locations are shown only to callers who may change them.
    assert ("location" in response.json()["projects"]["exp1"]) is can_edit


def test_localhost_get_config_can_edit_server_settings_without_auth(sandbox_on, monkeypatch):
    client = _client(monkeypatch, localhost=True)
    response = client.get(CONFIG_URL)
    assert response.status_code == 200
    assert response.json()["can_edit_server_settings"] is True


def test_remote_get_config_cannot_edit_server_settings(auth_enabled, monkeypatch):
    client = _client(monkeypatch, localhost=False)
    response = client.get(CONFIG_URL, headers=_auth("admin"))
    assert response.status_code == 200
    assert response.json()["can_edit_server_settings"] is False


def test_localhost_non_admin_writes_own_provider_config(auth_enabled, monkeypatch):
    client = _client(monkeypatch, localhost=True)
    response = client.put(
        CONFIG_URL,
        json={"providers": {"mlflow_gateway": {"model": "gpt-x", "selected": True}}},
        headers=_auth("alice"),
    )
    assert response.status_code == 200

    set_config_user("alice")
    assert AssistantConfig.load().providers["mlflow_gateway"].model == "gpt-x"


def test_localhost_non_admin_cannot_install_skills(auth_enabled, sandbox_on, monkeypatch):
    client = _client(monkeypatch, localhost=True)
    response = client.post(
        "/ajax-api/3.0/mlflow/assistant/skills/install",
        json={"type": "global"},
        headers=_auth("alice"),
    )
    assert response.status_code == 403
    assert "by an administrator" in response.json()["detail"]


def test_user_missing_from_the_auth_store_saves_own_provider_config(auth_enabled, monkeypatch):
    # A user the auth store does not know (e.g. from a custom authorization_function) is treated
    # as a non-admin, not as an error, so they can still save their own provider settings.
    def _missing_user(username):
        raise MlflowException(f"User with username={username} not found")

    monkeypatch.setattr(sys.modules["mlflow.server.auth"].store, "get_user", _missing_user)
    client = _client(monkeypatch, localhost=True)
    response = client.put(
        CONFIG_URL,
        json={"providers": {"mlflow_gateway": {"model": "gpt-x", "selected": True}}},
        headers=_auth("sso-user"),
    )
    assert response.status_code == 200


def test_no_admin_lookup_without_the_sandbox(auth_enabled, monkeypatch):
    def _get_user(username):
        raise AssertionError("the caller must not be looked up when the sandbox is off")

    monkeypatch.setattr(sys.modules["mlflow.server.auth"].store, "get_user", _get_user)
    client = _client(monkeypatch, localhost=True)
    response = client.put(
        CONFIG_URL,
        json={"providers": {"mlflow_gateway": {"model": "gpt-x", "selected": True}}},
        headers=_auth("alice"),
    )
    assert response.status_code == 200
