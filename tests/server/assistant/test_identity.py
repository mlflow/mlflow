import base64
import sys
import types
from unittest import mock

import pytest
from fastapi import APIRouter, FastAPI, Request
from fastapi.testclient import TestClient
from starlette.responses import PlainTextResponse

from mlflow.server.assistant.api import (
    _AssistantAPIRoute,
    _remote_access_policy,
    _RemoteAccessPolicy,
    assistant_router,
)
from mlflow.server.assistant.identity import (
    AssistantAuthError,
    auth_plugin_active,
    resolve_authenticated_username,
)


def _basic_header(username: str, password: str) -> str:
    encoded = base64.b64encode(f"{username}:{password}".encode()).decode()
    return f"Basic {encoded}"


def _fake_auth_module(*, initialized: bool, auth_result=None):
    """A stand-in ``mlflow.server.auth``.

    ``initialized`` drives ``is_auth_enabled()``; ``auth_result`` is what the FastAPI auth entry
    point returns (a user object, None, or a Response).
    """
    module = types.ModuleType("mlflow.server.auth")
    module.is_auth_enabled = lambda: initialized
    module.authenticate_fastapi_request_user = mock.MagicMock(return_value=auth_result)
    return module


@pytest.fixture
def auth_disabled(monkeypatch):
    monkeypatch.delitem(sys.modules, "mlflow.server.auth", raising=False)


@pytest.fixture
def auth_enabled(monkeypatch):
    """Simulate an active, initialized auth plugin that authenticates as user 'alice'."""
    module = _fake_auth_module(
        initialized=True, auth_result=types.SimpleNamespace(username="alice")
    )
    monkeypatch.setitem(sys.modules, "mlflow.server.auth", module)
    return module


def _request_with_auth(header: str | None):
    headers = {"Authorization": header} if header is not None else {}
    return types.SimpleNamespace(headers=headers)


def test_auth_plugin_active_requires_initialized_plugin(monkeypatch):
    monkeypatch.delitem(sys.modules, "mlflow.server.auth", raising=False)
    assert auth_plugin_active() is False

    # Module imported (e.g. by the GraphQL middleware) but the auth app factory never ran: the
    # plugin is not active, so the Assistant must NOT start demanding credentials.
    monkeypatch.setitem(sys.modules, "mlflow.server.auth", _fake_auth_module(initialized=False))
    assert auth_plugin_active() is False

    monkeypatch.setitem(sys.modules, "mlflow.server.auth", _fake_auth_module(initialized=True))
    assert auth_plugin_active() is True


def test_resolve_username_none_when_auth_disabled(auth_disabled):
    assert resolve_authenticated_username(_request_with_auth(None)) is None
    assert resolve_authenticated_username(_request_with_auth(_basic_header("a", "b"))) is None


def test_resolve_username_none_when_plugin_imported_but_not_initialized(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow.server.auth", _fake_auth_module(initialized=False))
    assert resolve_authenticated_username(_request_with_auth(_basic_header("a", "b"))) is None


def test_resolve_username_returns_authenticated_user(auth_enabled):
    request = _request_with_auth(_basic_header("alice", "pw"))
    assert resolve_authenticated_username(request) == "alice"
    auth_enabled.authenticate_fastapi_request_user.assert_called_once_with(request)


def test_resolve_username_raises_when_unauthenticated(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "mlflow.server.auth", _fake_auth_module(initialized=True, auth_result=None)
    )
    with pytest.raises(AssistantAuthError, match="Valid MLflow credentials"):
        resolve_authenticated_username(_request_with_auth(None))


def test_resolve_username_raises_on_custom_auth_response(monkeypatch):
    # A custom authorization_function can return an error Response; the Assistant treats any
    # non-user result as unauthenticated.
    error_response = PlainTextResponse("nope", status_code=403)
    monkeypatch.setitem(
        sys.modules,
        "mlflow.server.auth",
        _fake_auth_module(initialized=True, auth_result=error_response),
    )
    with pytest.raises(AssistantAuthError, match="Valid MLflow credentials"):
        resolve_authenticated_username(_request_with_auth(_basic_header("alice", "pw")))


@pytest.fixture
def config_route_client():
    """A client hitting GET /config (remote policy NONE), so only the identity layer gates it."""
    app = FastAPI()
    app.include_router(assistant_router)
    with mock.patch("mlflow.server.assistant.api._is_localhost", return_value=True):
        yield TestClient(app)


def test_route_allows_anonymous_when_auth_disabled(config_route_client, auth_disabled):
    assert config_route_client.get("/ajax-api/3.0/mlflow/assistant/config").status_code == 200


def test_route_rejects_missing_credentials_when_auth_enabled(config_route_client, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "mlflow.server.auth", _fake_auth_module(initialized=True, auth_result=None)
    )
    response = config_route_client.get("/ajax-api/3.0/mlflow/assistant/config")
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == 'Basic realm="mlflow"'


def test_route_allows_valid_credentials_when_auth_enabled(config_route_client, auth_enabled):
    response = config_route_client.get(
        "/ajax-api/3.0/mlflow/assistant/config",
        headers={"Authorization": _basic_header("alice", "pw")},
    )
    assert response.status_code == 200


@pytest.fixture
def probe_client():
    """A client with a route that echoes the identity the route handler threaded onto state."""
    router = APIRouter(route_class=_AssistantAPIRoute)

    @router.get("/_probe")
    @_remote_access_policy(_RemoteAccessPolicy.NONE)
    async def _probe(request: Request):
        return {"username": request.state.assistant_username}

    app = FastAPI()
    app.include_router(router)
    with mock.patch("mlflow.server.assistant.api._is_localhost", return_value=True):
        yield TestClient(app)


def test_route_threads_none_when_auth_disabled(probe_client, auth_disabled):
    assert probe_client.get("/_probe").json() == {"username": None}


def test_route_threads_username_when_auth_enabled(probe_client, auth_enabled):
    response = probe_client.get("/_probe", headers={"Authorization": _basic_header("alice", "pw")})
    assert response.json() == {"username": "alice"}
