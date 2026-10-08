import base64
import sys
import types
from unittest import mock

import pytest
from fastapi import APIRouter, FastAPI, Request
from fastapi.testclient import TestClient
from starlette.responses import PlainTextResponse

from mlflow.assistant.providers.tool_executor import is_remote_caller
from mlflow.exceptions import MlflowException
from mlflow.server.assistant.api import (
    _AssistantAPIRoute,
    _is_restricted_caller,
    _remote_access_policy,
    _RemoteAccessPolicy,
    assistant_router,
)
from mlflow.server.assistant.identity import (
    AssistantAuthError,
    auth_plugin_active,
    resolve_authenticated_username,
    user_is_admin,
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
    module.store = types.SimpleNamespace(
        get_user=lambda username: types.SimpleNamespace(is_admin=False)
    )
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
    # Without the permission middleware to populate request.state.username, the route authenticates
    # the request itself -- exactly once.
    auth_enabled.authenticate_fastapi_request_user.assert_called_once()


def test_route_reuses_middleware_username_without_reauthenticating(auth_enabled):
    # On an authenticated server the FastAPI permission middleware authenticates the request and
    # stores the user on request.state.username before the route handler runs. The Assistant route
    # must reuse that identity, not authenticate a second time (a custom authorization_function
    # would otherwise run twice).
    router = APIRouter(route_class=_AssistantAPIRoute)

    @router.get("/_probe")
    @_remote_access_policy(_RemoteAccessPolicy.NONE)
    async def _probe(request: Request):
        return {"username": request.state.assistant_username}

    app = FastAPI()
    app.include_router(router)

    @app.middleware("http")
    async def _populate_username(request: Request, call_next):
        request.state.username = "alice"
        return await call_next(request)

    with mock.patch("mlflow.server.assistant.api._is_localhost", return_value=True):
        response = TestClient(app).get("/_probe")

    assert response.json() == {"username": "alice"}
    auth_enabled.authenticate_fastapi_request_user.assert_not_called()


def _auth_module_with_admins(admins: set[str]):
    module = _fake_auth_module(initialized=True)

    def _get_user(username):
        if username == "ghost":
            raise MlflowException("User with username=ghost not found")
        return types.SimpleNamespace(is_admin=username in admins)

    module.store = types.SimpleNamespace(get_user=_get_user)
    return module


def test_user_is_admin(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow.server.auth", _auth_module_with_admins({"admin"}))

    assert user_is_admin("admin") is True
    assert user_is_admin("alice") is False
    # A user missing from the auth store (e.g. from a custom authorization_function) is not an
    # admin, rather than an error.
    assert user_is_admin("ghost") is False
    assert user_is_admin(None) is False


def _caller(host: str, username: str | None):
    return types.SimpleNamespace(
        client=types.SimpleNamespace(host=host),
        state=types.SimpleNamespace(assistant_username=username),
    )


@pytest.mark.parametrize(
    ("host", "username", "auth_on", "sandbox_on", "expected"),
    [
        # A remote caller is always restricted.
        ("10.0.0.5", "admin", True, True, True),
        # On an auth server with the sandbox on, a local non-admin is restricted too...
        ("127.0.0.1", "alice", True, True, True),
        # ...but a local admin is not.
        ("127.0.0.1", "admin", True, True, False),
        # Without the sandbox, local callers keep the host behavior of earlier releases.
        ("127.0.0.1", "alice", True, False, False),
        # Without auth there is no admin distinction.
        ("127.0.0.1", None, False, True, False),
    ],
)
def test_is_restricted_caller(monkeypatch, host, username, auth_on, sandbox_on, expected):
    if auth_on:
        monkeypatch.setitem(sys.modules, "mlflow.server.auth", _auth_module_with_admins({"admin"}))
    else:
        monkeypatch.delitem(sys.modules, "mlflow.server.auth", raising=False)
    monkeypatch.setattr("mlflow.server.assistant.api.assistant_sandbox_enabled", lambda: sandbox_on)

    assert _is_restricted_caller(_caller(host, username)) is expected


@pytest.mark.parametrize(("username", "restricted"), [("alice", True), ("admin", False)])
def test_route_marks_local_non_admin_as_restricted_with_sandbox_on(
    monkeypatch, username, restricted
):
    auth_module = _auth_module_with_admins({"admin"})
    auth_module.authenticate_fastapi_request_user = mock.MagicMock(
        return_value=types.SimpleNamespace(username=username)
    )
    monkeypatch.setitem(sys.modules, "mlflow.server.auth", auth_module)
    monkeypatch.setattr("mlflow.server.assistant.api.assistant_sandbox_enabled", lambda: True)

    router = APIRouter(route_class=_AssistantAPIRoute)

    @router.get("/_probe")
    @_remote_access_policy(_RemoteAccessPolicy.NONE)
    async def _probe():
        return {"restricted": is_remote_caller()}

    app = FastAPI()
    app.include_router(router)
    with mock.patch("mlflow.server.assistant.api._is_localhost", return_value=True):
        response = TestClient(app).get("/_probe", headers={"Authorization": "Basic x"})

    assert response.json() == {"restricted": restricted}
