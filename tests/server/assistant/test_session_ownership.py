import base64
import shutil
import sys
import types
from unittest import mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mlflow.server.assistant.api import assistant_router
from mlflow.server.assistant.session import SESSION_DIR, Session, SessionManager


def _auth(username: str) -> dict[str, str]:
    encoded = base64.b64encode(f"{username}:pw".encode()).decode()
    return {"Authorization": f"Basic {encoded}"}


@pytest.fixture(autouse=True)
def clear_sessions():
    if SESSION_DIR.exists():
        shutil.rmtree(SESSION_DIR)
    yield
    if SESSION_DIR.exists():
        shutil.rmtree(SESSION_DIR)


@pytest.fixture
def auth_enabled(monkeypatch):
    """An active auth plugin that authenticates each request as the Basic-auth username it sends."""

    def _authenticate(request):
        header = request.headers.get("Authorization")
        if not header or not header.lower().startswith("basic "):
            return None
        username = base64.b64decode(header.split(" ", 1)[1]).decode().partition(":")[0]
        return types.SimpleNamespace(username=username, id=username)

    module = types.ModuleType("mlflow.server.auth")
    module.is_auth_enabled = lambda: True
    module.authenticate_fastapi_request_user = _authenticate
    monkeypatch.setitem(sys.modules, "mlflow.server.auth", module)
    return module


@pytest.fixture
def client(auth_enabled):
    app = FastAPI()
    app.include_router(assistant_router)
    with mock.patch("mlflow.server.assistant.api._is_localhost", return_value=True):
        yield TestClient(app)


@pytest.fixture
def no_auth_client(monkeypatch):
    """A client on a server without the auth plugin (no identity, single-user)."""
    monkeypatch.delitem(sys.modules, "mlflow.server.auth", raising=False)
    app = FastAPI()
    app.include_router(assistant_router)
    with mock.patch("mlflow.server.assistant.api._is_localhost", return_value=True):
        yield TestClient(app)


def _create_session_as(client, username: str) -> str:
    response = client.post(
        "/ajax-api/3.0/mlflow/assistant/message",
        json={"message": "hello"},
        headers=_auth(username),
    )
    assert response.status_code == 200
    return response.json()["session_id"]


def test_session_created_with_owner(client):
    session_id = _create_session_as(client, "alice")
    assert SessionManager.load(session_id).owner == "alice"


def test_owner_can_send_follow_up_to_own_session(client):
    session_id = _create_session_as(client, "alice")
    response = client.post(
        "/ajax-api/3.0/mlflow/assistant/message",
        json={"message": "again", "session_id": session_id},
        headers=_auth("alice"),
    )
    assert response.status_code == 200


def test_other_user_cannot_send_to_someone_elses_session(client):
    session_id = _create_session_as(client, "alice")
    response = client.post(
        "/ajax-api/3.0/mlflow/assistant/message",
        json={"message": "intrude", "session_id": session_id},
        headers=_auth("bob"),
    )
    assert response.status_code == 404


def test_other_user_cannot_stream_someone_elses_session(client):
    session_id = _create_session_as(client, "alice")
    response = client.get(
        f"/ajax-api/3.0/mlflow/assistant/sessions/{session_id}/stream", headers=_auth("bob")
    )
    assert response.status_code == 404


def test_other_user_cannot_cancel_someone_elses_session(client):
    session_id = _create_session_as(client, "alice")
    response = client.patch(
        f"/ajax-api/3.0/mlflow/assistant/sessions/{session_id}",
        json={"status": "cancelled"},
        headers=_auth("bob"),
    )
    assert response.status_code == 404


def test_other_user_cannot_resolve_permission_on_someone_elses_session(client):
    session_id = _create_session_as(client, "alice")
    response = client.post(
        f"/ajax-api/3.0/mlflow/assistant/sessions/{session_id}/permission",
        json={"request_id": "t1", "decision": "allow"},
        headers=_auth("bob"),
    )
    assert response.status_code == 404


def test_other_user_cannot_deliver_tool_result_to_someone_elses_session(client):
    session_id = _create_session_as(client, "alice")
    response = client.post(
        f"/ajax-api/3.0/mlflow/assistant/sessions/{session_id}/tool-result",
        json={"request_id": "t1", "content": "x", "is_error": False},
        headers=_auth("bob"),
    )
    assert response.status_code == 404


def test_owner_can_cancel_own_session(client):
    session_id = _create_session_as(client, "alice")
    response = client.patch(
        f"/ajax-api/3.0/mlflow/assistant/sessions/{session_id}",
        json={"status": "cancelled"},
        headers=_auth("alice"),
    )
    assert response.status_code == 200


def test_owner_round_trips_through_serialization():
    session = SessionManager.create(owner="alice")
    assert Session.from_dict(session.to_dict()).owner == "alice"
    # A legacy session persisted before the owner field defaults to unowned.
    assert Session.from_dict({"messages": []}).owner is None


def test_no_auth_server_preserves_single_user_access(no_auth_client):
    # With no auth plugin, create / follow-up / cancel all work without credentials (owner None).
    create = no_auth_client.post("/ajax-api/3.0/mlflow/assistant/message", json={"message": "hi"})
    assert create.status_code == 200
    session_id = create.json()["session_id"]
    assert SessionManager.load(session_id).owner is None

    follow_up = no_auth_client.post(
        "/ajax-api/3.0/mlflow/assistant/message",
        json={"message": "again", "session_id": session_id},
    )
    assert follow_up.status_code == 200

    cancel = no_auth_client.patch(
        f"/ajax-api/3.0/mlflow/assistant/sessions/{session_id}", json={"status": "cancelled"}
    )
    assert cancel.status_code == 200


def test_legacy_unowned_session_unreachable_under_auth(client):
    # A session persisted on a no-auth server (owner None) is not loadable once auth is enabled.
    session_id = "f5f28c66-5ec6-46a1-9a2e-ca55fb64bf47"
    SessionManager.save(session_id, SessionManager.create(owner=None))

    response = client.get(
        f"/ajax-api/3.0/mlflow/assistant/sessions/{session_id}/stream", headers=_auth("alice")
    )
    assert response.status_code == 404
