from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from mlflow.entities.workspace import Workspace
from mlflow.exceptions import MlflowException
from mlflow.store.workspace.rest_store import RestWorkspaceStore


def _response(status_code=200, payload=None):
    text = json.dumps(payload) if payload is not None else ""
    return SimpleNamespace(status_code=status_code, text=text)


@pytest.fixture
def host_creds():
    return SimpleNamespace()


@pytest.fixture
def store(host_creds):
    return RestWorkspaceStore(lambda: host_creds)


@pytest.fixture(autouse=True)
def _stub_verify(monkeypatch):
    monkeypatch.setattr(
        "mlflow.store.workspace.rest_store.verify_rest_response",
        lambda response, _: response,
    )


def test_list_workspaces_parses_response(store, host_creds, monkeypatch):
    response = _response(
        payload={
            "workspaces": [
                {"name": "default", "description": "Default workspace"},
                {"name": "team-a", "description": "Team A"},
            ]
        }
    )
    captured = {}

    def fake_http(**kwargs):
        captured.update(kwargs)
        return response

    monkeypatch.setattr("mlflow.store.workspace.rest_store.http_request", fake_http)

    workspaces = store.list_workspaces()
    assert [ws.name for ws in workspaces] == ["default", "team-a"]
    assert captured["host_creds"] is host_creds
    assert captured["endpoint"] == "/api/2.0/mlflow/workspaces"
    assert captured["method"] == "GET"


def test_get_workspace_returns_entity(store, host_creds, monkeypatch):
    response = _response(payload={"name": "team-b", "description": "Team B"})
    captured = {}

    def fake_http(**kwargs):
        captured.update(kwargs)
        return response

    monkeypatch.setattr("mlflow.store.workspace.rest_store.http_request", fake_http)

    workspace = store.get_workspace("team-b")
    assert workspace.name == "team-b"
    assert workspace.description == "Team B"
    assert captured["endpoint"] == "/api/2.0/mlflow/workspaces/team-b"
    assert captured["method"] == "GET"


def test_create_workspace_sends_payload(store, host_creds, monkeypatch):
    response = _response(status_code=201, payload={"name": "team-c", "description": "Team C"})
    captured = {}

    def fake_http(**kwargs):
        captured.update(kwargs)
        return response

    monkeypatch.setattr("mlflow.store.workspace.rest_store.http_request", fake_http)

    workspace = store.create_workspace(Workspace(name="team-c", description="Team C"))
    assert workspace.name == "team-c"
    assert workspace.description == "Team C"
    assert captured["json"] == {"name": "team-c", "description": "Team C"}
    assert captured["method"] == "POST"


def test_create_workspace_uses_request_payload_when_response_empty(store, monkeypatch):
    response = _response(status_code=201, payload=None)
    monkeypatch.setattr("mlflow.store.workspace.rest_store.http_request", lambda **_: response)

    workspace = store.create_workspace(Workspace(name="team-d", description="desc"))
    assert workspace.name == "team-d"
    assert workspace.description == "desc"


def test_create_workspace_conflict_raises_resource_exists(store, monkeypatch):
    response = _response(
        status_code=409,
        payload={
            "error_code": "RESOURCE_ALREADY_EXISTS",
            "message": "Workspace 'team-a' already exists.",
        },
    )
    monkeypatch.setattr("mlflow.store.workspace.rest_store.http_request", lambda **_: response)

    with pytest.raises(
        MlflowException,
        match="Workspace 'team-a' already exists.",
    ) as exc:
        store.create_workspace(Workspace(name="team-a"))

    assert exc.value.error_code == "RESOURCE_ALREADY_EXISTS"
    assert "already exists" in exc.value.message


def test_create_workspace_handles_400_resource_exists(store, monkeypatch):
    response = _response(
        status_code=400,
        payload={
            "error_code": "RESOURCE_ALREADY_EXISTS",
            "message": "Workspace 'team-a' already exists.",
        },
    )
    monkeypatch.setattr("mlflow.store.workspace.rest_store.http_request", lambda **_: response)

    with pytest.raises(
        MlflowException,
        match="Workspace 'team-a' already exists.",
    ) as exc:
        store.create_workspace(Workspace(name="team-a"))

    assert exc.value.error_code == "RESOURCE_ALREADY_EXISTS"


def test_update_workspace_returns_new_description(store, monkeypatch):
    response = _response(payload={"name": "team-e", "description": "updated"})
    captured = {}

    def fake_http(**kwargs):
        captured.update(kwargs)
        return response

    monkeypatch.setattr("mlflow.store.workspace.rest_store.http_request", fake_http)

    workspace = store.update_workspace(Workspace(name="team-e", description="updated"))
    assert workspace.description == "updated"
    assert captured["method"] == "PATCH"
    assert captured["json"] == {"description": "updated"}
    assert captured["endpoint"] == "/api/2.0/mlflow/workspaces/team-e"


def test_delete_workspace_returns_on_success(store, monkeypatch):
    response = _response(status_code=204)
    captured = {}

    def fake_http(**kwargs):
        captured.update(kwargs)
        return response

    monkeypatch.setattr("mlflow.store.workspace.rest_store.http_request", fake_http)

    store.delete_workspace("team-f")
    assert captured["method"] == "DELETE"
    assert captured["endpoint"] == "/api/2.0/mlflow/workspaces/team-f"


def test_get_default_workspace_not_supported(store):
    with pytest.raises(
        MlflowException,
        match="REST workspace provider does not expose a default workspace",
    ) as exc:
        store.get_default_workspace()
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
