from __future__ import annotations

import json
from types import SimpleNamespace
from unittest import mock

import pytest

from mlflow.entities.workspace import Workspace
from mlflow.exceptions import MlflowException, RestException
from mlflow.protos.service_pb2 import (
    CreateWorkspace,
    DeleteWorkspace,
    GetWorkspace,
    ListWorkspaces,
    UpdateWorkspace,
)
from mlflow.store.workspace.rest_store import WORKSPACES_ENDPOINT, RestWorkspaceStore


@pytest.fixture
def host_creds():
    return SimpleNamespace()


@pytest.fixture
def store(host_creds):
    return RestWorkspaceStore(lambda: host_creds)


def test_list_workspaces_parses_response(store, host_creds):
    response = ListWorkspaces.Response()
    response.workspaces.add(name="default", description="Default workspace")
    response.workspaces.add(name="team-a", description="Team A")
    with mock.patch(
        "mlflow.store.workspace.rest_store.call_endpoint", return_value=response
    ) as call_endpoint:
        workspaces = store.list_workspaces()

    assert [ws.name for ws in workspaces] == ["default", "team-a"]
    call_endpoint.assert_called_once()
    kwargs = call_endpoint.call_args.kwargs
    assert kwargs["host_creds"] is host_creds
    assert kwargs["endpoint"] == WORKSPACES_ENDPOINT
    assert kwargs["method"] == "GET"
    assert kwargs["json_body"] is None
    assert kwargs.get("expected_status", 200) == 200


def test_get_workspace_returns_entity(store, host_creds):
    response = GetWorkspace.Response()
    response.workspace.name = "team-b"
    response.workspace.description = "Team B"
    with mock.patch(
        "mlflow.store.workspace.rest_store.call_endpoint", return_value=response
    ) as call_endpoint:
        workspace = store.get_workspace("team-b")

    assert workspace.name == "team-b"
    assert workspace.description == "Team B"
    call_endpoint.assert_called_once()
    kwargs = call_endpoint.call_args.kwargs
    assert kwargs["endpoint"] == f"{WORKSPACES_ENDPOINT}/team-b"
    assert kwargs["method"] == "GET"


def test_create_workspace_sends_payload(store, host_creds):
    response = CreateWorkspace.Response()
    response.workspace.name = "team-c"
    response.workspace.description = "Team C"
    with mock.patch(
        "mlflow.store.workspace.rest_store.call_endpoint", return_value=response
    ) as call_endpoint:
        workspace = store.create_workspace(Workspace(name="team-c", description="Team C"))

    assert workspace.name == "team-c"
    assert workspace.description == "Team C"
    call_endpoint.assert_called_once()
    kwargs = call_endpoint.call_args.kwargs
    assert kwargs["endpoint"] == WORKSPACES_ENDPOINT
    assert kwargs["method"] == "POST"
    assert kwargs["expected_status"] == 201
    assert json.loads(kwargs["json_body"]) == {"name": "team-c", "description": "Team C"}


def test_create_workspace_conflict_raises_resource_exists(store, monkeypatch):
    exc = RestException({"error_code": "RESOURCE_ALREADY_EXISTS", "message": "already exists"})
    monkeypatch.setattr(
        "mlflow.store.workspace.rest_store.call_endpoint",
        mock.Mock(side_effect=exc),
    )

    with pytest.raises(
        MlflowException,
        match="already exists",
    ) as exc_info:
        store.create_workspace(Workspace(name="team-a"))

    assert exc_info.value.error_code == "RESOURCE_ALREADY_EXISTS"
    assert "already exists" in exc_info.value.message


def test_create_workspace_handles_400_resource_exists(store, monkeypatch):
    exc = RestException({"error_code": "RESOURCE_ALREADY_EXISTS", "message": "already exists"})
    monkeypatch.setattr(
        "mlflow.store.workspace.rest_store.call_endpoint",
        mock.Mock(side_effect=exc),
    )

    with pytest.raises(
        MlflowException,
        match="already exists",
    ) as exc_info:
        store.create_workspace(Workspace(name="team-a"))

    assert exc_info.value.error_code == "RESOURCE_ALREADY_EXISTS"


def test_update_workspace_returns_new_description(store, host_creds):
    response = UpdateWorkspace.Response()
    response.workspace.name = "team-e"
    response.workspace.description = "updated"
    with mock.patch(
        "mlflow.store.workspace.rest_store.call_endpoint", return_value=response
    ) as call_endpoint:
        workspace = store.update_workspace(Workspace(name="team-e", description="updated"))

    assert workspace.description == "updated"
    call_endpoint.assert_called_once()
    kwargs = call_endpoint.call_args.kwargs
    assert kwargs["endpoint"] == f"{WORKSPACES_ENDPOINT}/team-e"
    assert kwargs["method"] == "PATCH"
    assert json.loads(kwargs["json_body"]) == {"description": "updated"}


def test_delete_workspace_returns_on_success(store, host_creds):
    response = DeleteWorkspace.Response()
    with mock.patch(
        "mlflow.store.workspace.rest_store.call_endpoint", return_value=response
    ) as call_endpoint:
        store.delete_workspace("team-f")

    call_endpoint.assert_called_once()
    kwargs = call_endpoint.call_args.kwargs
    assert kwargs["endpoint"] == f"{WORKSPACES_ENDPOINT}/team-f"
    assert kwargs["method"] == "DELETE"
    assert kwargs["expected_status"] == 204
    assert kwargs["json_body"] is None


def test_get_default_workspace_not_supported(store):
    with pytest.raises(
        MlflowException,
        match="REST workspace provider does not expose a default workspace",
    ) as exc:
        store.get_default_workspace()
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


def test_rest_store_validates_workspace_names_before_http(monkeypatch, store):
    mock_call = mock.Mock()
    monkeypatch.setattr("mlflow.store.workspace.rest_store.call_endpoint", mock_call)

    with pytest.raises(MlflowException, match="must match the pattern"):
        store.get_workspace("Invalid")

    mock_call.assert_not_called()
