from __future__ import annotations

import json
from unittest import mock

import pytest
from flask import Flask

from mlflow.entities.workspace import Workspace
from mlflow.server.handlers import get_endpoints


@pytest.fixture(autouse=True)
def enable_workspaces(monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_WORKSPACES", "true")


@pytest.fixture(autouse=True)
def stub_workspace_dependencies(monkeypatch):
    """
    Prevent workspace endpoint tests from hitting real tracking stores.
    """
    monkeypatch.setattr(
        "mlflow.server.handlers._ensure_default_workspace_experiment",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "mlflow.server.handlers._workspace_contains_resources",
        lambda *_args, **_kwargs: False,
    )


@pytest.fixture
def app(monkeypatch):
    flask_app = Flask(__name__)
    for rule, view_func, methods in get_endpoints():
        flask_app.add_url_rule(rule, view_func=view_func, methods=methods)
    return flask_app


@pytest.fixture
def mock_workspace_store(monkeypatch):
    store = mock.Mock()
    monkeypatch.setattr(
        "mlflow.server.handlers._get_workspace_store",
        lambda *_, **__: store,
    )
    return store


def _workspace_to_json(payload):
    return json.loads(payload)


def test_list_workspaces_endpoint(app, mock_workspace_store):
    mock_workspace_store.list_workspaces.return_value = [
        Workspace(name="default", description="Default"),
        Workspace(name="team-a", description=None),
    ]
    with app.test_client() as client:
        response = client.get("/api/2.0/mlflow/workspaces")

    assert response.status_code == 200
    payload = _workspace_to_json(response.get_data(True))
    assert payload == {
        "workspaces": [
            {"name": "default", "description": "Default"},
            {"name": "team-a", "description": None},
        ]
    }
    mock_workspace_store.list_workspaces.assert_called_once_with()


def test_create_workspace_endpoint(app, mock_workspace_store):
    created = Workspace(name="team-b", description="Team B")
    mock_workspace_store.create_workspace.return_value = created
    with app.test_client() as client:
        response = client.post(
            "/api/2.0/mlflow/workspaces",
            json={"name": "team-b", "description": "Team B"},
        )

    assert response.status_code == 201
    payload = _workspace_to_json(response.get_data(True))
    assert payload == created.to_dict()
    mock_workspace_store.create_workspace.assert_called_once()


def test_get_workspace_endpoint(app, mock_workspace_store):
    mock_workspace_store.get_workspace.return_value = Workspace(name="team-c", description="Team C")
    with app.test_client() as client:
        response = client.get("/api/2.0/mlflow/workspaces/team-c")

    assert response.status_code == 200
    payload = _workspace_to_json(response.get_data(True))
    assert payload == {"name": "team-c", "description": "Team C"}
    mock_workspace_store.get_workspace.assert_called_once_with("team-c")


def test_update_workspace_endpoint(app, mock_workspace_store):
    updated = Workspace(name="team-d", description="Updated")
    mock_workspace_store.update_workspace.return_value = updated
    with app.test_client() as client:
        response = client.patch(
            "/api/2.0/mlflow/workspaces/team-d",
            json={"description": "Updated"},
        )

    assert response.status_code == 200
    payload = _workspace_to_json(response.get_data(True))
    assert payload == updated.to_dict()
    mock_workspace_store.update_workspace.assert_called_once()


def test_delete_workspace_endpoint(app, mock_workspace_store):
    with app.test_client() as client:
        response = client.delete("/api/2.0/mlflow/workspaces/team-e")

    assert response.status_code == 204
    mock_workspace_store.delete_workspace.assert_called_once_with("team-e")
