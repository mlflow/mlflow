from __future__ import annotations

import json
from unittest import mock

import pytest
from flask import Flask

from mlflow.entities.workspace import Workspace, WorkspaceDeletionMode
from mlflow.server.handlers import get_endpoints


@pytest.fixture(autouse=True)
def enable_workspaces(monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_WORKSPACES", "true")


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


@pytest.fixture
def mock_tracking_store(monkeypatch):
    store = mock.Mock()
    store.artifact_root_uri = "/default/artifact/root"
    monkeypatch.setattr(
        "mlflow.server.handlers._get_tracking_store",
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
        response = client.get("/api/3.0/mlflow/workspaces")

    assert response.status_code == 200
    payload = _workspace_to_json(response.get_data(True))
    assert payload["workspaces"][0] == {"name": "default", "description": "Default"}
    assert payload["workspaces"][1] == {"name": "team-a"}
    mock_workspace_store.list_workspaces.assert_called_once_with()


def test_create_workspace_endpoint(app, mock_workspace_store, mock_tracking_store):
    created = Workspace(
        name="team-b",
        description="Team B",
        trace_archival_location="s3://archive/team-b",
        trace_archival_retention="30d",
    )
    mock_workspace_store.create_workspace.return_value = created
    with app.test_client() as client:
        response = client.post(
            "/api/3.0/mlflow/workspaces",
            json={
                "name": "team-b",
                "description": "Team B",
                "trace_archival_config": {
                    "location": "s3://archive/team-b",
                    "retention": "30d",
                },
            },
        )

    assert response.status_code == 201
    payload = _workspace_to_json(response.get_data(True))
    assert payload == {
        "workspace": {
            "name": "team-b",
            "description": "Team B",
            "trace_archival_config": {
                "location": "s3://archive/team-b",
                "retention": "30d",
            },
        }
    }
    mock_workspace_store.create_workspace.assert_called_once()
    args, _ = mock_workspace_store.create_workspace.call_args
    assert args[0].trace_archival_location == "s3://archive/team-b"
    assert args[0].trace_archival_retention == "30d"
    mock_tracking_store.get_experiment_by_name.assert_not_called()
    mock_tracking_store.create_experiment.assert_not_called()


def test_get_workspace_endpoint(app, mock_workspace_store):
    mock_workspace_store.get_workspace.return_value = Workspace(name="team-c", description="Team C")
    with app.test_client() as client:
        response = client.get("/api/3.0/mlflow/workspaces/team-c")

    assert response.status_code == 200
    payload = _workspace_to_json(response.get_data(True))
    assert payload == {"workspace": {"name": "team-c", "description": "Team C"}}
    mock_workspace_store.get_workspace.assert_called_once_with("team-c")


def test_update_workspace_endpoint(app, mock_workspace_store):
    updated = Workspace(
        name="team-d",
        description="Updated",
        trace_archival_location="s3://archive/team-d",
        trace_archival_retention="14d",
    )
    mock_workspace_store.update_workspace.return_value = updated
    with app.test_client() as client:
        response = client.patch(
            "/api/3.0/mlflow/workspaces/team-d",
            json={
                "description": "Updated",
                "trace_archival_config": {
                    "location": "s3://archive/team-d",
                    "retention": "14d",
                },
            },
        )

    assert response.status_code == 200
    payload = _workspace_to_json(response.get_data(True))
    assert payload == {
        "workspace": {
            "name": "team-d",
            "description": "Updated",
            "trace_archival_config": {
                "location": "s3://archive/team-d",
                "retention": "14d",
            },
        }
    }
    mock_workspace_store.update_workspace.assert_called_once()
    args, _ = mock_workspace_store.update_workspace.call_args
    assert args[0].trace_archival_location == "s3://archive/team-d"
    assert args[0].trace_archival_retention == "14d"


def test_update_default_workspace_allows_reserved_name(app, mock_workspace_store):
    updated = Workspace(name="default", default_artifact_root="s3://bucket/root")
    mock_workspace_store.update_workspace.return_value = updated

    with app.test_client() as client:
        response = client.patch(
            "/api/3.0/mlflow/workspaces/default",
            json={"default_artifact_root": "s3://bucket/root"},
        )

    assert response.status_code == 200
    payload = _workspace_to_json(response.get_data(True))
    assert payload == {
        "workspace": {"name": "default", "default_artifact_root": "s3://bucket/root"}
    }
    args, _ = mock_workspace_store.update_workspace.call_args
    assert args[0].name == "default"
    assert args[0].default_artifact_root == "s3://bucket/root"


def test_update_workspace_can_clear_default_artifact_root(
    app, mock_workspace_store, mock_tracking_store
):
    cleared = Workspace(name="team-clear", description=None, default_artifact_root=None)
    mock_workspace_store.update_workspace.return_value = cleared
    with app.test_client() as client:
        response = client.patch(
            "/api/3.0/mlflow/workspaces/team-clear",
            json={"default_artifact_root": " "},
        )

    assert response.status_code == 200
    payload = _workspace_to_json(response.get_data(True))
    assert payload == {"workspace": {"name": "team-clear"}}
    args, _ = mock_workspace_store.update_workspace.call_args
    assert isinstance(args[0], Workspace)
    assert args[0].name == "team-clear"
    # Handler passes "" to indicate "clear"; the store converts "" to None
    assert args[0].default_artifact_root == ""


def test_update_workspace_can_clear_trace_archival_location(app, mock_workspace_store):
    cleared = Workspace(name="team-clear", description=None, trace_archival_location=None)
    mock_workspace_store.update_workspace.return_value = cleared
    with app.test_client() as client:
        response = client.patch(
            "/api/3.0/mlflow/workspaces/team-clear",
            json={"trace_archival_config": {"location": ""}},
        )

    assert response.status_code == 200
    payload = _workspace_to_json(response.get_data(True))
    assert payload == {"workspace": {"name": "team-clear"}}
    args, _ = mock_workspace_store.update_workspace.call_args
    assert isinstance(args[0], Workspace)
    assert args[0].name == "team-clear"
    assert args[0].trace_archival_location == ""


def test_update_workspace_can_clear_trace_archival_retention(app, mock_workspace_store):
    cleared = Workspace(name="team-clear", description=None, trace_archival_retention=None)
    mock_workspace_store.update_workspace.return_value = cleared
    with app.test_client() as client:
        response = client.patch(
            "/api/3.0/mlflow/workspaces/team-clear",
            json={"trace_archival_config": {"retention": ""}},
        )

    assert response.status_code == 200
    payload = _workspace_to_json(response.get_data(True))
    assert payload == {"workspace": {"name": "team-clear"}}
    args, _ = mock_workspace_store.update_workspace.call_args
    assert isinstance(args[0], Workspace)
    assert args[0].name == "team-clear"
    assert args[0].trace_archival_retention == ""


def test_update_workspace_with_only_trace_archival_location(app, mock_workspace_store):
    updated = Workspace(name="team-only-location", trace_archival_location="s3://archive/team-only")
    mock_workspace_store.update_workspace.return_value = updated
    with app.test_client() as client:
        response = client.patch(
            "/api/3.0/mlflow/workspaces/team-only-location",
            json={"trace_archival_config": {"location": "s3://archive/team-only"}},
        )

    assert response.status_code == 200
    payload = _workspace_to_json(response.get_data(True))
    assert payload == {
        "workspace": {
            "name": "team-only-location",
            "trace_archival_config": {"location": "s3://archive/team-only"},
        }
    }
    args, _ = mock_workspace_store.update_workspace.call_args
    assert args[0].trace_archival_location == "s3://archive/team-only"
    assert args[0].trace_archival_retention is None


def test_update_workspace_with_only_trace_archival_retention(app, mock_workspace_store):
    updated = Workspace(name="team-only-retention", trace_archival_retention="14d")
    mock_workspace_store.update_workspace.return_value = updated
    with app.test_client() as client:
        response = client.patch(
            "/api/3.0/mlflow/workspaces/team-only-retention",
            json={"trace_archival_config": {"retention": "14d"}},
        )

    assert response.status_code == 200
    payload = _workspace_to_json(response.get_data(True))
    assert payload == {
        "workspace": {
            "name": "team-only-retention",
            "trace_archival_config": {"retention": "14d"},
        }
    }
    args, _ = mock_workspace_store.update_workspace.call_args
    assert args[0].trace_archival_location is None
    assert args[0].trace_archival_retention == "14d"


def test_create_workspace_rejects_invalid_trace_archival_retention(
    app, mock_workspace_store, mock_tracking_store
):
    with app.test_client() as client:
        response = client.post(
            "/api/3.0/mlflow/workspaces",
            json={"name": "team-bad", "trace_archival_config": {"retention": "90days"}},
        )

    assert response.status_code == 400
    payload = _workspace_to_json(response.get_data(True))
    assert payload["message"] == (
        "Invalid value for 'trace_archival_config.retention'. Expected a duration in the form "
        "`<int><unit>`, where unit is one of 'm', 'h', or 'd' (for example '30d' or '12h')."
    )
    mock_workspace_store.create_workspace.assert_not_called()


def test_create_workspace_rejects_invalid_trace_archival_location(
    app, mock_workspace_store, mock_tracking_store
):
    with app.test_client() as client:
        response = client.post(
            "/api/3.0/mlflow/workspaces",
            json={
                "name": "team-bad",
                "trace_archival_config": {"location": "s3://archive/team#fragment"},
            },
        )

    assert response.status_code == 400
    payload = _workspace_to_json(response.get_data(True))
    assert "trace_archival_config.location" in payload["message"]
    mock_workspace_store.create_workspace.assert_not_called()


def test_create_workspace_rejects_proxy_only_trace_archival_location(
    app, mock_workspace_store, mock_tracking_store
):
    with app.test_client() as client:
        response = client.post(
            "/api/3.0/mlflow/workspaces",
            json={
                "name": "team-proxy",
                "trace_archival_config": {"location": "mlflow-artifacts:/archive/team-proxy"},
            },
        )

    assert response.status_code == 400
    payload = _workspace_to_json(response.get_data(True))
    assert payload["message"] == (
        "Invalid value for 'trace_archival_config.location'. Trace archival location cannot use "
        "the proxy-only `mlflow-artifacts:` scheme."
    )
    mock_workspace_store.create_workspace.assert_not_called()


def test_update_workspace_rejects_invalid_trace_archival_retention(app, mock_workspace_store):
    with app.test_client() as client:
        response = client.patch(
            "/api/3.0/mlflow/workspaces/team-bad",
            json={"trace_archival_config": {"retention": "5w"}},
        )

    assert response.status_code == 400
    payload = _workspace_to_json(response.get_data(True))
    assert payload["message"] == (
        "Invalid value for 'trace_archival_config.retention'. Expected a duration in the form "
        "`<int><unit>`, where unit is one of 'm', 'h', or 'd' (for example '30d' or '12h')."
    )
    mock_workspace_store.update_workspace.assert_not_called()


def test_create_workspace_rejects_trace_archival_retention_longer_than_32_chars(
    app, mock_workspace_store, mock_tracking_store
):
    with app.test_client() as client:
        response = client.post(
            "/api/3.0/mlflow/workspaces",
            json={"name": "team-bad", "trace_archival_config": {"retention": f"{'1' * 32}d"}},
        )

    assert response.status_code == 400
    payload = _workspace_to_json(response.get_data(True))
    assert payload["message"] == (
        "Invalid value for 'trace_archival_config.retention'. Maximum length is 32 characters."
    )
    mock_workspace_store.create_workspace.assert_not_called()


def test_update_workspace_rejects_trace_archival_retention_longer_than_32_chars(
    app, mock_workspace_store
):
    with app.test_client() as client:
        response = client.patch(
            "/api/3.0/mlflow/workspaces/team-bad",
            json={"trace_archival_config": {"retention": f"{'1' * 32}d"}},
        )

    assert response.status_code == 400
    payload = _workspace_to_json(response.get_data(True))
    assert payload["message"] == (
        "Invalid value for 'trace_archival_config.retention'. Maximum length is 32 characters."
    )
    mock_workspace_store.update_workspace.assert_not_called()


def test_delete_workspace_endpoint(app, mock_workspace_store):
    with app.test_client() as client:
        response = client.delete("/api/3.0/mlflow/workspaces/team-e")

    assert response.status_code == 204
    mock_workspace_store.delete_workspace.assert_called_once_with(
        "team-e", mode=WorkspaceDeletionMode.RESTRICT
    )


def test_delete_default_workspace_rejected_by_validation(app, mock_workspace_store):
    with app.test_client() as client:
        response = client.delete("/api/3.0/mlflow/workspaces/default")

    assert response.status_code == 400
    payload = _workspace_to_json(response.get_data(True))
    assert "cannot be deleted" in payload["message"]
    mock_workspace_store.delete_workspace.assert_not_called()


def test_create_workspace_fails_without_artifact_root(app, mock_workspace_store, monkeypatch):
    tracking_store = mock.Mock()
    tracking_store.artifact_root_uri = None
    monkeypatch.setattr(
        "mlflow.server.handlers._get_tracking_store",
        lambda *_, **__: tracking_store,
    )
    with app.test_client() as client:
        response = client.post(
            "/api/3.0/mlflow/workspaces",
            json={"name": "team-no-root"},
        )

    assert response.status_code == 400
    payload = _workspace_to_json(response.get_data(True))
    assert "artifact root" in payload["message"].lower()


def test_create_workspace_with_artifact_root_succeeds_without_server_default(
    app, mock_workspace_store, monkeypatch
):
    tracking_store = mock.Mock()
    tracking_store.artifact_root_uri = None
    monkeypatch.setattr(
        "mlflow.server.handlers._get_tracking_store",
        lambda *_, **__: tracking_store,
    )
    created = Workspace(name="team-with-root", default_artifact_root="s3://bucket/path")
    mock_workspace_store.create_workspace.return_value = created
    with app.test_client() as client:
        response = client.post(
            "/api/3.0/mlflow/workspaces",
            json={"name": "team-with-root", "default_artifact_root": "s3://bucket/path"},
        )

    assert response.status_code == 201


def test_create_default_workspace_rejected(app, mock_workspace_store, mock_tracking_store):
    with app.test_client() as client:
        response = client.post(
            "/api/3.0/mlflow/workspaces",
            json={"name": "default"},
        )

    assert response.status_code == 400
    payload = _workspace_to_json(response.get_data(True))
    assert "reserved" in payload["message"]
    mock_workspace_store.create_workspace.assert_not_called()


def test_update_workspace_clear_artifact_root_fails_without_server_default(
    app, mock_workspace_store, monkeypatch
):
    tracking_store = mock.Mock()
    tracking_store.artifact_root_uri = None
    monkeypatch.setattr(
        "mlflow.server.handlers._get_tracking_store",
        lambda *_, **__: tracking_store,
    )
    with app.test_client() as client:
        response = client.patch(
            "/api/3.0/mlflow/workspaces/team-clear",
            json={"default_artifact_root": ""},
        )

    assert response.status_code == 400
    payload = _workspace_to_json(response.get_data(True))
    assert "artifact root" in payload["message"].lower()
