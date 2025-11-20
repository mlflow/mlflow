from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from flask import g, request

from mlflow.entities import Experiment, Workspace
from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.server import app, handlers, workspace_helpers
from mlflow.server.handlers import _create_workspace_handler, _delete_workspace_handler
from mlflow.tracking._workspace import context as workspace_context


@pytest.fixture(autouse=True)
def _reset_handler_state():
    workspace_helpers._workspace_store = None
    handlers._tracking_store = None
    handlers._model_registry_store = None
    workspace_context.clear_workspace()
    yield
    workspace_helpers._workspace_store = None
    handlers._tracking_store = None
    handlers._model_registry_store = None
    workspace_context.clear_workspace()


def test_create_workspace_handler_creates_default_experiment(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")

    class DummyWorkspaceStore:
        def __init__(self):
            self.created = []

        def create_workspace(self, workspace, request):
            self.created.append(workspace.name)
            return workspace

    class DummyTrackingStore:
        def __init__(self):
            self.created_names: list[tuple[str, str]] = []
            self._experiments: dict[tuple[str | None, str], SimpleNamespace] = {}

        def get_experiment_by_name(self, name: str):
            key = (workspace_context.get_current_workspace(), name)
            return self._experiments.get(key)

        def create_experiment(self, name: str):
            key = (workspace_context.get_current_workspace(), name)
            self.created_names.append(key)
            self._experiments[key] = SimpleNamespace(
                name=name, workspace=workspace_context.get_current_workspace()
            )

        def search_experiments(self, **_):
            return list(self._experiments.values())

    dummy_store = DummyWorkspaceStore()
    dummy_tracking_store = DummyTrackingStore()

    monkeypatch.setattr(handlers, "_get_workspace_store", lambda *args, **kwargs: dummy_store)
    monkeypatch.setattr(
        handlers, "_get_tracking_store", lambda *args, **kwargs: dummy_tracking_store
    )
    monkeypatch.setattr(handlers, "_get_model_registry_store", lambda *args, **kwargs: None)

    with app.test_request_context(
        "/mlflow/workspaces", method="POST", json={"name": "team-a", "description": None}
    ):
        response = _create_workspace_handler()

    assert response.status_code == 201
    assert dummy_store.created == ["team-a"]
    assert dummy_tracking_store.created_names == [("team-a", Experiment.DEFAULT_EXPERIMENT_NAME)]


def test_delete_workspace_handler_rejects_non_empty_workspace(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")

    class DummyWorkspaceStore:
        def __init__(self):
            self.deleted = False

        def delete_workspace(self, workspace_name, request):
            self.deleted = True

    class DummyTrackingStore:
        def get_experiment_by_name(self, name: str):
            return None

        def create_experiment(self, name: str):
            raise NotImplementedError

        def search_experiments(self, view_type=None, max_results=None, **__):
            workspace = workspace_context.get_current_workspace()
            return [
                SimpleNamespace(name="exp-1", workspace=workspace, experiment_id="1"),
                SimpleNamespace(
                    name=Experiment.DEFAULT_EXPERIMENT_NAME, workspace=workspace, experiment_id="0"
                ),
            ]

        def search_runs(self, *_, **__):
            return []

    dummy_store = DummyWorkspaceStore()
    dummy_tracking_store = DummyTrackingStore()

    monkeypatch.setattr(handlers, "_get_workspace_store", lambda *args, **kwargs: dummy_store)
    monkeypatch.setattr(
        handlers, "_get_tracking_store", lambda *args, **kwargs: dummy_tracking_store
    )
    monkeypatch.setattr(handlers, "_get_model_registry_store", lambda *args, **kwargs: None)

    with app.test_request_context("/mlflow/workspaces/team-a", method="DELETE"):
        response = _delete_workspace_handler("team-a")

    assert response.status_code == 500
    payload = json.loads(response.get_data())
    assert payload["message"] == "Cannot delete workspace 'team-a' because it contains resources"
    assert payload["error_code"] == "INVALID_STATE"
    assert not dummy_store.deleted


def test_workspace_before_request_handles_artifact_routes(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")

    class DummyWorkspaceStore:
        def get_workspace(self, workspace_name, request):
            return Workspace(name=workspace_name)

    dummy_store = DummyWorkspaceStore()
    monkeypatch.setattr(handlers, "_get_workspace_store", lambda *_, **__: dummy_store)
    monkeypatch.setattr(workspace_helpers, "_get_workspace_store", lambda *_, **__: dummy_store)
    monkeypatch.setattr(handlers, "_get_tracking_store", lambda *_, **__: None)
    monkeypatch.setattr(handlers, "_get_model_registry_store", lambda *_, **__: None)

    with app.test_request_context(
        "/mlflow-artifacts/workspaces/team-a/artifacts/foo", method="GET"
    ):
        request.view_args = {"workspace_name": "team-a"}
        response = handlers.workspace_before_request_handler()
        assert response is None
        assert g.mlflow_workspace.name == "team-a"
        handlers.workspace_teardown_request_handler(None)


def test_delete_workspace_handler_rejects_default_experiment_with_runs(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")

    class DummyWorkspaceStore:
        def __init__(self):
            self.deleted = False

        def delete_workspace(self, workspace_name, request):
            self.deleted = True

    class DummyTrackingStore:
        def get_experiment_by_name(self, name: str):
            workspace = workspace_context.get_current_workspace()
            if name == Experiment.DEFAULT_EXPERIMENT_NAME:
                return SimpleNamespace(
                    name=name,
                    workspace=workspace,
                    experiment_id="0",
                )
            return None

        def search_experiments(self, filter_string=None, **_):
            if filter_string and "name !=" in filter_string:
                return []
            workspace = workspace_context.get_current_workspace()
            return [
                SimpleNamespace(
                    name=Experiment.DEFAULT_EXPERIMENT_NAME, workspace=workspace, experiment_id="0"
                )
            ]

        def search_runs(self, experiment_ids, **_):
            assert experiment_ids == ["0"]
            return [SimpleNamespace()]

    dummy_store = DummyWorkspaceStore()
    dummy_tracking_store = DummyTrackingStore()

    monkeypatch.setattr(handlers, "_get_workspace_store", lambda *args, **kwargs: dummy_store)
    monkeypatch.setattr(
        handlers, "_get_tracking_store", lambda *args, **kwargs: dummy_tracking_store
    )
    monkeypatch.setattr(handlers, "_get_model_registry_store", lambda *args, **kwargs: None)

    with app.test_request_context("/mlflow/workspaces/team-a", method="DELETE"):
        response = _delete_workspace_handler("team-a")

    assert response.status_code == 500
    payload = json.loads(response.get_data())
    assert payload["message"] == "Cannot delete workspace 'team-a' because it contains resources"
    assert payload["error_code"] == "INVALID_STATE"
    assert not dummy_store.deleted


def test_delete_workspace_handler_succeeds_for_empty_workspace(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")

    class DummyWorkspaceStore:
        def __init__(self):
            self.deleted = False

        def delete_workspace(self, workspace_name, request):
            self.deleted = True

    class DummyTrackingStore:
        def get_experiment_by_name(self, name: str):
            workspace = workspace_context.get_current_workspace()
            if name == Experiment.DEFAULT_EXPERIMENT_NAME:
                return SimpleNamespace(
                    name=name,
                    workspace=workspace,
                    experiment_id="0",
                )
            return None

        def search_experiments(self, filter_string=None, **_):
            if filter_string and "name !=" in filter_string:
                return []
            workspace = workspace_context.get_current_workspace()
            return [
                SimpleNamespace(
                    name=Experiment.DEFAULT_EXPERIMENT_NAME, workspace=workspace, experiment_id="0"
                )
            ]

        def search_runs(self, experiment_ids, **_):
            return []

    dummy_store = DummyWorkspaceStore()
    dummy_tracking_store = DummyTrackingStore()

    monkeypatch.setattr(handlers, "_get_workspace_store", lambda *args, **kwargs: dummy_store)
    monkeypatch.setattr(
        handlers, "_get_tracking_store", lambda *args, **kwargs: dummy_tracking_store
    )
    monkeypatch.setattr(handlers, "_get_model_registry_store", lambda *args, **kwargs: None)

    with app.test_request_context("/mlflow/workspaces/team-a", method="DELETE"):
        response = _delete_workspace_handler("team-a")

    assert response.status_code == 204
    assert dummy_store.deleted
