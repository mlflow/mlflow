from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from mlflow.entities import Experiment
from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.exceptions import MlflowException
from mlflow.server import app, handlers, workspace_helpers
from mlflow.server.handlers import _create_workspace_handler, _delete_workspace_handler
from mlflow.utils import workspace_context


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

        def create_workspace(self, workspace):
            self.created.append(workspace.name)
            return workspace

    class DummyTrackingStore:
        def __init__(self):
            self.created_names: list[tuple[str, str]] = []
            self._experiments: dict[tuple[str | None, str], SimpleNamespace] = {}

        def get_experiment_by_name(self, name: str):
            key = (workspace_context.get_request_workspace(), name)
            return self._experiments.get(key)

        def create_experiment(self, name: str):
            key = (workspace_context.get_request_workspace(), name)
            self.created_names.append(key)
            self._experiments[key] = SimpleNamespace(
                name=name, workspace=workspace_context.get_request_workspace()
            )

        def search_experiments(self, **_):
            return list(self._experiments.values())

    class DummyJobStore:
        def list_jobs(self):
            return iter([SimpleNamespace()])

    class DummyRegistryStore:
        def search_registered_models(self, **_):
            return []

        def list_webhooks(self, max_results=1):
            return [SimpleNamespace()]

    dummy_store = DummyWorkspaceStore()
    dummy_tracking_store = DummyTrackingStore()
    dummy_job_store = DummyJobStore()
    dummy_registry_store = DummyRegistryStore()

    monkeypatch.setattr(handlers, "_get_workspace_store", lambda *args, **kwargs: dummy_store)
    monkeypatch.setattr(
        handlers, "_get_tracking_store", lambda *args, **kwargs: dummy_tracking_store
    )
    monkeypatch.setattr(
        handlers, "_get_model_registry_store", lambda *args, **kwargs: dummy_registry_store
    )
    monkeypatch.setattr(handlers, "_get_job_store", lambda *args, **kwargs: dummy_job_store)

    with app.test_request_context(
        "/mlflow/workspaces", method="POST", json={"name": "team-a", "description": None}
    ):
        response = _create_workspace_handler()

    assert response.status_code == 201
    assert dummy_store.created == ["team-a"]
    assert dummy_tracking_store.created_names == [("team-a", Experiment.DEFAULT_EXPERIMENT_NAME)]


@pytest.mark.parametrize(
    ("has_experiments", "has_jobs", "has_models", "has_webhooks"),
    [
        (True, False, False, False),
        (False, True, False, False),
        (False, False, True, False),
        (False, False, False, True),
    ],
)
def test_delete_workspace_handler_rejects_non_empty_workspace(
    monkeypatch, has_experiments, has_jobs, has_models, has_webhooks
):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")

    class DummyWorkspaceStore:
        def __init__(self):
            self.deleted = False

        def delete_workspace(self, workspace_name):
            self.deleted = True

    class DummyTrackingStore:
        def __init__(self, has_experiments: bool):
            self._has_experiments = has_experiments

        def get_experiment_by_name(self, name: str):
            return None

        def create_experiment(self, name: str):
            raise NotImplementedError

        def search_experiments(self, view_type=None, max_results=None, **__):
            if not self._has_experiments:
                return []
            workspace = workspace_context.get_request_workspace()
            return [
                SimpleNamespace(name="exp-1", workspace=workspace, experiment_id="1"),
            ]

        def search_runs(self, *_, **__):
            return []

    class DummyJobStore:
        def __init__(self, has_jobs: bool):
            self._has_jobs = has_jobs

        def list_jobs(self):
            if self._has_jobs:
                return iter([SimpleNamespace()])
            return iter([])

    class DummyRegistryStore:
        def __init__(self, has_models: bool, has_webhooks: bool):
            self._has_models = has_models
            self._has_webhooks = has_webhooks

        def search_registered_models(self, **_):
            return [SimpleNamespace()] if self._has_models else []

        def list_webhooks(self, max_results=1):
            if self._has_webhooks:
                return [SimpleNamespace()]
            return []

    dummy_store = DummyWorkspaceStore()
    dummy_tracking_store = DummyTrackingStore(has_experiments)
    dummy_job_store = DummyJobStore(has_jobs)
    dummy_registry_store = DummyRegistryStore(has_models, has_webhooks)

    monkeypatch.setattr(handlers, "_get_workspace_store", lambda *args, **kwargs: dummy_store)
    monkeypatch.setattr(
        handlers, "_get_tracking_store", lambda *args, **kwargs: dummy_tracking_store
    )
    monkeypatch.setattr(
        handlers, "_get_model_registry_store", lambda *args, **kwargs: dummy_registry_store
    )
    monkeypatch.setattr(handlers, "_get_job_store", lambda *_, **__: dummy_job_store)

    with app.test_request_context("/mlflow/workspaces/team-a", method="DELETE"):
        response = _delete_workspace_handler("team-a")

    assert response.status_code == 500
    payload = json.loads(response.get_data())
    assert payload["message"] == "Cannot delete workspace 'team-a' because it contains resources"
    assert payload["error_code"] == "INVALID_STATE"
    assert not dummy_store.deleted


def test_delete_workspace_handler_rejects_default_experiment_with_runs(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")

    class DummyWorkspaceStore:
        def __init__(self):
            self.deleted = False

        def delete_workspace(self, workspace_name):
            self.deleted = True

    class DummyTrackingStore:
        def get_experiment_by_name(self, name: str):
            workspace = workspace_context.get_request_workspace()
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
            workspace = workspace_context.get_request_workspace()
            return [
                SimpleNamespace(
                    name=Experiment.DEFAULT_EXPERIMENT_NAME, workspace=workspace, experiment_id="0"
                )
            ]

        def search_runs(self, experiment_ids, **_):
            assert experiment_ids == ["0"]
            return [SimpleNamespace()]

    class DummyJobStore:
        def list_jobs(self):
            return iter([])

    dummy_store = DummyWorkspaceStore()
    dummy_tracking_store = DummyTrackingStore()
    dummy_job_store = DummyJobStore()

    monkeypatch.setattr(handlers, "_get_workspace_store", lambda *args, **kwargs: dummy_store)
    monkeypatch.setattr(
        handlers, "_get_tracking_store", lambda *args, **kwargs: dummy_tracking_store
    )
    monkeypatch.setattr(handlers, "_get_model_registry_store", lambda *args, **kwargs: None)
    monkeypatch.setattr(handlers, "_get_job_store", lambda *args, **kwargs: dummy_job_store)

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

        def delete_workspace(self, workspace_name):
            self.deleted = True

    class DummyTrackingStore:
        def get_experiment_by_name(self, name: str):
            workspace = workspace_context.get_request_workspace()
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
            workspace = workspace_context.get_request_workspace()
            return [
                SimpleNamespace(
                    name=Experiment.DEFAULT_EXPERIMENT_NAME, workspace=workspace, experiment_id="0"
                )
            ]

        def search_runs(self, experiment_ids, **_):
            return []

    class DummyJobStore:
        def list_jobs(self):
            return iter([])

    dummy_store = DummyWorkspaceStore()
    dummy_tracking_store = DummyTrackingStore()
    dummy_job_store = DummyJobStore()

    monkeypatch.setattr(handlers, "_get_workspace_store", lambda *args, **kwargs: dummy_store)
    monkeypatch.setattr(
        handlers, "_get_tracking_store", lambda *args, **kwargs: dummy_tracking_store
    )
    monkeypatch.setattr(handlers, "_get_model_registry_store", lambda *args, **kwargs: None)
    monkeypatch.setattr(handlers, "_get_job_store", lambda *args, **kwargs: dummy_job_store)

    with app.test_request_context("/mlflow/workspaces/team-a", method="DELETE"):
        response = _delete_workspace_handler("team-a")

    assert response.status_code == 204
    assert dummy_store.deleted


def test_get_workspace_scoped_repo_path_if_enabled_prefixes_workspace(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")

    with workspace_context.WorkspaceContext("team-a"):
        assert (
            handlers._get_workspace_scoped_repo_path_if_enabled("5/abc")
            == "workspaces/team-a/5/abc"
        )
        assert handlers._get_workspace_scoped_repo_path_if_enabled(None) == "workspaces/team-a"
        assert (
            handlers._get_workspace_scoped_repo_path_if_enabled("workspaces/team-a/5/abc")
            == "workspaces/team-a/5/abc"
        )

    workspace_context.clear_workspace()
    with pytest.raises(MlflowException, match="Active workspace is required"):
        handlers._get_workspace_scoped_repo_path_if_enabled("foo/bar")

    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")
    assert handlers._get_workspace_scoped_repo_path_if_enabled("foo/bar") == "foo/bar"


def test_workspace_scoped_repo_path_blocks_cross_workspace_access(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")

    with workspace_context.WorkspaceContext("team-a"):
        with pytest.raises(MlflowException, match="team-b"):
            handlers._get_workspace_scoped_repo_path_if_enabled("workspaces/team-b/5/abc")

        with pytest.raises(MlflowException, match="include a workspace name"):
            handlers._get_workspace_scoped_repo_path_if_enabled("workspaces/")
