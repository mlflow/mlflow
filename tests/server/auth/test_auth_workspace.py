import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
from flask import Response, request

from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.server import auth as auth_module
from mlflow.server.auth.permissions import MANAGE, NO_PERMISSIONS, READ
from mlflow.server.auth.routes import (
    CREATE_PROMPTLAB_RUN,
    GET_ARTIFACT,
    GET_METRIC_HISTORY_BULK,
    GET_METRIC_HISTORY_BULK_INTERVAL,
    GET_MODEL_VERSION_ARTIFACT,
    GET_TRACE_ARTIFACT,
    SEARCH_DATASETS,
    UPLOAD_ARTIFACT,
)
from mlflow.server.auth.sqlalchemy_store import SqlAlchemyStore
from mlflow.utils import workspace_context
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

from tests.helper_functions import random_str


def test_cleanup_workspace_permissions_handler(monkeypatch):
    mock_delete = Mock()

    monkeypatch.setattr(
        auth_module.store,
        "delete_workspace_permissions_for_workspace",
        mock_delete,
        raising=True,
    )

    workspace_name = f"team-{random_str(10)}"
    with auth_module.app.test_request_context(
        f"/api/3.0/mlflow/workspaces/{workspace_name}", method="DELETE"
    ):
        request.view_args = {"workspace_name": workspace_name}
        response = Response(status=204)
        auth_module._after_request(response)

    mock_delete.assert_called_once_with(workspace_name)


class _TrackingStore:
    def __init__(
        self,
        experiment_workspaces: dict[str, str],
        run_experiments: dict[str, str],
        trace_experiments: dict[str, str],
        experiment_names: dict[str, str] | None = None,
        logged_model_experiments: dict[str, str] | None = None,
        gateway_secret_workspaces: dict[str, str] | None = None,
        gateway_endpoint_workspaces: dict[str, str] | None = None,
        gateway_model_def_workspaces: dict[str, str] | None = None,
        engine=None,
        ManagedSessionMaker=None,
    ):
        self._experiment_workspaces = experiment_workspaces
        self._run_experiments = run_experiments
        self._trace_experiments = trace_experiments
        self._experiment_names = experiment_names or {}
        self._logged_model_experiments = logged_model_experiments or {}
        self._gateway_secret_workspaces = gateway_secret_workspaces or {}
        self._gateway_endpoint_workspaces = gateway_endpoint_workspaces or {}
        self._gateway_model_def_workspaces = gateway_model_def_workspaces or {}
        self.engine = engine
        self.ManagedSessionMaker = ManagedSessionMaker

    def get_experiment(self, experiment_id: str):
        return SimpleNamespace(workspace=self._experiment_workspaces[experiment_id])

    def get_experiment_by_name(self, experiment_name: str):
        experiment_id = self._experiment_names.get(experiment_name)
        if experiment_id is None:
            return None
        return SimpleNamespace(
            experiment_id=experiment_id,
            workspace=self._experiment_workspaces[experiment_id],
        )

    def get_run(self, run_id: str):
        return SimpleNamespace(info=SimpleNamespace(experiment_id=self._run_experiments[run_id]))

    def get_trace_info(self, request_id: str):
        return SimpleNamespace(experiment_id=self._trace_experiments[request_id])

    def get_logged_model(self, model_id: str):
        experiment_id = self._logged_model_experiments[model_id]
        return SimpleNamespace(experiment_id=experiment_id)

    def get_secret_info(self, secret_id: str | None = None, secret_name: str | None = None):
        if secret_id:
            if secret_id not in self._gateway_secret_workspaces:
                raise MlflowException(
                    f"GatewaySecret not found ({secret_id})",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            # Add workspace attribute so _get_resource_workspace can extract it
            return SimpleNamespace(
                secret_id=secret_id, workspace=self._gateway_secret_workspaces[secret_id]
            )
        raise ValueError("Must provide secret_id or secret_name")

    def get_gateway_endpoint(self, endpoint_id: str | None = None, name: str | None = None):
        if endpoint_id:
            if endpoint_id not in self._gateway_endpoint_workspaces:
                raise MlflowException(
                    f"GatewayEndpoint not found ({endpoint_id})",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            # Add workspace attribute so _get_resource_workspace can extract it
            return SimpleNamespace(
                endpoint_id=endpoint_id, workspace=self._gateway_endpoint_workspaces[endpoint_id]
            )
        raise ValueError("Must provide endpoint_id or name")

    def get_gateway_model_definition(
        self, model_definition_id: str | None = None, name: str | None = None
    ):
        if model_definition_id:
            if model_definition_id not in self._gateway_model_def_workspaces:
                raise MlflowException(
                    f"GatewayModelDefinition not found ({model_definition_id})",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            # Add workspace attribute so _get_resource_workspace can extract it
            return SimpleNamespace(
                model_definition_id=model_definition_id,
                workspace=self._gateway_model_def_workspaces[model_definition_id],
            )
        raise ValueError("Must provide model_definition_id or name")

    def _create_mock_session(self):
        """Create a mock session that can query gateway SQL models."""
        mock_session = MagicMock()

        def _filter_by_secret_id(secret_id):
            if secret_id in self._gateway_secret_workspaces:
                mock_result = MagicMock()
                mock_result.first.return_value = SimpleNamespace(
                    workspace=self._gateway_secret_workspaces[secret_id]
                )
                return mock_result
            mock_result = MagicMock()
            mock_result.first.return_value = None
            return mock_result

        def _filter_by_endpoint_id(endpoint_id):
            if endpoint_id in self._gateway_endpoint_workspaces:
                mock_result = MagicMock()
                mock_result.first.return_value = SimpleNamespace(
                    workspace=self._gateway_endpoint_workspaces[endpoint_id]
                )
                return mock_result
            mock_result = MagicMock()
            mock_result.first.return_value = None
            return mock_result

        def _filter_by_model_def_id(model_definition_id):
            if model_definition_id in self._gateway_model_def_workspaces:
                mock_result = MagicMock()
                mock_result.first.return_value = SimpleNamespace(
                    workspace=self._gateway_model_def_workspaces[model_definition_id]
                )
                return mock_result
            mock_result = MagicMock()
            mock_result.first.return_value = None
            return mock_result

        def _query(model_class):
            mock_query_result = MagicMock()
            # Mock the filter method to return different results based on the filter

            def _mock_filter(*args, **kwargs):
                if "secret_id" in kwargs:
                    return _filter_by_secret_id(kwargs["secret_id"])
                elif "endpoint_id" in kwargs:
                    return _filter_by_endpoint_id(kwargs["endpoint_id"])
                elif "model_definition_id" in kwargs:
                    return _filter_by_model_def_id(kwargs["model_definition_id"])
                return mock_query_result

            mock_query_result.filter = _mock_filter
            return mock_query_result

        mock_session.query = _query
        return mock_session

    def _create_mock_session_maker(self):
        """Create a mock ManagedSessionMaker context manager."""

        @contextmanager
        def _mock_session_maker():
            yield self._create_mock_session()

        return _mock_session_maker


class _RegistryStore:
    def __init__(self, model_workspaces: dict[str, str]):
        self._model_workspaces = model_workspaces

    def get_registered_model(self, name: str):
        return SimpleNamespace(workspace=self._model_workspaces[name])


@pytest.fixture
def workspace_permission_setup(tmp_path, monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    monkeypatch.setattr(
        auth_module,
        "auth_config",
        auth_module.auth_config._replace(default_permission=NO_PERMISSIONS.name),
    )

    db_uri = f"sqlite:///{tmp_path / 'auth-store.db'}"
    auth_store = SqlAlchemyStore()
    auth_store.init_db(db_uri)
    monkeypatch.setattr(auth_module, "store", auth_store, raising=False)

    username = "alice"
    auth_store.create_user(username, "supersecurepassword", is_admin=False)

    tracking_store = _TrackingStore(
        experiment_workspaces={"exp-1": "team-a", "exp-2": "team-a", "1": "team-a"},
        run_experiments={"run-1": "exp-1", "run-2": "exp-2"},
        trace_experiments={"trace-1": "exp-1"},
        experiment_names={"Primary Experiment": "exp-1"},
        logged_model_experiments={"model-1": "exp-1"},
        gateway_secret_workspaces={"secret-1": "team-a", "secret-2": "team-a"},
        gateway_endpoint_workspaces={"endpoint-1": "team-a", "endpoint-2": "team-a"},
        gateway_model_def_workspaces={"model-def-1": "team-a", "model-def-2": "team-a"},
        engine=MagicMock(),  # Mock engine for SQL model queries
    )
    # Set ManagedSessionMaker after creating the store
    tracking_store.ManagedSessionMaker = tracking_store._create_mock_session_maker()
    monkeypatch.setattr(auth_module, "_get_tracking_store", lambda: tracking_store)

    registry_store = _RegistryStore({"model-xyz": "team-a"})
    monkeypatch.setattr(auth_module, "_get_model_registry_store", lambda: registry_store)

    monkeypatch.setattr(
        auth_module,
        "authenticate_request",
        lambda: SimpleNamespace(username=username),
    )

    auth_store.set_workspace_permission("team-a", username, MANAGE.name)

    with workspace_context.WorkspaceContext("team-a"):
        yield {"store": auth_store, "username": username}
    auth_store.engine.dispose()


def _set_workspace_permission(store: SqlAlchemyStore, username: str, permission: str):
    store.set_workspace_permission("team-a", username, permission)


def test_workspace_permission_grants_default_access(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")

    default_permission = MANAGE.name
    monkeypatch.setattr(
        auth_module,
        "auth_config",
        auth_module.auth_config._replace(
            default_permission=default_permission,
            grant_default_workspace_access=True,
        ),
        raising=False,
    )

    class DummyStore:
        def get_workspace_permission(self, workspace_name, username):
            return None

        def list_accessible_workspace_names(self, username):
            return []

    dummy_store = DummyStore()
    monkeypatch.setattr(auth_module, "store", dummy_store, raising=False)

    default_workspace = DEFAULT_WORKSPACE_NAME
    monkeypatch.setattr(auth_module, "_get_workspace_store", lambda: None, raising=False)
    monkeypatch.setattr(
        auth_module,
        "get_default_workspace_optional",
        lambda *args, **kwargs: (SimpleNamespace(name=default_workspace), True),
        raising=False,
    )

    auth = SimpleNamespace(username="alice")
    permission = auth_module._workspace_permission(auth.username, default_workspace)
    assert permission is not None
    assert permission.can_manage

    with workspace_context.WorkspaceContext(default_workspace):
        monkeypatch.setattr(auth_module, "authenticate_request", lambda: auth)
        assert auth_module.validate_can_create_experiment()


def test_filter_list_workspaces_includes_default_when_autogrant(monkeypatch):
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    auth = SimpleNamespace(username="alice")
    monkeypatch.setattr(auth_module, "authenticate_request", lambda: auth)
    monkeypatch.setattr(
        auth_module,
        "auth_config",
        auth_module.auth_config._replace(
            grant_default_workspace_access=True,
            default_permission=READ.name,
        ),
        raising=False,
    )

    default_workspace = "team-default"
    monkeypatch.setattr(auth_module, "_get_workspace_store", lambda: None, raising=False)
    monkeypatch.setattr(
        auth_module,
        "get_default_workspace_optional",
        lambda *args, **kwargs: (SimpleNamespace(name=default_workspace), True),
        raising=False,
    )

    class DummyStore:
        def list_accessible_workspace_names(self, username):
            return []

    monkeypatch.setattr(auth_module, "store", DummyStore(), raising=False)

    response = Response(
        json.dumps(
            {
                "workspaces": [
                    {"name": default_workspace},
                    {"name": "other-workspace"},
                ]
            }
        ),
        mimetype="application/json",
    )

    auth_module.filter_list_workspaces(response)
    payload = json.loads(response.get_data(as_text=True))
    assert payload["workspaces"] == [{"name": default_workspace}]


def test_filter_list_workspaces_filters_to_allowed(monkeypatch):
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    auth = SimpleNamespace(username="alice")
    monkeypatch.setattr(auth_module, "authenticate_request", lambda: auth)
    monkeypatch.setattr(
        auth_module,
        "auth_config",
        auth_module.auth_config._replace(
            grant_default_workspace_access=False,
        ),
        raising=False,
    )

    class DummyStore:
        def list_accessible_workspace_names(self, username):
            return ["team-a"]

    monkeypatch.setattr(auth_module, "store", DummyStore(), raising=False)

    response = Response(
        json.dumps({"workspaces": [{"name": "team-a"}, {"name": "team-b"}]}),
        mimetype="application/json",
    )

    auth_module.filter_list_workspaces(response)
    payload = json.loads(response.get_data(as_text=True))
    assert [ws["name"] for ws in payload["workspaces"]] == ["team-a"]


def test_validate_can_view_workspace_allows_default_autogrant(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    auth = SimpleNamespace(username="alice")
    monkeypatch.setattr(auth_module, "authenticate_request", lambda: auth)
    monkeypatch.setattr(
        auth_module,
        "auth_config",
        auth_module.auth_config._replace(
            grant_default_workspace_access=True,
            default_permission=READ.name,
        ),
        raising=False,
    )

    default_workspace = "team-default"
    monkeypatch.setattr(auth_module, "_get_workspace_store", lambda: None, raising=False)
    monkeypatch.setattr(
        auth_module,
        "get_default_workspace_optional",
        lambda *args, **kwargs: (SimpleNamespace(name=default_workspace), True),
        raising=False,
    )

    class DummyStore:
        def list_accessible_workspace_names(self, username):
            return []

    monkeypatch.setattr(auth_module, "store", DummyStore(), raising=False)

    with auth_module.app.test_request_context(
        f"/api/3.0/mlflow/workspaces/{default_workspace}", method="GET"
    ):
        request.view_args = {"workspace_name": default_workspace}
        assert auth_module.validate_can_view_workspace()

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/workspaces/other-team", method="GET"
    ):
        request.view_args = {"workspace_name": "other-team"}
        assert not auth_module.validate_can_view_workspace()


def test_experiment_validators_allow_manage_permission(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, MANAGE.name)

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/experiments/get", method="GET", query_string={"experiment_id": "exp-1"}
    ):
        assert auth_module.validate_can_read_experiment()
        assert auth_module.validate_can_update_experiment()
        assert auth_module.validate_can_delete_experiment()
        assert auth_module.validate_can_manage_experiment()

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/experiments/get-by-name",
        method="GET",
        query_string={"experiment_name": "Primary Experiment"},
    ):
        assert auth_module.validate_can_read_experiment_by_name()

    with workspace_context.WorkspaceContext("team-a"):
        assert auth_module.validate_can_create_experiment()


def test_experiment_validators_read_permission_blocks_writes(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, READ.name)

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/experiments/get", method="GET", query_string={"experiment_id": "exp-1"}
    ):
        assert auth_module.validate_can_read_experiment()
        assert not auth_module.validate_can_update_experiment()
        assert not auth_module.validate_can_delete_experiment()
        assert not auth_module.validate_can_manage_experiment()

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/experiments/get-by-name",
        method="GET",
        query_string={"experiment_name": "Primary Experiment"},
    ):
        assert auth_module.validate_can_read_experiment_by_name()

    with workspace_context.WorkspaceContext("team-a"):
        assert not auth_module.validate_can_create_experiment()


def test_experiment_artifact_proxy_validators_respect_permissions(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, MANAGE.name)

    with auth_module.app.test_request_context(
        "/ajax-api/2.0/mlflow-artifacts/artifacts/1/path",
        method="GET",
    ):
        request.view_args = {"artifact_path": "1/path"}
        assert auth_module.validate_can_read_experiment_artifact_proxy()
        assert auth_module.validate_can_update_experiment_artifact_proxy()
        assert auth_module.validate_can_delete_experiment_artifact_proxy()

    _set_workspace_permission(store, username, READ.name)

    with auth_module.app.test_request_context(
        "/ajax-api/2.0/mlflow-artifacts/artifacts/1/path",
        method="GET",
    ):
        request.view_args = {"artifact_path": "1/path"}
        assert auth_module.validate_can_read_experiment_artifact_proxy()
        assert not auth_module.validate_can_update_experiment_artifact_proxy()
        assert not auth_module.validate_can_delete_experiment_artifact_proxy()


def test_experiment_artifact_proxy_without_experiment_id_uses_workspace_permissions(
    workspace_permission_setup,
):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, READ.name)

    with auth_module.app.test_request_context(
        "/ajax-api/2.0/mlflow-artifacts/artifacts/uploads/path",
        method="GET",
    ):
        request.view_args = {"artifact_path": "uploads/path"}
        assert auth_module.validate_can_read_experiment_artifact_proxy()
        assert not auth_module.validate_can_update_experiment_artifact_proxy()


def test_experiment_artifact_proxy_without_experiment_id_denied_without_workspace_permission(
    workspace_permission_setup,
):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)

    with auth_module.app.test_request_context(
        "/ajax-api/2.0/mlflow-artifacts/artifacts/uploads/path",
        method="GET",
    ):
        request.view_args = {"artifact_path": "uploads/path"}
        assert not auth_module.validate_can_read_experiment_artifact_proxy()


def test_filter_experiment_ids_respects_workspace_permissions(
    workspace_permission_setup, monkeypatch
):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)

    experiment_ids = ["exp-1", "exp-2"]
    assert auth_module.filter_experiment_ids(experiment_ids) == experiment_ids

    _set_workspace_permission(store, username, NO_PERMISSIONS.name)
    assert auth_module.filter_experiment_ids(experiment_ids) == []


def test_run_validators_allow_manage_permission(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, MANAGE.name)

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/runs/get", method="GET", query_string={"run_id": "run-1"}
    ):
        assert auth_module.validate_can_read_run()
        assert auth_module.validate_can_update_run()
        assert auth_module.validate_can_delete_run()
        assert auth_module.validate_can_manage_run()


def test_run_validators_read_permission_blocks_writes(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, READ.name)

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/runs/get", method="GET", query_string={"run_id": "run-1"}
    ):
        assert auth_module.validate_can_read_run()
        assert not auth_module.validate_can_update_run()
        assert not auth_module.validate_can_delete_run()
        assert not auth_module.validate_can_manage_run()


def test_logged_model_validators_respect_permissions(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]

    _set_workspace_permission(store, username, MANAGE.name)
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/logged-models/get",
        method="GET",
        query_string={"model_id": "model-1"},
    ):
        assert auth_module.validate_can_read_logged_model()
        assert auth_module.validate_can_update_logged_model()
        assert auth_module.validate_can_delete_logged_model()
        assert auth_module.validate_can_manage_logged_model()

    _set_workspace_permission(store, username, READ.name)
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/logged-models/get",
        method="GET",
        query_string={"model_id": "model-1"},
    ):
        assert auth_module.validate_can_read_logged_model()
        assert not auth_module.validate_can_update_logged_model()
        assert not auth_module.validate_can_delete_logged_model()
        assert not auth_module.validate_can_manage_logged_model()


def test_scorer_validators_use_workspace_permissions(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, MANAGE.name)

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/scorers/get",
        method="GET",
        query_string={"experiment_id": "exp-1", "name": "score-1"},
    ):
        assert auth_module.validate_can_read_scorer()
        assert auth_module.validate_can_update_scorer()
        assert auth_module.validate_can_delete_scorer()
        assert auth_module.validate_can_manage_scorer()

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/scorers/permissions/create",
        method="POST",
        json={
            "experiment_id": "exp-1",
            "scorer_name": "score-1",
            "username": "bob",
            "permission": "READ",
        },
    ):
        assert auth_module.validate_can_manage_scorer_permission()


def test_scorer_validators_read_permission_blocks_writes(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, READ.name)

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/scorers/get",
        method="GET",
        query_string={"experiment_id": "exp-1", "name": "score-1"},
    ):
        assert auth_module.validate_can_read_scorer()
        assert not auth_module.validate_can_update_scorer()
        assert not auth_module.validate_can_delete_scorer()
        assert not auth_module.validate_can_manage_scorer()

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/scorers/permissions/create",
        method="POST",
        json={
            "experiment_id": "exp-1",
            "scorer_name": "score-1",
            "username": "bob",
            "permission": "READ",
        },
    ):
        assert not auth_module.validate_can_manage_scorer_permission()


def test_registered_model_validators_require_manage_for_writes(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]

    with workspace_context.WorkspaceContext("team-a"):
        _set_workspace_permission(store, username, MANAGE.name)
        with auth_module.app.test_request_context(
            "/api/2.0/mlflow/registered-models/get",
            method="GET",
            query_string={"name": "model-xyz"},
        ):
            assert auth_module.validate_can_read_registered_model()
            assert auth_module.validate_can_update_registered_model()
            assert auth_module.validate_can_delete_registered_model()
            assert auth_module.validate_can_manage_registered_model()
        perm = auth_module._workspace_permission(
            auth_module.authenticate_request().username, "team-a"
        )
        assert perm is not None
        assert perm.can_manage
        assert workspace_context.get_request_workspace() == "team-a"
        assert auth_module.validate_can_create_registered_model()

        _set_workspace_permission(store, username, READ.name)
        with auth_module.app.test_request_context(
            "/api/2.0/mlflow/registered-models/get",
            method="GET",
            query_string={"name": "model-xyz"},
        ):
            assert auth_module.validate_can_read_registered_model()
            assert not auth_module.validate_can_update_registered_model()
            assert not auth_module.validate_can_delete_registered_model()
            assert not auth_module.validate_can_manage_registered_model()
        assert not auth_module.validate_can_create_registered_model()


def test_validate_can_view_workspace_requires_access(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/workspaces/team-a",
        method="GET",
    ):
        request.view_args = {"workspace_name": "team-a"}
        assert auth_module.validate_can_view_workspace()

    store.delete_workspace_permission("team-a", username)

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/workspaces/team-a",
        method="GET",
    ):
        request.view_args = {"workspace_name": "team-a"}
        assert not auth_module.validate_can_view_workspace()


def test_run_artifact_validators_use_workspace_permissions(workspace_permission_setup):
    with auth_module.app.test_request_context(
        GET_ARTIFACT,
        method="GET",
        query_string={"run_id": "run-1"},
    ):
        assert auth_module.validate_can_read_run_artifact()

    with auth_module.app.test_request_context(
        UPLOAD_ARTIFACT,
        method="POST",
        query_string={"run_id": "run-1"},
    ):
        assert auth_module.validate_can_update_run_artifact()


def test_model_version_artifact_validator_uses_workspace_permissions(workspace_permission_setup):
    with auth_module.app.test_request_context(
        GET_MODEL_VERSION_ARTIFACT,
        method="GET",
        query_string={"name": "model-xyz"},
    ):
        assert auth_module.validate_can_read_model_version_artifact()


def test_metric_history_bulk_validator_uses_workspace_permissions(workspace_permission_setup):
    with auth_module.app.test_request_context(
        GET_METRIC_HISTORY_BULK,
        method="GET",
        query_string=[("run_id", "run-1"), ("run_id", "run-2")],
    ):
        assert auth_module.validate_can_read_metric_history_bulk()


def test_metric_history_bulk_interval_validator_uses_workspace_permissions(
    workspace_permission_setup,
):
    with auth_module.app.test_request_context(
        GET_METRIC_HISTORY_BULK_INTERVAL,
        method="GET",
        query_string=[
            ("run_ids", "run-1"),
            ("run_ids", "run-2"),
            ("metric_key", "loss"),
        ],
    ):
        assert auth_module.validate_can_read_metric_history_bulk_interval()


def test_search_datasets_validator_uses_workspace_permissions(workspace_permission_setup):
    with auth_module.app.test_request_context(
        SEARCH_DATASETS,
        method="POST",
        json={"experiment_ids": ["exp-1", "exp-2"]},
    ):
        assert auth_module.validate_can_search_datasets()


def test_create_promptlab_run_validator_uses_workspace_permissions(workspace_permission_setup):
    with auth_module.app.test_request_context(
        CREATE_PROMPTLAB_RUN,
        method="POST",
        json={"experiment_id": "exp-2"},
    ):
        assert auth_module.validate_can_create_promptlab_run()


def test_trace_artifact_validator_uses_workspace_permissions(workspace_permission_setup):
    with auth_module.app.test_request_context(
        GET_TRACE_ARTIFACT,
        method="GET",
        query_string={"request_id": "trace-1"},
    ):
        assert auth_module.validate_can_read_trace_artifact()


def test_experiment_artifact_proxy_without_workspaces_falls_back_to_default(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")
    monkeypatch.setattr(
        auth_module,
        "auth_config",
        auth_module.auth_config._replace(default_permission=READ.name),
        raising=False,
    )
    monkeypatch.setattr(
        auth_module,
        "authenticate_request",
        lambda: SimpleNamespace(username="carol"),
    )

    with auth_module.app.test_request_context(
        "/ajax-api/2.0/mlflow-artifacts/artifacts/uploads/path",
        method="GET",
    ):
        request.view_args = {"artifact_path": "uploads/path"}
        assert auth_module.validate_can_read_experiment_artifact_proxy()


def test_run_artifact_validators_denied_without_workspace_permission(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    store.set_workspace_permission("team-a", username, NO_PERMISSIONS.name)

    with auth_module.app.test_request_context(
        GET_ARTIFACT,
        method="GET",
        query_string={"run_id": "run-1"},
    ):
        assert not auth_module.validate_can_read_run_artifact()

    with auth_module.app.test_request_context(
        UPLOAD_ARTIFACT,
        method="POST",
        query_string={"run_id": "run-1"},
    ):
        assert not auth_module.validate_can_update_run_artifact()


def test_model_version_artifact_validator_denied_without_workspace_permission(
    workspace_permission_setup,
):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    store.set_workspace_permission("team-a", username, NO_PERMISSIONS.name)

    with auth_module.app.test_request_context(
        GET_MODEL_VERSION_ARTIFACT,
        method="GET",
        query_string={"name": "model-xyz"},
    ):
        assert not auth_module.validate_can_read_model_version_artifact()


def test_metric_history_bulk_validator_denied_without_workspace_permission(
    workspace_permission_setup,
):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    store.set_workspace_permission("team-a", username, NO_PERMISSIONS.name)

    with auth_module.app.test_request_context(
        GET_METRIC_HISTORY_BULK,
        method="GET",
        query_string=[("run_id", "run-1"), ("run_id", "run-2")],
    ):
        assert not auth_module.validate_can_read_metric_history_bulk()


def test_metric_history_bulk_interval_validator_denied_without_workspace_permission(
    workspace_permission_setup,
):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    store.set_workspace_permission("team-a", username, NO_PERMISSIONS.name)

    with auth_module.app.test_request_context(
        GET_METRIC_HISTORY_BULK_INTERVAL,
        method="GET",
        query_string=[
            ("run_ids", "run-1"),
            ("run_ids", "run-2"),
            ("metric_key", "loss"),
        ],
    ):
        assert not auth_module.validate_can_read_metric_history_bulk_interval()


def test_search_datasets_validator_denied_without_workspace_permission(
    workspace_permission_setup,
):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    store.set_workspace_permission("team-a", username, NO_PERMISSIONS.name)

    with auth_module.app.test_request_context(
        SEARCH_DATASETS,
        method="POST",
        json={"experiment_ids": ["exp-1", "exp-2"]},
    ):
        assert not auth_module.validate_can_search_datasets()


def test_create_promptlab_run_validator_denied_without_workspace_permission(
    workspace_permission_setup,
):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    store.set_workspace_permission("team-a", username, NO_PERMISSIONS.name)

    with auth_module.app.test_request_context(
        CREATE_PROMPTLAB_RUN,
        method="POST",
        json={"experiment_id": "exp-2"},
    ):
        assert not auth_module.validate_can_create_promptlab_run()


def test_trace_artifact_validator_denied_without_workspace_permission(
    workspace_permission_setup,
):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    store.set_workspace_permission("team-a", username, NO_PERMISSIONS.name)

    with auth_module.app.test_request_context(
        GET_TRACE_ARTIFACT,
        method="GET",
        query_string={"request_id": "trace-1"},
    ):
        assert not auth_module.validate_can_read_trace_artifact()


def test_cross_workspace_access_denied(workspace_permission_setup, monkeypatch):
    tracking_store = _TrackingStore(
        experiment_workspaces={"exp-other-ws": "team-b"},
        run_experiments={"run-other-ws": "exp-other-ws"},
        trace_experiments={},
    )
    monkeypatch.setattr(auth_module, "_get_tracking_store", lambda: tracking_store)

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/experiments/get",
        method="GET",
        query_string={"experiment_id": "exp-other-ws"},
    ):
        assert not auth_module.validate_can_read_experiment()
        assert not auth_module.validate_can_update_experiment()
        assert not auth_module.validate_can_delete_experiment()

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/runs/get",
        method="GET",
        query_string={"run_id": "run-other-ws"},
    ):
        assert not auth_module.validate_can_read_run()
        assert not auth_module.validate_can_update_run()


def test_cross_workspace_registered_model_access_denied(workspace_permission_setup, monkeypatch):
    registry_store = _RegistryStore({"model-other-ws": "team-b"})
    monkeypatch.setattr(auth_module, "_get_model_registry_store", lambda: registry_store)

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/registered-models/get",
        method="GET",
        query_string={"name": "model-other-ws"},
    ):
        assert not auth_module.validate_can_read_registered_model()
        assert not auth_module.validate_can_update_registered_model()
        assert not auth_module.validate_can_delete_registered_model()


def test_explicit_experiment_permission_overrides_workspace(
    workspace_permission_setup,
):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]

    store.set_workspace_permission("team-a", username, NO_PERMISSIONS.name)
    store.create_experiment_permission("exp-1", username, READ.name)

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/experiments/get",
        method="GET",
        query_string={"experiment_id": "exp-1"},
    ):
        assert auth_module.validate_can_read_experiment()
        assert not auth_module.validate_can_update_experiment()

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/experiments/get",
        method="GET",
        query_string={"experiment_id": "exp-2"},
    ):
        assert not auth_module.validate_can_read_experiment()


def test_cross_workspace_gateway_secret_access_denied(workspace_permission_setup, monkeypatch):
    tracking_store = _TrackingStore(
        experiment_workspaces={"exp-1": "team-a"},
        run_experiments={},
        trace_experiments={},
        gateway_secret_workspaces={"secret-other-ws": "team-b"},
        engine=MagicMock(),
    )
    tracking_store.ManagedSessionMaker = tracking_store._create_mock_session_maker()
    monkeypatch.setattr(auth_module, "_get_tracking_store", lambda: tracking_store)

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/gateway/secrets/get",
        method="GET",
        query_string={"secret_id": "secret-other-ws"},
    ):
        assert not auth_module.validate_can_read_gateway_secret()
        assert not auth_module.validate_can_update_gateway_secret()
        assert not auth_module.validate_can_delete_gateway_secret()


def test_cross_workspace_gateway_endpoint_access_denied(workspace_permission_setup, monkeypatch):
    tracking_store = _TrackingStore(
        experiment_workspaces={"exp-1": "team-a"},
        run_experiments={},
        trace_experiments={},
        gateway_endpoint_workspaces={"endpoint-other-ws": "team-b"},
        engine=MagicMock(),
    )
    tracking_store.ManagedSessionMaker = tracking_store._create_mock_session_maker()
    monkeypatch.setattr(auth_module, "_get_tracking_store", lambda: tracking_store)

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/gateway/endpoints/get",
        method="GET",
        query_string={"endpoint_id": "endpoint-other-ws"},
    ):
        assert not auth_module.validate_can_read_gateway_endpoint()
        assert not auth_module.validate_can_update_gateway_endpoint()
        assert not auth_module.validate_can_delete_gateway_endpoint()


def test_cross_workspace_gateway_model_definition_access_denied(
    workspace_permission_setup, monkeypatch
):
    tracking_store = _TrackingStore(
        experiment_workspaces={"exp-1": "team-a"},
        run_experiments={},
        trace_experiments={},
        gateway_model_def_workspaces={"model-def-other-ws": "team-b"},
        engine=MagicMock(),
    )
    tracking_store.ManagedSessionMaker = tracking_store._create_mock_session_maker()
    monkeypatch.setattr(auth_module, "_get_tracking_store", lambda: tracking_store)

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/gateway/model-definitions/get",
        method="GET",
        query_string={"model_definition_id": "model-def-other-ws"},
    ):
        assert not auth_module.validate_can_read_gateway_model_definition()
        assert not auth_module.validate_can_update_gateway_model_definition()
        assert not auth_module.validate_can_delete_gateway_model_definition()


def test_workspace_permission_required_for_gateway_creation(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]

    # Remove workspace permission
    store.set_workspace_permission("team-a", username, NO_PERMISSIONS.name)

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/gateway/endpoints/create",
        method="POST",
        json={"name": "test-endpoint", "model_configs": []},
    ):
        assert not auth_module.validate_can_create_gateway_endpoint()

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/gateway/model-definitions/create",
        method="POST",
        json={
            "name": "test-model",
            "secret_id": "secret-1",
            "provider": "openai",
            "model_name": "gpt-4",
        },
    ):
        assert not auth_module.validate_can_create_gateway_model_definition()

    # Restore workspace permission
    store.set_workspace_permission("team-a", username, MANAGE.name)

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/gateway/endpoints/create",
        method="POST",
        json={"name": "test-endpoint", "model_configs": []},
    ):
        assert auth_module.validate_can_create_gateway_endpoint()
