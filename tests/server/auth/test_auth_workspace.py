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
from mlflow.server.auth.permissions import MANAGE, NO_PERMISSIONS, READ, USE
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
    mock_delete_workspace_perms = Mock()
    mock_delete_roles = Mock()

    monkeypatch.setattr(
        auth_module.store,
        "delete_workspace_permissions_for_workspace",
        mock_delete_workspace_perms,
        raising=True,
    )
    monkeypatch.setattr(
        auth_module.store,
        "delete_roles_for_workspace",
        mock_delete_roles,
        raising=True,
    )

    workspace_name = f"team-{random_str(10)}"
    with auth_module.app.test_request_context(
        f"/api/3.0/mlflow/workspaces/{workspace_name}", method="DELETE"
    ):
        request.view_args = {"workspace_name": workspace_name}
        response = Response(status=204)
        auth_module._after_request(response)

    mock_delete_workspace_perms.assert_called_once_with(workspace_name)
    mock_delete_roles.assert_called_once_with(workspace_name)


def _create_workspace_response(workspace_name: str) -> Response:
    payload = {"workspace": {"name": workspace_name}}
    return Response(json.dumps(payload), status=201, content_type="application/json")


def test_seed_default_workspace_roles_happy_path(monkeypatch):
    monkeypatch.setenv("MLFLOW_RBAC_SEED_DEFAULT_ROLES", "true")
    workspace_name = f"team-{random_str(10)}"

    created_roles: list[dict[str, object]] = []
    added_perms: list[dict[str, object]] = []

    def fake_create_role(name, workspace, description=None):
        role_id = len(created_roles) + 1
        created_roles.append({
            "id": role_id,
            "name": name,
            "workspace": workspace,
            "description": description,
        })
        return SimpleNamespace(id=role_id, name=name, workspace=workspace)

    def fake_add_role_permission(role_id, resource_type, resource_pattern, permission):
        added_perms.append({
            "role_id": role_id,
            "resource_type": resource_type,
            "resource_pattern": resource_pattern,
            "permission": permission,
        })
        return SimpleNamespace(id=role_id)

    monkeypatch.setattr(auth_module.store, "create_role", fake_create_role, raising=True)
    monkeypatch.setattr(
        auth_module.store, "add_role_permission", fake_add_role_permission, raising=True
    )

    with auth_module.app.test_request_context("/api/3.0/mlflow/workspaces", method="POST"):
        auth_module._seed_default_workspace_roles(_create_workspace_response(workspace_name))

    names = [r["name"] for r in created_roles]
    assert names == ["admin", "user"]
    assert all(r["workspace"] == workspace_name for r in created_roles)

    # Both roles use resource_type='workspace' (the only supported workspace-wide
    # resource_type in VALID_RESOURCE_TYPES). The permission level differentiates them.
    assert [(p["resource_type"], p["permission"]) for p in added_perms] == [
        ("workspace", MANAGE.name),
        ("workspace", USE.name),
    ]
    assert all(p["resource_pattern"] == "*" for p in added_perms)


def test_seed_default_workspace_roles_disabled_skips_seeding(monkeypatch):
    # With seeding off, no roles are created. ``CreateWorkspace`` is gated to
    # super-admins so the creator already bypasses RBAC — there is nothing to
    # fall back to.
    monkeypatch.setenv("MLFLOW_RBAC_SEED_DEFAULT_ROLES", "false")
    workspace_name = f"team-{random_str(10)}"

    mock_create_role = Mock()
    mock_add_role_permission = Mock()
    mock_assign_role_to_user = Mock()
    mock_set_workspace_permission = Mock()

    monkeypatch.setattr(auth_module.store, "create_role", mock_create_role, raising=True)
    monkeypatch.setattr(
        auth_module.store, "add_role_permission", mock_add_role_permission, raising=True
    )
    monkeypatch.setattr(
        auth_module.store, "assign_role_to_user", mock_assign_role_to_user, raising=True
    )
    monkeypatch.setattr(
        auth_module.store,
        "set_workspace_permission",
        mock_set_workspace_permission,
        raising=True,
    )

    with auth_module.app.test_request_context("/api/3.0/mlflow/workspaces", method="POST"):
        auth_module._seed_default_workspace_roles(_create_workspace_response(workspace_name))

    mock_create_role.assert_not_called()
    mock_add_role_permission.assert_not_called()
    mock_assign_role_to_user.assert_not_called()
    mock_set_workspace_permission.assert_not_called()


def test_seed_default_workspace_roles_admin_creation_fails_still_seeds_others(monkeypatch):
    # Best-effort seeding: a failure on one role doesn't block the rest.
    monkeypatch.setenv("MLFLOW_RBAC_SEED_DEFAULT_ROLES", "true")
    workspace_name = f"team-{random_str(10)}"

    def fake_create_role(name, workspace, description=None):
        if name == "admin":
            raise MlflowException("simulated admin role failure")
        return SimpleNamespace(id=10, name=name, workspace=workspace)

    mock_add_role_permission = Mock()

    monkeypatch.setattr(auth_module.store, "create_role", fake_create_role, raising=True)
    monkeypatch.setattr(
        auth_module.store, "add_role_permission", mock_add_role_permission, raising=True
    )

    with auth_module.app.test_request_context("/api/3.0/mlflow/workspaces", method="POST"):
        auth_module._seed_default_workspace_roles(_create_workspace_response(workspace_name))

    # The remaining ``user`` role still got created (best-effort seeding).
    assert mock_add_role_permission.call_count == 1


def test_seed_default_workspace_roles_permission_add_fails_rolls_back_role(monkeypatch):
    # create_role succeeds but add_role_permission raises — the orphan role must be
    # deleted so the workspace doesn't end up with a named role that grants nothing.
    monkeypatch.setenv("MLFLOW_RBAC_SEED_DEFAULT_ROLES", "true")
    workspace_name = f"team-{random_str(10)}"

    def fake_create_role(name, workspace, description=None):
        return SimpleNamespace(
            id={"admin": 1, "user": 2}[name],
            name=name,
            workspace=workspace,
        )

    def fake_add_role_permission(role_id, resource_type, resource_pattern, permission):
        if role_id == 1:  # admin
            raise MlflowException("simulated add_role_permission failure")
        return SimpleNamespace(id=role_id)

    mock_delete_role = Mock()

    monkeypatch.setattr(auth_module.store, "create_role", fake_create_role, raising=True)
    monkeypatch.setattr(
        auth_module.store, "add_role_permission", fake_add_role_permission, raising=True
    )
    monkeypatch.setattr(auth_module.store, "delete_role", mock_delete_role, raising=True)

    with auth_module.app.test_request_context("/api/3.0/mlflow/workspaces", method="POST"):
        auth_module._seed_default_workspace_roles(_create_workspace_response(workspace_name))

    # Orphan admin role (id=1) was rolled back.
    mock_delete_role.assert_called_once_with(1)


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
        # For test simplicity we treat ``name`` as a synonym for ``endpoint_id``
        # (our fixture data uses the same string for both). This mirrors how the
        # real store resolves a name → id lookup before returning the endpoint.
        if lookup_id := (endpoint_id or name):
            if lookup_id not in self._gateway_endpoint_workspaces:
                raise MlflowException(
                    f"GatewayEndpoint not found ({lookup_id})",
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            # Add workspace attribute so _get_resource_workspace can extract it
            return SimpleNamespace(
                endpoint_id=lookup_id, workspace=self._gateway_endpoint_workspaces[lookup_id]
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

        def get_role_workspace_permission(self, workspace_name, username):
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
        json.dumps({
            "workspaces": [
                {"name": default_workspace},
                {"name": "other-workspace"},
            ]
        }),
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


def test_list_workspaces_filters_to_role_assigned_workspaces(tmp_path, monkeypatch):
    # End-to-end guard for the list_accessible_workspace_names fix: alice has NO
    # legacy workspace_permissions rows — her only workspace membership is via a
    # role assignment in ws-alpha. The ListWorkspaces filter must treat that role
    # assignment as workspace visibility and surface ws-alpha but not ws-beta.
    # Before the fix, the legacy-only query returned an empty set and alice saw
    # no workspaces in the UI.
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    monkeypatch.setattr(
        auth_module,
        "auth_config",
        auth_module.auth_config._replace(grant_default_workspace_access=False),
        raising=False,
    )

    db_uri = f"sqlite:///{tmp_path / 'auth-store.db'}"
    auth_store = SqlAlchemyStore()
    auth_store.init_db(db_uri)
    monkeypatch.setattr(auth_module, "store", auth_store, raising=False)

    alice = auth_store.create_user("alice", "supersecurepassword", is_admin=False)
    role = auth_store.create_role(name="viewer", workspace="ws-alpha")
    auth_store.add_role_permission(role.id, "experiment", "*", READ.name)
    auth_store.assign_role_to_user(alice.id, role.id)

    monkeypatch.setattr(
        auth_module, "authenticate_request", lambda: SimpleNamespace(username="alice")
    )

    response = Response(
        json.dumps({"workspaces": [{"name": "ws-alpha"}, {"name": "ws-beta"}]}),
        mimetype="application/json",
    )

    auth_module.filter_list_workspaces(response)
    payload = json.loads(response.get_data(as_text=True))
    assert [ws["name"] for ws in payload["workspaces"]] == ["ws-alpha"]

    auth_store.engine.dispose()


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


def test_experiment_validators_allow_role_based_workspace_manage(workspace_permission_setup):
    # Grant MANAGE on the workspace via a role (with no legacy
    # ``workspace_permissions`` row). Pre-fix this returned 403 — the
    # workspace-level permission check only consulted the legacy table.
    # ``_workspace_permission`` now max-merges role-based grants.
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]

    # Drop the legacy grant the fixture installs so we exercise the role path
    # in isolation.
    store.delete_workspace_permission("team-a", username)
    assert store.get_workspace_permission("team-a", username) is None

    role = store.create_role(name="ws-admin", workspace="team-a")
    store.add_role_permission(role.id, "workspace", "*", MANAGE.name)
    user = store.get_user(username)
    store.assign_role_to_user(user.id, role.id)

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/experiments/get", method="GET", query_string={"experiment_id": "exp-1"}
    ):
        assert auth_module.validate_can_read_experiment()
        assert auth_module.validate_can_update_experiment()
        assert auth_module.validate_can_delete_experiment()
        assert auth_module.validate_can_manage_experiment()

    with workspace_context.WorkspaceContext("team-a"):
        assert auth_module.validate_can_create_experiment()


def test_workspace_permission_max_merges_legacy_and_role(workspace_permission_setup):
    # Operators mid-migration may have BOTH a legacy grant and a role grant.
    # The effective permission must be the higher of the two — neither side
    # should silently downgrade the other.
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]

    # Legacy READ + role MANAGE → effective MANAGE.
    store.delete_workspace_permission("team-a", username)
    store.set_workspace_permission("team-a", username, READ.name)

    role = store.create_role(name="ws-admin", workspace="team-a")
    store.add_role_permission(role.id, "workspace", "*", MANAGE.name)
    user = store.get_user(username)
    store.assign_role_to_user(user.id, role.id)

    with workspace_context.WorkspaceContext("team-a"):
        assert auth_module.validate_can_create_experiment()


def test_use_workspace_permission_allows_create_but_blocks_others_writes(
    workspace_permission_setup,
):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)

    with workspace_context.WorkspaceContext("team-a"):
        assert auth_module.validate_can_create_experiment()
        assert auth_module.validate_can_create_registered_model()

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/experiments/get", method="GET", query_string={"experiment_id": "exp-1"}
    ):
        # USE preserves read; non-owned resources stay read-only.
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

    with (
        workspace_context.WorkspaceContext("team-a"),
        auth_module.app.test_request_context(
            "/api/2.0/mlflow/registered-models/get",
            method="GET",
            query_string={"name": "model-xyz"},
        ),
    ):
        # Same property for registered models.
        assert auth_module.validate_can_read_registered_model()
        assert not auth_module.validate_can_update_registered_model()
        assert not auth_module.validate_can_delete_registered_model()
        assert not auth_module.validate_can_manage_registered_model()


def test_read_workspace_permission_blocks_create(workspace_permission_setup):
    # READ is purely "can see" — no create. The contributor pattern needs USE.
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, READ.name)

    with workspace_context.WorkspaceContext("team-a"):
        assert not auth_module.validate_can_create_experiment()
        assert not auth_module.validate_can_create_registered_model()


def test_no_permissions_blocks_create(workspace_permission_setup):
    # Without any access to the workspace, create is denied.
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)

    with workspace_context.WorkspaceContext("team-a"):
        assert not auth_module.validate_can_create_experiment()
        assert not auth_module.validate_can_create_registered_model()


def test_role_grant_workspace_use_allows_create(workspace_permission_setup, monkeypatch):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    user_id = store.get_user(username).id
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    # Drop the legacy grant the fixture installs so we exercise the role-only
    # path in isolation.
    store.delete_workspace_permission("team-a", username)

    role = store.create_role(name="ws-contributor", workspace="team-a")
    store.add_role_permission(role.id, "workspace", "*", USE.name)
    store.assign_role_to_user(user_id, role.id)

    with workspace_context.WorkspaceContext("team-a"):
        assert auth_module.validate_can_create_experiment()
        assert auth_module.validate_can_create_registered_model()


def test_role_grant_resource_type_use_does_not_allow_create(
    workspace_permission_setup, monkeypatch
):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    user_id = store.get_user(username).id
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    store.delete_workspace_permission("team-a", username)

    role = store.create_role(name="exp-user", workspace="team-a")
    store.add_role_permission(role.id, "experiment", "*", USE.name)
    store.assign_role_to_user(user_id, role.id)

    with workspace_context.WorkspaceContext("team-a"):
        assert not auth_module.validate_can_create_experiment()
        assert not auth_module.validate_can_create_registered_model()


def test_role_grant_workspace_read_does_not_allow_create(workspace_permission_setup, monkeypatch):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    user_id = store.get_user(username).id
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    store.delete_workspace_permission("team-a", username)

    role = store.create_role(name="ws-viewer", workspace="team-a")
    store.add_role_permission(role.id, "workspace", "*", READ.name)
    store.assign_role_to_user(user_id, role.id)

    with workspace_context.WorkspaceContext("team-a"):
        assert not auth_module.validate_can_create_experiment()
        assert not auth_module.validate_can_create_registered_model()


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


def test_filter_experiment_ids_role_wildcard_grant(workspace_permission_setup, monkeypatch):
    # Role granting experiment(*) in the active workspace should include all experiments.
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    user_id = store.get_user(username).id
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    # Start from NO_PERMISSIONS: workspace fallback would exclude everything.
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)

    role = store.create_role(name="exp-reader", workspace="team-a")
    store.add_role_permission(role.id, "experiment", "*", "READ")
    store.assign_role_to_user(user_id, role.id)

    token = workspace_context.set_server_request_workspace("team-a")
    try:
        assert auth_module.filter_experiment_ids(["exp-1", "exp-2"]) == ["exp-1", "exp-2"]
    finally:
        workspace_context._WORKSPACE.reset(token)


def test_filter_experiment_ids_role_specific_grant(workspace_permission_setup, monkeypatch):
    # Role granting a specific experiment id should include that id only (plus direct grants).
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    user_id = store.get_user(username).id
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)

    role = store.create_role(name="exp-1-reader", workspace="team-a")
    store.add_role_permission(role.id, "experiment", "exp-1", "READ")
    store.assign_role_to_user(user_id, role.id)

    token = workspace_context.set_server_request_workspace("team-a")
    try:
        # Only exp-1 (via role); exp-2 is filtered out.
        assert auth_module.filter_experiment_ids(["exp-1", "exp-2"]) == ["exp-1"]
    finally:
        workspace_context._WORKSPACE.reset(token)


def test_filter_experiment_ids_workspace_scope_role(workspace_permission_setup, monkeypatch):
    # Role with (resource_type='workspace', '*', READ) should grant access to all experiments.
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    user_id = store.get_user(username).id
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)

    role = store.create_role(name="ws-reader", workspace="team-a")
    store.add_role_permission(role.id, "workspace", "*", "READ")
    store.assign_role_to_user(user_id, role.id)

    token = workspace_context.set_server_request_workspace("team-a")
    try:
        assert auth_module.filter_experiment_ids(["exp-1", "exp-2"]) == ["exp-1", "exp-2"]
    finally:
        workspace_context._WORKSPACE.reset(token)


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
        # READ is purely "can see" — create requires USE or higher.
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


def test_prompt_optimization_job_validators_use_workspace_permissions(
    workspace_permission_setup, monkeypatch
):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]

    # Mock get_job to return a job associated with exp-1 (in team-a)
    mock_job = SimpleNamespace(params='{"experiment_id": "exp-1"}')
    monkeypatch.setattr(auth_module, "get_job", lambda job_id: mock_job)

    _set_workspace_permission(store, username, MANAGE.name)

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/prompt-optimization/jobs/get",
        method="GET",
        query_string={"job_id": "job-1"},
    ):
        assert auth_module.validate_can_read_prompt_optimization_job()
        assert auth_module.validate_can_update_prompt_optimization_job()
        assert auth_module.validate_can_delete_prompt_optimization_job()


def test_prompt_optimization_job_validators_read_permission_blocks_writes(
    workspace_permission_setup, monkeypatch
):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]

    # Mock get_job to return a job associated with exp-1 (in team-a)
    mock_job = SimpleNamespace(params='{"experiment_id": "exp-1"}')
    monkeypatch.setattr(auth_module, "get_job", lambda job_id: mock_job)

    _set_workspace_permission(store, username, READ.name)

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/prompt-optimization/jobs/get",
        method="GET",
        query_string={"job_id": "job-1"},
    ):
        assert auth_module.validate_can_read_prompt_optimization_job()
        assert not auth_module.validate_can_update_prompt_optimization_job()
        assert not auth_module.validate_can_delete_prompt_optimization_job()


def test_prompt_optimization_job_validators_denied_without_workspace_permission(
    workspace_permission_setup, monkeypatch
):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]

    # Mock get_job to return a job associated with exp-1 (in team-a)
    mock_job = SimpleNamespace(params='{"experiment_id": "exp-1"}')
    monkeypatch.setattr(auth_module, "get_job", lambda job_id: mock_job)

    _set_workspace_permission(store, username, NO_PERMISSIONS.name)

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/prompt-optimization/jobs/get",
        method="GET",
        query_string={"job_id": "job-1"},
    ):
        assert not auth_module.validate_can_read_prompt_optimization_job()
        assert not auth_module.validate_can_update_prompt_optimization_job()
        assert not auth_module.validate_can_delete_prompt_optimization_job()


def test_graphql_permission_functions_use_workspace_permissions(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]

    _set_workspace_permission(store, username, MANAGE.name)

    # Test experiment permission
    assert auth_module._graphql_can_read_experiment("exp-1", username)

    # Test run permission (inherits from experiment)
    assert auth_module._graphql_can_read_run("run-1", username)

    # Test registered model permission
    assert auth_module._graphql_can_read_model("model-xyz", username)


def test_graphql_permission_functions_denied_without_workspace_permission(
    workspace_permission_setup,
):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]

    _set_workspace_permission(store, username, NO_PERMISSIONS.name)

    # Test experiment permission denied
    assert not auth_module._graphql_can_read_experiment("exp-1", username)

    # Test run permission denied (inherits from experiment)
    assert not auth_module._graphql_can_read_run("run-1", username)

    # Test registered model permission denied
    assert not auth_module._graphql_can_read_model("model-xyz", username)


def test_cross_workspace_graphql_access_denied(workspace_permission_setup, monkeypatch):
    # User has MANAGE in team-a but tries to access resources in team-b
    tracking_store = _TrackingStore(
        experiment_workspaces={"exp-other-ws": "team-b"},
        run_experiments={"run-other-ws": "exp-other-ws"},
        trace_experiments={},
    )
    monkeypatch.setattr(auth_module, "_get_tracking_store", lambda: tracking_store)

    registry_store = _RegistryStore({"model-other-ws": "team-b"})
    monkeypatch.setattr(auth_module, "_get_model_registry_store", lambda: registry_store)

    username = workspace_permission_setup["username"]

    # Should be denied access to resources in team-b
    assert not auth_module._graphql_can_read_experiment("exp-other-ws", username)
    assert not auth_module._graphql_can_read_run("run-other-ws", username)
    assert not auth_module._graphql_can_read_model("model-other-ws", username)


# =============================================================================
# Role-based permission coverage for gateway resources
# =============================================================================
#
# The fixture grants workspace MANAGE by default. These tests first strip that
# grant (set to NO_PERMISSIONS) so the only path to a positive permission is
# the role assignment being exercised. That isolates the role-based resolver
# from the legacy workspace_permissions fallback.


def _assign_role_with_permission(
    store: SqlAlchemyStore, username: str, workspace: str, resource_type: str, permission: str
) -> None:
    """Create a role in ``workspace`` with a wildcard grant of ``permission`` on
    ``resource_type``, and assign ``username`` to it.

    Using ``random_str`` keeps the role names unique so multiple calls within a
    single test don't collide on the (workspace, name) unique constraint.
    """
    role = store.create_role(name=random_str(), workspace=workspace)
    store.add_role_permission(role.id, resource_type, "*", permission)
    user = store.get_user(username)
    store.assign_role_to_user(user.id, role.id)


# ---- Gateway endpoint: role-based permission levels ----


@pytest.mark.parametrize(
    ("granted", "expected_read", "expected_delete", "expected_manage"),
    [
        ("READ", True, False, False),
        ("USE", True, False, False),
        ("EDIT", True, False, False),
        ("MANAGE", True, True, True),
    ],
)
def test_role_grant_on_gateway_endpoint_gates_validator_capabilities(
    workspace_permission_setup, granted, expected_read, expected_delete, expected_manage
):
    """A role grant at permission level ``granted`` exposes exactly the
    capabilities that level implies on the endpoint validators — no more, no
    less. Catches regressions where a validator starts accepting a weaker
    permission than it should (or refuses a stronger one).
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]

    # Strip the default workspace MANAGE so the only positive grant is the role.
    store.set_workspace_permission("team-a", username, NO_PERMISSIONS.name)

    _assign_role_with_permission(store, username, "team-a", "gateway_endpoint", granted)

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/gateway/endpoints/get",
        method="GET",
        query_string={"endpoint_id": "endpoint-1"},
    ):
        assert auth_module.validate_can_read_gateway_endpoint() is expected_read
        assert auth_module.validate_can_delete_gateway_endpoint() is expected_delete
        assert auth_module.validate_can_manage_gateway_endpoint() is expected_manage


def test_role_grant_read_on_gateway_endpoint_does_not_permit_use(
    workspace_permission_setup,
):
    """Regression guard specific to the bug class the user called out:
    a user with only READ on a gateway endpoint should not be able to *invoke*
    it. USE is a stricter capability than READ and has its own validator.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    store.set_workspace_permission("team-a", username, NO_PERMISSIONS.name)

    _assign_role_with_permission(store, username, "team-a", "gateway_endpoint", "READ")

    # _validate_gateway_use_permission looks up the endpoint by name, resolves
    # the endpoint id, then checks ``can_use`` via the permission resolver.
    with auth_module.app.test_request_context("/"):
        assert auth_module._validate_gateway_use_permission("endpoint-1", username) is False


def test_role_grant_use_on_gateway_endpoint_permits_use(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    store.set_workspace_permission("team-a", username, NO_PERMISSIONS.name)

    _assign_role_with_permission(store, username, "team-a", "gateway_endpoint", "USE")

    with auth_module.app.test_request_context("/"):
        assert auth_module._validate_gateway_use_permission("endpoint-1", username) is True


@pytest.mark.parametrize(
    ("granted", "expected_can_use"),
    [
        ("READ", False),  # READ does not imply USE.
        ("USE", True),
        ("EDIT", True),  # EDIT implies USE.
        ("MANAGE", True),  # MANAGE implies USE.
    ],
)
def test_role_grant_permission_level_determines_use_capability(
    workspace_permission_setup, granted, expected_can_use
):
    """Parametrized matrix for the USE capability specifically. READ should NOT
    let the user invoke; every stronger permission should.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    store.set_workspace_permission("team-a", username, NO_PERMISSIONS.name)

    _assign_role_with_permission(store, username, "team-a", "gateway_endpoint", granted)

    with auth_module.app.test_request_context("/"):
        assert (
            auth_module._validate_gateway_use_permission("endpoint-1", username) is expected_can_use
        )


# ---- Workspace-wide role grants on gateway resources ----


@pytest.mark.parametrize("granted", ["READ", "USE", "EDIT", "MANAGE"])
def test_role_workspace_wide_grant_applies_to_gateway_endpoints(
    workspace_permission_setup, granted
):
    """``(workspace, *, X)`` grants apply to every resource type in the
    workspace — including gateway endpoints. Confirms the workspace-wide
    short-circuit isn't accidentally gated behind resource_type=='experiment'
    or similar, which would silently lock workspace admins out of gateway
    resources.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    store.set_workspace_permission("team-a", username, NO_PERMISSIONS.name)

    _assign_role_with_permission(store, username, "team-a", "workspace", granted)

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/gateway/endpoints/get",
        method="GET",
        query_string={"endpoint_id": "endpoint-1"},
    ):
        # All four levels grant READ.
        assert auth_module.validate_can_read_gateway_endpoint() is True

        # Only MANAGE grants can_delete / can_manage.
        assert auth_module.validate_can_delete_gateway_endpoint() is (granted == "MANAGE")
        assert auth_module.validate_can_manage_gateway_endpoint() is (granted == "MANAGE")


def test_role_workspace_wide_read_does_not_imply_use_on_gateway_endpoint(
    workspace_permission_setup,
):
    """``(workspace, *, READ)`` grants READ on every resource but not USE.
    Users with a workspace-wide viewer role shouldn't be able to invoke
    gateway endpoints.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    store.set_workspace_permission("team-a", username, NO_PERMISSIONS.name)

    _assign_role_with_permission(store, username, "team-a", "workspace", "READ")

    with auth_module.app.test_request_context("/"):
        assert auth_module._validate_gateway_use_permission("endpoint-1", username) is False


@pytest.mark.parametrize("granted", ["USE", "EDIT", "MANAGE"])
def test_role_workspace_wide_non_read_grants_imply_use_on_gateway_endpoint(
    workspace_permission_setup, granted
):
    """``(workspace, *, {USE,EDIT,MANAGE})`` all imply USE → invocation
    allowed.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    store.set_workspace_permission("team-a", username, NO_PERMISSIONS.name)

    _assign_role_with_permission(store, username, "team-a", "workspace", granted)

    with auth_module.app.test_request_context("/"):
        assert auth_module._validate_gateway_use_permission("endpoint-1", username) is True


# ---- Gateway secret and model definition parity ----


@pytest.mark.parametrize(
    ("granted", "expected_read", "expected_delete"),
    [
        ("READ", True, False),
        ("EDIT", True, False),
        ("MANAGE", True, True),
    ],
)
def test_role_grant_on_gateway_secret_gates_validator(
    workspace_permission_setup, granted, expected_read, expected_delete
):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    store.set_workspace_permission("team-a", username, NO_PERMISSIONS.name)

    _assign_role_with_permission(store, username, "team-a", "gateway_secret", granted)

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/gateway/secrets/get",
        method="GET",
        query_string={"secret_id": "secret-1"},
    ):
        assert auth_module.validate_can_read_gateway_secret() is expected_read
        assert auth_module.validate_can_delete_gateway_secret() is expected_delete


@pytest.mark.parametrize(
    ("granted", "expected_read", "expected_delete"),
    [
        ("READ", True, False),
        ("EDIT", True, False),
        ("MANAGE", True, True),
    ],
)
def test_role_grant_on_gateway_model_definition_gates_validator(
    workspace_permission_setup, granted, expected_read, expected_delete
):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    store.set_workspace_permission("team-a", username, NO_PERMISSIONS.name)

    _assign_role_with_permission(store, username, "team-a", "gateway_model_definition", granted)

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/gateway/model-definitions/get",
        method="GET",
        query_string={"model_definition_id": "model-def-1"},
    ):
        assert auth_module.validate_can_read_gateway_model_definition() is expected_read
        assert auth_module.validate_can_delete_gateway_model_definition() is expected_delete


# ---- Cross-workspace isolation for role-based gateway grants ----


def test_role_in_other_workspace_does_not_grant_gateway_endpoint_access(
    workspace_permission_setup,
):
    """A role in team-b with MANAGE on gateway_endpoints must not grant any
    access when resolving an endpoint that belongs to team-a. The resolver
    scopes role permissions to the role's workspace.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    store.set_workspace_permission("team-a", username, NO_PERMISSIONS.name)

    # Role with MANAGE in team-b — should NOT apply to team-a endpoints.
    _assign_role_with_permission(store, username, "team-b", "gateway_endpoint", "MANAGE")

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/gateway/endpoints/get",
        method="GET",
        query_string={"endpoint_id": "endpoint-1"},  # endpoint-1 is in team-a.
    ):
        assert auth_module.validate_can_read_gateway_endpoint() is False
        assert auth_module.validate_can_manage_gateway_endpoint() is False


def test_role_in_other_workspace_does_not_grant_gateway_use(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    store.set_workspace_permission("team-a", username, NO_PERMISSIONS.name)

    _assign_role_with_permission(store, username, "team-b", "gateway_endpoint", "USE")

    with auth_module.app.test_request_context("/"):
        # endpoint-1 is in team-a; role grant is in team-b.
        assert auth_module._validate_gateway_use_permission("endpoint-1", username) is False


# ---- Multi-role union: best grant wins ----


def test_role_union_best_permission_wins_for_gateway_endpoint(workspace_permission_setup):
    """Two roles: one grants READ, the other grants MANAGE. Validator should
    reflect the max (MANAGE).
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    store.set_workspace_permission("team-a", username, NO_PERMISSIONS.name)

    _assign_role_with_permission(store, username, "team-a", "gateway_endpoint", "READ")
    _assign_role_with_permission(store, username, "team-a", "gateway_endpoint", "MANAGE")

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/gateway/endpoints/get",
        method="GET",
        query_string={"endpoint_id": "endpoint-1"},
    ):
        assert auth_module.validate_can_manage_gateway_endpoint() is True


# =============================================================================
# Authorization for role management endpoints (Batch 5)
# =============================================================================
#
# Four validators guard the role endpoints:
#   - validate_can_manage_roles: create/update/delete role, add/remove/update
#     role_permission, assign/unassign role. Super admin OR workspace admin
#     in the resolved workspace.
#   - validate_can_view_roles: get_role, list_role_permissions. Super admin
#     OR any role assignment in the resolved workspace.
#   - validate_can_list_roles: list_roles. Super admin unconditionally; for
#     non-admins the request must scope to a workspace where the caller holds
#     at least one role.
#   - validate_can_view_user_roles: list_user_roles. Super admin, the target
#     themselves, or a workspace admin over any workspace the target is in.
#
# _get_role_workspace_from_request resolves the workspace from role_id,
# role_permission_id, or a literal ``workspace`` param. These tests exercise
# all three shapes and every actor x endpoint combination.


@pytest.fixture
def role_auth_setup(tmp_path, monkeypatch):
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

    auth_store.create_user("super_admin", "supersecurepassword", is_admin=True)
    for name in ("ws_admin_foo", "ws_admin_bar", "ws_member_foo", "outsider"):
        auth_store.create_user(name, "supersecurepassword", is_admin=False)

    admin_role_foo = auth_store.create_role(name="admin-foo", workspace="foo")
    auth_store.add_role_permission(admin_role_foo.id, "workspace", "*", MANAGE.name)
    auth_store.assign_role_to_user(auth_store.get_user("ws_admin_foo").id, admin_role_foo.id)

    admin_role_bar = auth_store.create_role(name="admin-bar", workspace="bar")
    auth_store.add_role_permission(admin_role_bar.id, "workspace", "*", MANAGE.name)
    auth_store.assign_role_to_user(auth_store.get_user("ws_admin_bar").id, admin_role_bar.id)

    member_role_foo = auth_store.create_role(name="member-foo", workspace="foo")
    auth_store.add_role_permission(member_role_foo.id, "experiment", "*", READ.name)
    auth_store.assign_role_to_user(auth_store.get_user("ws_member_foo").id, member_role_foo.id)

    role_foo = auth_store.create_role(name="target-foo", workspace="foo")
    role_bar = auth_store.create_role(name="target-bar", workspace="bar")
    rp_foo = auth_store.add_role_permission(role_foo.id, "experiment", "*", READ.name)
    rp_bar = auth_store.add_role_permission(role_bar.id, "experiment", "*", READ.name)

    def login_as(username: str) -> None:
        monkeypatch.setattr(
            auth_module,
            "authenticate_request",
            lambda: SimpleNamespace(username=username),
        )

    yield {
        "store": auth_store,
        "login_as": login_as,
        "role_foo_id": role_foo.id,
        "role_bar_id": role_bar.id,
        "role_permission_foo_id": rp_foo.id,
        "role_permission_bar_id": rp_bar.id,
    }
    auth_store.engine.dispose()


def _request_context_for_shape(shape, role_auth_setup, workspace):
    match shape:
        case "role_id":
            role_id = (
                role_auth_setup["role_foo_id"]
                if workspace == "foo"
                else role_auth_setup["role_bar_id"]
            )
            return auth_module.app.test_request_context(
                "/api/3.0/mlflow/roles/get",
                method="GET",
                query_string={"role_id": str(role_id)},
            )
        case "role_permission_id":
            rp_id = (
                role_auth_setup["role_permission_foo_id"]
                if workspace == "foo"
                else role_auth_setup["role_permission_bar_id"]
            )
            return auth_module.app.test_request_context(
                "/api/3.0/mlflow/roles/permissions/update",
                method="PATCH",
                json={"role_permission_id": rp_id, "permission": READ.name},
            )
        case "workspace":
            return auth_module.app.test_request_context(
                "/api/3.0/mlflow/roles/create",
                method="POST",
                json={"name": "new-role", "workspace": workspace},
            )
        case _:
            raise ValueError(f"Unknown shape: {shape}")


# Authorization matrices are exercised with a single request shape (role_id);
# shape-resolution itself is covered independently below so we don't multiply
# every actor-case by three shape-cases.


@pytest.mark.parametrize(
    ("actor", "workspace", "expected"),
    [
        # Super admin short-circuits regardless of workspace — one case suffices.
        ("super_admin", "foo", True),
        # Outsider has no role anywhere — one case suffices.
        ("outsider", "foo", False),
        # Workspace admins manage only their own workspace.
        ("ws_admin_foo", "foo", True),
        ("ws_admin_foo", "bar", False),
        ("ws_admin_bar", "foo", False),
        ("ws_admin_bar", "bar", True),
        # Plain role membership is not enough to manage — needs workspace MANAGE.
        ("ws_member_foo", "foo", False),
        ("ws_member_foo", "bar", False),
    ],
)
def test_validate_can_manage_roles_authorization(role_auth_setup, actor, workspace, expected):
    role_auth_setup["login_as"](actor)
    with _request_context_for_shape("role_id", role_auth_setup, workspace):
        assert auth_module.validate_can_manage_roles() is expected


@pytest.mark.parametrize("workspace", ["foo", "bar"])
@pytest.mark.parametrize("shape", ["role_id", "role_permission_id", "workspace"])
def test_manage_roles_resolves_workspace_from_each_shape(role_auth_setup, shape, workspace):
    # Sanity check that _get_role_workspace_from_request correctly dispatches
    # on every request shape. Use ws_admin_foo — their answer differs by
    # workspace, so an incorrectly resolved (or swapped) workspace flips the
    # result and the test fails.
    role_auth_setup["login_as"]("ws_admin_foo")
    expected = workspace == "foo"
    with _request_context_for_shape(shape, role_auth_setup, workspace):
        assert auth_module.validate_can_manage_roles() is expected


@pytest.mark.parametrize(
    ("actor", "workspace", "expected"),
    [
        ("super_admin", "foo", True),
        ("outsider", "foo", False),
        ("ws_admin_foo", "foo", True),
        ("ws_admin_foo", "bar", False),
        ("ws_admin_bar", "foo", False),
        ("ws_admin_bar", "bar", True),
        # Unlike manage, a plain workspace member can view roles.
        ("ws_member_foo", "foo", True),
        ("ws_member_foo", "bar", False),
    ],
)
def test_validate_can_view_roles_authorization(role_auth_setup, actor, workspace, expected):
    role_auth_setup["login_as"](actor)
    with _request_context_for_shape("role_id", role_auth_setup, workspace):
        assert auth_module.validate_can_view_roles() is expected


@pytest.mark.parametrize(
    ("actor", "expected"),
    [
        ("super_admin", True),
        # Any non-admin is denied regardless of their workspace memberships —
        # one representative non-admin is enough.
        ("ws_admin_foo", False),
    ],
)
def test_validate_can_list_roles_unscoped_is_super_admin_only(role_auth_setup, actor, expected):
    # No workspace param: only super admins may list every role in the system.
    role_auth_setup["login_as"](actor)
    with auth_module.app.test_request_context("/api/3.0/mlflow/roles/list", method="GET"):
        assert auth_module.validate_can_list_roles() is expected


@pytest.mark.parametrize(
    ("actor", "workspace", "expected"),
    [
        ("super_admin", "foo", True),
        ("outsider", "foo", False),
        ("ws_admin_foo", "foo", True),
        ("ws_admin_foo", "bar", False),
        ("ws_admin_bar", "foo", False),
        ("ws_admin_bar", "bar", True),
        ("ws_member_foo", "foo", True),
        ("ws_member_foo", "bar", False),
    ],
)
def test_validate_can_list_roles_workspace_scoped(role_auth_setup, actor, workspace, expected):
    role_auth_setup["login_as"](actor)
    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/roles/list", method="GET", query_string={"workspace": workspace}
    ):
        assert auth_module.validate_can_list_roles() is expected


def test_validate_can_list_roles_blank_workspace_denied_for_non_admin(role_auth_setup):
    # Blank workspace param hits a *different* branch from the missing-param
    # case: validate_can_list_roles checks ``workspace.strip()`` and denies
    # rather than raising, unlike _get_role_workspace_from_request which would
    # raise on blank workspace. Kept as a guard for that specific branch.
    role_auth_setup["login_as"]("ws_admin_foo")
    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/roles/list",
        method="GET",
        query_string={"workspace": "   "},
    ):
        assert auth_module.validate_can_list_roles() is False


def test_validate_can_view_user_roles_self_always_allowed(role_auth_setup):
    # A user can always read their own role list, even one with no roles.
    # Using ``outsider`` (zero roles) exercises the self-short-circuit without
    # any membership helping.
    role_auth_setup["login_as"]("outsider")
    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/users/roles/list",
        method="GET",
        query_string={"username": "outsider"},
    ):
        assert auth_module.validate_can_view_user_roles() is True


@pytest.mark.parametrize(
    ("requester", "target", "expected"),
    [
        ("super_admin", "ws_member_foo", True),
        ("ws_admin_foo", "ws_member_foo", True),
        ("ws_admin_bar", "ws_member_foo", False),
        ("ws_member_foo", "ws_admin_foo", False),
        ("outsider", "ws_member_foo", False),
    ],
)
def test_validate_can_view_user_roles_cross_user(role_auth_setup, requester, target, expected):
    role_auth_setup["login_as"](requester)
    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/users/roles/list",
        method="GET",
        query_string={"username": target},
    ):
        assert auth_module.validate_can_view_user_roles() is expected


def test_validate_can_view_user_roles_nonexistent_target_denied_for_non_admin(
    role_auth_setup,
):
    # Non-existent target: return False rather than leaking existence via the
    # RESOURCE_DOES_NOT_EXIST the handler would raise downstream.
    role_auth_setup["login_as"]("ws_admin_foo")
    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/users/roles/list",
        method="GET",
        query_string={"username": "ghost"},
    ):
        assert auth_module.validate_can_view_user_roles() is False


def test_validate_can_view_user_roles_nonexistent_target_allowed_for_super_admin(
    role_auth_setup,
):
    # Super admin short-circuits before the target lookup — they're authorized
    # regardless of whether the target exists (the handler then 404s cleanly).
    role_auth_setup["login_as"]("super_admin")
    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/users/roles/list",
        method="GET",
        query_string={"username": "ghost"},
    ):
        assert auth_module.validate_can_view_user_roles() is True


@pytest.mark.parametrize("shape", ["role_id", "role_permission_id"])
def test_validate_can_manage_roles_nonexistent_resource_denied(role_auth_setup, shape):
    # A non-admin pointing at a role/role_permission that doesn't exist fails
    # closed: _get_role_workspace_from_request returns None and the validator
    # treats that as unauthorized rather than leaking existence.
    role_auth_setup["login_as"]("ws_admin_foo")
    bogus_id = 999_999
    if shape == "role_id":
        ctx = auth_module.app.test_request_context(
            "/api/3.0/mlflow/roles/get",
            method="GET",
            query_string={"role_id": str(bogus_id)},
        )
    else:
        ctx = auth_module.app.test_request_context(
            "/api/3.0/mlflow/roles/permissions/update",
            method="PATCH",
            json={"role_permission_id": bogus_id, "permission": READ.name},
        )
    with ctx:
        assert auth_module.validate_can_manage_roles() is False


def test_validate_can_manage_roles_nonexistent_role_id_bypassed_by_super_admin(
    role_auth_setup,
):
    # Super admins skip the workspace resolution entirely — an unresolvable
    # role_id still produces True at the validator layer.
    role_auth_setup["login_as"]("super_admin")
    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/roles/get",
        method="GET",
        query_string={"role_id": "999999"},
    ):
        assert auth_module.validate_can_manage_roles() is True


def test_validate_can_manage_roles_missing_workspace_params_raises(role_auth_setup):
    # No role_id / role_permission_id / workspace in the request body: the
    # resolver raises INVALID_PARAMETER_VALUE — callers that hit this path have
    # a client bug, and we surface it instead of silently denying.
    role_auth_setup["login_as"]("ws_admin_foo")
    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/roles/create", method="POST", json={}
    ):
        with pytest.raises(MlflowException, match="must include one of"):
            auth_module.validate_can_manage_roles()


def test_validate_can_manage_roles_blank_workspace_raises(role_auth_setup):
    role_auth_setup["login_as"]("ws_admin_foo")
    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/roles/create",
        method="POST",
        json={"name": "new-role", "workspace": "   "},
    ):
        with pytest.raises(MlflowException, match="non-empty string"):
            auth_module.validate_can_manage_roles()


def test_validate_can_manage_roles_propagates_param_coercion_errors(role_auth_setup):
    # Integration check: a non-integer role_id in the request surfaces the
    # coercion error through the validator chain rather than silently denying.
    role_auth_setup["login_as"]("ws_admin_foo")
    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/roles/get",
        method="GET",
        query_string={"role_id": "not-an-int"},
    ):
        with pytest.raises(MlflowException, match="must be an integer"):
            auth_module.validate_can_manage_roles()
