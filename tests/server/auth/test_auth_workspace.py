import asyncio
import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
from flask import Response, request

from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.exceptions import MlflowException
from mlflow.prompt.constants import IS_PROMPT_TAG_KEY
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.server import auth as auth_module
from mlflow.server.auth.permissions import DENY, EDIT, MANAGE, NO_PERMISSIONS, READ, USE
from mlflow.server.auth.requirements import ACTION_NOT_DENIED, Requirement
from mlflow.server.auth.routes import (
    CREATE_PROMPTLAB_RUN,
    GET_ARTIFACT,
    GET_METRIC_HISTORY_BULK,
    GET_METRIC_HISTORY_BULK_INTERVAL,
    GET_MODEL_VERSION_ARTIFACT,
    GET_TRACE_ARTIFACT,
    GET_TRACE_ARTIFACT_V3,
    SEARCH_DATASETS,
    UPLOAD_ARTIFACT,
)
from mlflow.server.auth.sqlalchemy_store import RoleGrantRow, SqlAlchemyStore
from mlflow.utils import workspace_context

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

    # The simplified two-tier model lives in a single ``resource_type='workspace'``
    # slot: ``admin`` carries MANAGE (admin grant), ``user`` carries USE (regular
    # member). The permission tier distinguishes the two without needing a separate
    # ``resource_type`` discriminant.
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

    # ``user`` still got created (best-effort seeding).
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
        mcp_server_workspaces: dict[str, str] | None = None,
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
        self._mcp_server_workspaces = mcp_server_workspaces or {}
        self.engine = engine
        self.ManagedSessionMaker = ManagedSessionMaker

    def get_experiment(self, experiment_id: str):
        if experiment_id not in self._experiment_workspaces:
            raise MlflowException(
                f"Experiment {experiment_id!r} not found", RESOURCE_DOES_NOT_EXIST
            )
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

    def get_mcp_server(self, name: str):
        if name not in self._mcp_server_workspaces:
            raise MlflowException(
                f"MCP server not found ({name})",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        return SimpleNamespace(workspace=self._mcp_server_workspaces[name])

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
    def __init__(self, model_workspaces: dict[str, str], prompts: set[str] | None = None):
        self._model_workspaces = model_workspaces
        self._prompts = prompts or set()

    def get_registered_model(self, name: str):
        if name not in self._model_workspaces:
            raise MlflowException(
                f"Registered Model with name={name!r} not found",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        is_prompt = name in self._prompts
        return SimpleNamespace(
            workspace=self._model_workspaces[name],
            _is_prompt=lambda: is_prompt,
        )


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
        mcp_server_workspaces={"server-1": "team-a", "server-2": "team-a"},
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
    """Replace the user's workspace grant on ``team-a`` with ``permission``.

    The ``workspace_permission_setup`` fixture pre-grants MANAGE so each test
    starts from a known authority; this helper rewrites that grant. ``permission
    == NO_PERMISSIONS`` is treated as "clear the grant" since the simplified
    model rejects NO_PERMISSIONS as a workspace-grant value — absence of a
    grant combined with ``default_permission=NO_PERMISSIONS`` produces the same
    deny semantics the explicit row used to provide.
    """
    if permission == NO_PERMISSIONS.name:
        try:
            store.delete_workspace_permission("team-a", username)
        except MlflowException:
            pass
        return
    store.set_workspace_permission("team-a", username, permission)


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


def test_list_workspaces_hides_workspace_with_only_synthetic_resource_grant(tmp_path, monkeypatch):
    # Pins the docs scenario: a user with only a per-resource grant (e.g.
    # ``(experiment, exp-456, EDIT)``) inside team-a does NOT see team-a in
    # ``mlflow.list_workspaces()`` — workspace visibility requires a
    # workspace-scoped role assignment, not a synthetic per-resource grant.
    #
    # The grant lives on Carol's synthetic ``__user_<id>__`` role with a
    # single ``(experiment, exp-456, EDIT)`` row — no ``(workspace, *)``
    # grant. ``list_accessible_workspace_names`` should treat this as "no
    # workspace-level access" and filter team-a out.
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

    auth_store.create_user("carol", "supersecurepassword", is_admin=False)
    # Direct per-resource grant via ``grant_user_permission`` — writes to
    # Carol's synthetic ``__user_<id>__`` role in team-a. This is exactly the
    # surface the admin UI's "Direct permissions" section sits on top of.
    with workspace_context.WorkspaceContext("team-a"):
        auth_store.grant_user_permission("carol", "experiment", "exp-456", "EDIT")

    monkeypatch.setattr(
        auth_module, "authenticate_request", lambda: SimpleNamespace(username="carol")
    )

    response = Response(
        json.dumps({"workspaces": [{"name": "team-a"}, {"name": "team-b"}]}),
        mimetype="application/json",
    )
    auth_module.filter_list_workspaces(response)
    payload = json.loads(response.get_data(as_text=True))
    # protobuf-to-JSON omits empty repeated fields, so an empty list of
    # workspaces serializes as ``{}``. Both forms mean "Carol sees no
    # workspaces", which is what the docs claim.
    assert [ws["name"] for ws in payload.get("workspaces", [])] == [], (
        "A direct-permission-only user must not see team-a in list_workspaces; "
        "only workspace-scoped role assignments confer visibility."
    )

    auth_store.engine.dispose()


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
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)

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


def test_experiment_validators_workspace_use_allows_create_but_blocks_reads_on_others(
    workspace_permission_setup,
):
    # Workspace USE confers create + workspace access, not read on others'
    # resources — that needs an explicit per-resource grant.
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/experiments/get", method="GET", query_string={"experiment_id": "exp-1"}
    ):
        assert not auth_module.validate_can_read_experiment()
        assert not auth_module.validate_can_update_experiment()
        assert not auth_module.validate_can_delete_experiment()
        assert not auth_module.validate_can_manage_experiment()

    with workspace_context.WorkspaceContext("team-a"):
        assert auth_module.validate_can_create_experiment()


def test_workspace_permission_max_merges_legacy_and_role(workspace_permission_setup):
    # Operators mid-migration may have BOTH a legacy grant and a role grant.
    # The effective permission must be the higher of the two — neither side
    # should silently downgrade the other.
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]

    # Legacy USE + role MANAGE → effective MANAGE.
    _set_workspace_permission(store, username, USE.name)

    role = store.create_role(name="ws-admin", workspace="team-a")
    store.add_role_permission(role.id, "workspace", "*", MANAGE.name)
    user = store.get_user(username)
    store.assign_role_to_user(user.id, role.id)

    with workspace_context.WorkspaceContext("team-a"):
        assert auth_module.validate_can_create_experiment()


def test_use_workspace_permission_allows_create_but_blocks_reads_and_writes_on_others(
    workspace_permission_setup,
):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)

    with workspace_context.WorkspaceContext("team-a"):
        assert auth_module.validate_can_create_experiment()
        assert auth_module.validate_can_create_registered_model()
        assert auth_module.validate_can_create_mcp_server(username)

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/experiments/get", method="GET", query_string={"experiment_id": "exp-1"}
    ):
        assert not auth_module.validate_can_read_experiment()
        assert not auth_module.validate_can_update_experiment()
        assert not auth_module.validate_can_delete_experiment()
        assert not auth_module.validate_can_manage_experiment()

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/experiments/get-by-name",
        method="GET",
        query_string={"experiment_name": "Primary Experiment"},
    ):
        assert not auth_module.validate_can_read_experiment_by_name()

    with (
        workspace_context.WorkspaceContext("team-a"),
        auth_module.app.test_request_context(
            "/api/2.0/mlflow/registered-models/get",
            method="GET",
            query_string={"name": "model-xyz"},
        ),
    ):
        assert not auth_module.validate_can_read_registered_model()
        assert not auth_module.validate_can_update_registered_model()
        assert not auth_module.validate_can_delete_registered_model()
        assert not auth_module.validate_can_manage_registered_model()


def test_no_permissions_blocks_create(workspace_permission_setup):
    # Without any access to the workspace, create is denied.
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)

    with workspace_context.WorkspaceContext("team-a"):
        assert not auth_module.validate_can_create_experiment()
        assert not auth_module.validate_can_create_registered_model()
        assert not auth_module.validate_can_create_mcp_server(username)
        assert not auth_module.validate_can_create_gateway_secret()


def test_gateway_secret_create_requires_workspace_create_grant(workspace_permission_setup):
    # Gateway secret creation crosses the same workspace create boundary as
    # experiments and registered models (GHSA-4449-4cjp-ffp5).
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]

    _set_workspace_permission(store, username, USE.name)
    with workspace_context.WorkspaceContext("team-a"):
        assert auth_module.validate_can_create_gateway_secret()

    _set_workspace_permission(store, username, NO_PERMISSIONS.name)
    with workspace_context.WorkspaceContext("team-a"):
        assert not auth_module.validate_can_create_gateway_secret()


def test_role_grant_workspace_use_allows_create(workspace_permission_setup, monkeypatch):
    # Workspace-wide USE grant ('workspace', '*', USE) confers create rights.
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    user_id = store.get_user(username).id
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    # Drop the legacy grant the fixture installs so we exercise the role-only
    # path in isolation.
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)

    role = store.create_role(name="ws-contributor", workspace="team-a")
    store.add_role_permission(role.id, "workspace", "*", USE.name)
    store.assign_role_to_user(user_id, role.id)

    with workspace_context.WorkspaceContext("team-a"):
        assert auth_module.validate_can_create_experiment()
        assert auth_module.validate_can_create_registered_model()
        assert auth_module.validate_can_create_gateway_secret()


def test_role_grant_resource_type_use_does_not_allow_create(
    workspace_permission_setup, monkeypatch
):
    # Resource-specific USE on ``experiment`` doesn't confer workspace-wide
    # create rights — only workspace-wide grants do.
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    user_id = store.get_user(username).id
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)

    role = store.create_role(name="exp-user", workspace="team-a")
    store.add_role_permission(role.id, "experiment", "*", USE.name)
    store.assign_role_to_user(user_id, role.id)

    with workspace_context.WorkspaceContext("team-a"):
        assert not auth_module.validate_can_create_experiment()
        assert not auth_module.validate_can_create_registered_model()
        assert not auth_module.validate_can_create_gateway_secret()


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

    _set_workspace_permission(store, username, USE.name)

    with auth_module.app.test_request_context(
        "/ajax-api/2.0/mlflow-artifacts/artifacts/1/path",
        method="GET",
    ):
        request.view_args = {"artifact_path": "1/path"}
        assert not auth_module.validate_can_read_experiment_artifact_proxy()
        assert not auth_module.validate_can_update_experiment_artifact_proxy()
        assert not auth_module.validate_can_delete_experiment_artifact_proxy()


def test_experiment_artifact_proxy_without_experiment_id_uses_workspace_permissions(
    workspace_permission_setup,
):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)

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


def test_experiment_artifact_proxy_resolves_experiment_id_under_workspace_prefix(
    workspace_permission_setup,
):
    # In a non-default workspace the proxied artifact path is prefixed with
    # ``workspaces/<ws>/``. The experiment id must still be resolved from the path so
    # per-experiment grants apply, instead of falling back to the workspace-tier grant
    # (which, for a USE member, has no can_update and would wrongly reject writes).
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]

    # "DataScientist" shape: workspace USE (member, no can_update at the workspace tier)
    # plus an explicit experiment-level EDIT grant.
    _set_workspace_permission(store, username, USE.name)
    store.create_experiment_permission("1", username, EDIT.name)

    prefixed_path = "workspaces/team-a/1/run-1/artifacts/plots/x.png"
    with auth_module.app.test_request_context(
        f"/ajax-api/2.0/mlflow-artifacts/artifacts/{prefixed_path}",
        method="GET",
    ):
        request.view_args = {"artifact_path": prefixed_path}
        # EDIT on the experiment resolves through the workspace prefix -> reads and
        # writes are allowed even though the workspace-tier grant is only USE.
        assert auth_module.validate_can_read_experiment_artifact_proxy()
        assert auth_module.validate_can_update_experiment_artifact_proxy()
        # EDIT does not confer delete.
        assert not auth_module.validate_can_delete_experiment_artifact_proxy()


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
    # Role with ('workspace', '*', USE) should grant read access to all experiments.
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    user_id = store.get_user(username).id
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)

    role = store.create_role(name="ws-user", workspace="team-a")
    store.add_role_permission(role.id, "workspace", "*", "USE")
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


def test_run_validators_workspace_use_blocks_reads_and_writes(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/runs/get", method="GET", query_string={"run_id": "run-1"}
    ):
        assert not auth_module.validate_can_read_run()
        assert not auth_module.validate_can_update_run()
        assert not auth_module.validate_can_delete_run()
        assert not auth_module.validate_can_manage_run()


def test_create_model_version_source_read_blocks_cross_workspace(
    workspace_permission_setup, monkeypatch
):
    # Target model lives in team-a (user has MANAGE); source run/model live in team-b
    # (user has no access). The source-read check must block anchoring the model version
    # at a run/model the caller cannot read, even though they can update the target model.
    tracking_store = _TrackingStore(
        experiment_workspaces={"exp-a": "team-a", "exp-b": "team-b"},
        run_experiments={"run-a": "exp-a", "run-b": "exp-b"},
        trace_experiments={},
        logged_model_experiments={"model-a": "exp-a", "model-b": "exp-b"},
    )
    monkeypatch.setattr(auth_module, "_get_tracking_store", lambda: tracking_store)

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/model-versions/create",
        method="POST",
        json={"name": "model-xyz", "source": "s3://bucket/x", "run_id": "run-b"},
    ):
        assert not auth_module.validate_can_create_model_version()

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/model-versions/create",
        method="POST",
        json={"name": "model-xyz", "source": "s3://bucket/x", "model_id": "model-b"},
    ):
        assert not auth_module.validate_can_create_model_version()

    # The camelCase proto aliases the handler accepts must be blocked the same way.
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/model-versions/create",
        method="POST",
        json={"name": "model-xyz", "source": "s3://bucket/x", "runId": "run-b"},
    ):
        assert not auth_module.validate_can_create_model_version()

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/model-versions/create",
        method="POST",
        json={"name": "model-xyz", "source": "s3://bucket/x", "modelId": "model-b"},
    ):
        assert not auth_module.validate_can_create_model_version()

    # Same-workspace source (team-a) is allowed.
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/model-versions/create",
        method="POST",
        json={"name": "model-xyz", "source": "s3://bucket/x", "run_id": "run-a"},
    ):
        assert auth_module.validate_can_create_model_version()

    # A truthy non-dict JSON body must be coerced to {} rather than raising
    # AttributeError on `.get(...)`. Bypass the target-model check (which reads
    # `name` from the body) to isolate the source-read coercion.
    monkeypatch.setattr(
        auth_module,
        "_registered_model_or_prompt_target",
        lambda: (auth_module.RESOURCE_TYPE_REGISTERED_MODEL, "model-xyz"),
    )
    monkeypatch.setattr(auth_module, "_authorize_create_version", lambda _target: True)
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/model-versions/create",
        method="POST",
        json=[1],
    ):
        assert auth_module.validate_can_create_model_version()


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

    _set_workspace_permission(store, username, USE.name)
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/logged-models/get",
        method="GET",
        query_string={"model_id": "model-1"},
    ):
        assert not auth_module.validate_can_read_logged_model()
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


def test_scorer_validators_workspace_use_blocks_reads_and_writes(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/scorers/get",
        method="GET",
        query_string={"experiment_id": "exp-1", "name": "score-1"},
    ):
        assert not auth_module.validate_can_read_scorer()
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
        user = store.get_user(auth_module.authenticate_request().username)
        assert store.is_workspace_admin(user.id, "team-a")
        assert workspace_context.get_request_workspace() == "team-a"
        assert auth_module.validate_can_create_registered_model()

        _set_workspace_permission(store, username, USE.name)
        with auth_module.app.test_request_context(
            "/api/2.0/mlflow/registered-models/get",
            method="GET",
            query_string={"name": "model-xyz"},
        ):
            assert not auth_module.validate_can_read_registered_model()
            assert not auth_module.validate_can_update_registered_model()
            assert not auth_module.validate_can_delete_registered_model()
            assert not auth_module.validate_can_manage_registered_model()
        # USE still confers create rights via creator-as-owner.
        assert auth_module.validate_can_create_registered_model()


def test_prompt_validators_require_manage_for_writes(workspace_permission_setup, monkeypatch):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    registry_store = _RegistryStore({"prompt-xyz": "team-a"}, prompts={"prompt-xyz"})
    monkeypatch.setattr(auth_module, "_get_model_registry_store", lambda: registry_store)

    with workspace_context.WorkspaceContext("team-a"):
        _set_workspace_permission(store, username, MANAGE.name)
        with auth_module.app.test_request_context(
            "/api/2.0/mlflow/registered-models/get",
            method="GET",
            query_string={"name": "prompt-xyz"},
        ):
            assert auth_module.validate_can_read_prompt()
            assert auth_module.validate_can_update_prompt()
            assert auth_module.validate_can_delete_prompt()
            assert auth_module.validate_can_manage_prompt()

        _set_workspace_permission(store, username, USE.name)
        with auth_module.app.test_request_context(
            "/api/2.0/mlflow/registered-models/get",
            method="GET",
            query_string={"name": "prompt-xyz"},
        ):
            assert not auth_module.validate_can_read_prompt()
            assert not auth_module.validate_can_update_prompt()
            assert not auth_module.validate_can_delete_prompt()
            assert not auth_module.validate_can_manage_prompt()


def test_prompt_dispatch_routes_request_by_is_prompt_tag(workspace_permission_setup, monkeypatch):
    # Shared registered-model route resolves to the prompt resource_type only when
    # the entity is a prompt; the dispatching wrappers pick via
    # `_get_permission_from_registered_model_or_prompt_name()` (single fetch + `._is_prompt()`).
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    registry_store = _RegistryStore(
        {"prompt-xyz": "team-a", "model-xyz": "team-a"},
        prompts={"prompt-xyz"},
    )
    monkeypatch.setattr(auth_module, "_get_model_registry_store", lambda: registry_store)

    # Grant prompt READ only — NOT registered_model READ.
    role = store.create_role(name="prompt-reader", workspace="team-a")
    store.add_role_permission(role.id, "prompt", "prompt-xyz", "READ")
    user = store.get_user(username)
    store.assign_role_to_user(user.id, role.id)
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)

    with workspace_context.WorkspaceContext("team-a"):
        # The prompt name maps to the prompt validator and succeeds.
        with auth_module.app.test_request_context(
            "/api/2.0/mlflow/registered-models/get",
            method="GET",
            query_string={"name": "prompt-xyz"},
        ):
            assert auth_module._validate_can_read_registered_model_or_prompt()

        # A non-prompt name maps to the registered_model validator and FAILS:
        # pins cross-resource isolation (prompt grant ≠ registered_model grant).
        with auth_module.app.test_request_context(
            "/api/2.0/mlflow/registered-models/get",
            method="GET",
            query_string={"name": "model-xyz"},
        ):
            assert not auth_module._validate_can_read_registered_model_or_prompt()


def test_registered_model_grant_does_not_satisfy_prompt_request(
    workspace_permission_setup, monkeypatch
):
    """The inverse of the dispatch test: a `(registered_model, foo, READ)`
    grant must NOT satisfy a request for prompt `foo` after the resource_type
    split.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    registry_store = _RegistryStore({"foo": "team-a"}, prompts={"foo"})
    monkeypatch.setattr(auth_module, "_get_model_registry_store", lambda: registry_store)

    role = store.create_role(name="rm-reader", workspace="team-a")
    store.add_role_permission(role.id, "registered_model", "foo", "READ")
    user = store.get_user(username)
    store.assign_role_to_user(user.id, role.id)
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)

    with workspace_context.WorkspaceContext("team-a"):
        with auth_module.app.test_request_context(
            "/api/2.0/mlflow/registered-models/get",
            method="GET",
            query_string={"name": "foo"},
        ):
            # registered_model grant must not leak into the prompt namespace.
            assert not auth_module.validate_can_read_prompt()


@pytest.mark.parametrize(
    ("is_prompt", "deny_tier", "allowed"),
    [
        (True, "prompt_version", False),
        (True, "registered_model_version", True),
        (False, "registered_model_version", False),
        (False, "prompt_version", True),
    ],
    ids=[
        "prompt-denied-by-prompt-version",
        "prompt-unaffected-by-model-version",
        "model-denied-by-model-version",
        "model-unaffected-by-prompt-version",
    ],
)
def test_model_version_artifact_vetoes_on_the_persisted_family(
    workspace_permission_setup, monkeypatch, is_prompt, deny_tier, allowed
):
    """A prompt is a registered model carrying a tag and `_get_sql_model_version` has no prompt
    guard, so a prompt version reaches the artifact route. The veto must consult the family the
    entity actually belongs to, or a prompt_version DENY is unenforceable there while a
    registered_model_version DENY over-blocks.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    registry_store = _RegistryStore({"foo": "team-a"}, prompts={"foo"} if is_prompt else set())
    monkeypatch.setattr(auth_module, "_get_model_registry_store", lambda: registry_store)
    _set_workspace_permission(store, username, USE.name)
    # The positive gate is master's and resolves the registered_model tier for either family.
    _grant(
        store,
        username,
        "team-a",
        [("registered_model", "foo", MANAGE.name), (deny_tier, "*", DENY.name)],
    )

    with workspace_context.WorkspaceContext("team-a"):
        with auth_module.app.test_request_context(
            "/api/2.0/mlflow-artifacts/model-version/artifact",
            method="GET",
            query_string={"name": "foo", "version": "3", "path": "MLmodel"},
        ):
            assert auth_module.validate_can_read_model_version_artifact() is allowed


def test_request_targets_prompt_is_registry_driven_not_body_driven(
    workspace_permission_setup, monkeypatch
):
    # Spoofing regression: classification reads the persisted tag, not the body.
    # Otherwise `(prompt, foo, MANAGE)` could flip the namespace on a non-CREATE
    # registered-model route via a spoofed body tag.
    # ``foo`` exists as a regular registered model, not a prompt.
    registry_store = _RegistryStore({"foo": "team-a"}, prompts=set())
    monkeypatch.setattr(auth_module, "_get_model_registry_store", lambda: registry_store)

    with workspace_context.WorkspaceContext("team-a"):
        # Spoofed body tag on a non-CREATE route must NOT classify ``foo`` as a prompt.
        with auth_module.app.test_request_context(
            "/api/2.0/mlflow/registered-models/delete",
            method="DELETE",
            json={
                "name": "foo",
                "tags": [{"key": "mlflow.prompt.is_prompt", "value": "true"}],
            },
        ):
            assert not auth_module._request_targets_prompt()

        # Without the spoofed tag, the persisted-state lookup still says
        # registered_model — body tags have no impact on classification.
        with auth_module.app.test_request_context(
            "/api/2.0/mlflow/registered-models/delete",
            method="DELETE",
            json={"name": "foo"},
        ):
            assert not auth_module._request_targets_prompt()


def test_request_targets_prompt_persisted_prompt_classifies_true(
    workspace_permission_setup, monkeypatch
):
    # Mirror of the spoofing regression: persisted `_is_prompt() == True` routes
    # to the prompt validator regardless of the body.
    registry_store = _RegistryStore({"foo": "team-a"}, prompts={"foo"})
    monkeypatch.setattr(auth_module, "_get_model_registry_store", lambda: registry_store)

    with workspace_context.WorkspaceContext("team-a"):
        with auth_module.app.test_request_context(
            "/api/2.0/mlflow/registered-models/get",
            method="GET",
            query_string={"name": "foo"},
        ):
            assert auth_module._request_targets_prompt()


def test_request_targets_prompt_unknown_entity_falls_back_to_registered_model(
    workspace_permission_setup, monkeypatch
):
    # Non-existent entity → False (registered-model path surfaces the 404).
    # Body-tag spoof can't force the prompt namespace because the body is ignored.
    registry_store = _RegistryStore({})  # empty — every lookup raises 404
    monkeypatch.setattr(auth_module, "_get_model_registry_store", lambda: registry_store)

    with workspace_context.WorkspaceContext("team-a"):
        with auth_module.app.test_request_context(
            "/api/2.0/mlflow/registered-models/create",
            method="POST",
            json={
                "name": "new-prompt",
                "tags": [{"key": "mlflow.prompt.is_prompt", "value": "true"}],
            },
        ):
            assert not auth_module._request_targets_prompt()


def test_request_targets_prompt_propagates_unexpected_errors(
    workspace_permission_setup, monkeypatch
):
    # Non-`RESOURCE_DOES_NOT_EXIST` errors must propagate; silencing them would
    # quietly route every request down the registered-model path.

    class _BrokenRegistryStore:
        def get_registered_model(self, name):
            raise RuntimeError("registry store backend is down")

    monkeypatch.setattr(auth_module, "_get_model_registry_store", lambda: _BrokenRegistryStore())

    with workspace_context.WorkspaceContext("team-a"):
        with auth_module.app.test_request_context(
            "/api/2.0/mlflow/registered-models/get",
            method="GET",
            query_string={"name": "foo"},
        ):
            with pytest.raises(RuntimeError, match="registry store backend is down"):
                auth_module._request_targets_prompt()


def test_filter_search_registered_models_uses_prompt_grant_for_prompt_rows(
    workspace_permission_setup, monkeypatch
):
    # A user holding only ``(prompt, foo, READ)`` previously had prompt ``foo``
    # silently filtered out of ``SearchRegisteredModels`` results because the
    # filter checked the ``registered_model`` namespace exclusively. With the
    # per-row classify, the prompt grant satisfies the prompt row.
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)
    role = store.create_role(name="prompt-reader", workspace="team-a")
    store.add_role_permission(role.id, "prompt", "foo", READ.name)
    store.assign_role_to_user(store.get_user(username).id, role.id)

    payload = json.dumps({
        "registered_models": [
            {"name": "foo", "tags": [{"key": IS_PROMPT_TAG_KEY, "value": "true"}]},
            {"name": "bar", "tags": []},
        ],
        "next_page_token": "",
    })
    flask_resp = Response(payload, mimetype="application/json")

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/registered-models/search",
        method="GET",
        query_string={"max_results": "100"},
    ):
        auth_module.filter_search_registered_models(flask_resp)

    out = json.loads(flask_resp.get_data(as_text=True))
    names = [rm["name"] for rm in out.get("registered_models", [])]
    # Prompt ``foo`` is kept (grant satisfies prompt namespace); ``bar`` is filtered out.
    assert names == ["foo"]


def test_filter_search_registered_models_does_not_satisfy_prompt_with_rm_grant(
    workspace_permission_setup, monkeypatch
):
    # Inverse direction: a ``(registered_model, foo, READ)`` grant must NOT
    # leak through and make a prompt row readable. Pins cross-namespace
    # isolation on the response-filtering path.
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)
    role = store.create_role(name="rm-reader", workspace="team-a")
    store.add_role_permission(role.id, "registered_model", "foo", READ.name)
    store.assign_role_to_user(store.get_user(username).id, role.id)

    payload = json.dumps({
        "registered_models": [
            {"name": "foo", "tags": [{"key": IS_PROMPT_TAG_KEY, "value": "true"}]},
        ],
        "next_page_token": "",
    })
    flask_resp = Response(payload, mimetype="application/json")

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/registered-models/search",
        method="GET",
        query_string={"max_results": "100"},
    ):
        auth_module.filter_search_registered_models(flask_resp)

    out = json.loads(flask_resp.get_data(as_text=True))
    assert out.get("registered_models", []) == []


def test_filter_search_model_versions_uses_prompt_grant_for_prompt_versions(
    workspace_permission_setup, monkeypatch
):
    # Same gap on ``SearchModelVersions``: prompt versions carry the
    # ``mlflow.prompt.is_prompt`` tag and must be checked against ``prompt``
    # grants, not ``registered_model`` grants.
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)
    role = store.create_role(name="prompt-reader", workspace="team-a")
    store.add_role_permission(role.id, "prompt", "foo", READ.name)
    store.assign_role_to_user(store.get_user(username).id, role.id)

    payload = json.dumps({
        "model_versions": [
            {"name": "foo", "tags": [{"key": IS_PROMPT_TAG_KEY, "value": "true"}]},
            {"name": "bar", "tags": []},
        ],
    })
    flask_resp = Response(payload, mimetype="application/json")

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/model-versions/search", method="GET"
    ):
        auth_module.filter_search_model_versions(flask_resp)

    out = json.loads(flask_resp.get_data(as_text=True))
    names = [mv["name"] for mv in out.get("model_versions", [])]
    assert names == ["foo"]


def test_rename_registered_model_permission_sweeps_prompt_namespace(
    workspace_permission_setup,
):
    # Renaming a prompt must propagate to ``(prompt, old_name, ...)`` grants.
    # Without sweeping both namespaces, the rename leaves those grants orphaned
    # under the old name and creates nothing under the new one.
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    store.grant_user_permission(username, "prompt", "foo", READ.name)
    # Confirm the seed grant landed where we expect.
    user_id = store.get_user(username).id
    before = {
        (rp.resource_type, rp.resource_pattern)
        for role in store.list_user_roles(user_id)
        for rp in role.permissions
    }
    assert ("prompt", "foo") in before

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/registered-models/rename",
        method="POST",
        json={"name": "foo", "new_name": "bar"},
    ):
        auth_module.rename_registered_model_permission(Response(status=200))

    after = {
        (rp.resource_type, rp.resource_pattern)
        for role in store.list_user_roles(user_id)
        for rp in role.permissions
    }
    assert ("prompt", "foo") not in after
    assert ("prompt", "bar") in after


def test_delete_can_manage_registered_model_permission_sweeps_prompt_namespace(
    workspace_permission_setup,
):
    # Deleting a prompt must sweep its ``(prompt, name, ...)`` grants.
    # Without the prompt-side delete, those rows would leak permanently.
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    store.grant_user_permission(username, "prompt", "foo", READ.name)
    user_id = store.get_user(username).id
    before = {
        (rp.resource_type, rp.resource_pattern)
        for role in store.list_user_roles(user_id)
        for rp in role.permissions
    }
    assert ("prompt", "foo") in before

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/registered-models/delete",
        method="DELETE",
        json={"name": "foo"},
    ):
        auth_module.delete_can_manage_registered_model_permission(Response(status=200))

    after = {
        (rp.resource_type, rp.resource_pattern)
        for role in store.list_user_roles(user_id)
        for rp in role.permissions
    }
    assert ("prompt", "foo") not in after


def test_set_can_manage_registered_model_permission_grants_prompt_for_prompt_entity(
    workspace_permission_setup,
):
    # ``CreateRegisteredModel`` is shared with prompt creation. When the
    # created entity is a prompt, the creator-default MANAGE grant must land
    # in the ``prompt`` namespace — otherwise the prompt-side validators
    # immediately lock the creator out of their own prompt.
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    user_id = store.get_user(username).id

    flask_resp = Response(
        json.dumps({
            "registered_model": {
                "name": "my-prompt",
                "tags": [{"key": IS_PROMPT_TAG_KEY, "value": "true"}],
            }
        }),
        mimetype="application/json",
    )
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/registered-models/create",
        method="POST",
        json={"name": "my-prompt"},
    ):
        auth_module.set_can_manage_registered_model_permission(flask_resp)

    grants = {
        (rp.resource_type, rp.resource_pattern, rp.permission)
        for role in store.list_user_roles(user_id)
        for rp in role.permissions
    }
    assert ("prompt", "my-prompt", MANAGE.name) in grants
    assert ("registered_model", "my-prompt", MANAGE.name) not in grants


def test_set_can_manage_registered_model_permission_grants_registered_model_for_plain_entity(
    workspace_permission_setup,
):
    # Inverse: a non-prompt registered model still grants in the
    # ``registered_model`` namespace — pins that the new classification path
    # didn't accidentally flip the default for normal models.
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    user_id = store.get_user(username).id

    flask_resp = Response(
        json.dumps({"registered_model": {"name": "my-model", "tags": []}}),
        mimetype="application/json",
    )
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/registered-models/create",
        method="POST",
        json={"name": "my-model"},
    ):
        auth_module.set_can_manage_registered_model_permission(flask_resp)

    grants = {
        (rp.resource_type, rp.resource_pattern, rp.permission)
        for role in store.list_user_roles(user_id)
        for rp in role.permissions
    }
    assert ("registered_model", "my-model", MANAGE.name) in grants
    assert ("prompt", "my-model", MANAGE.name) not in grants


def test_filter_search_registered_models_classifies_refetched_rows(
    workspace_permission_setup, monkeypatch
):
    # The initial filter pass works on protos; if it doesn't fill
    # ``max_results``, the loop refetches more rows as ORM
    # ``RegisteredModel`` entities. Those ORM rows have ``.tags`` that hide
    # the ``mlflow.prompt.is_prompt`` key, so naive ``_proto_is_prompt`` on
    # them would misclassify every prompt as a registered_model. Pins that
    # ``_entity_is_prompt`` dispatches to ``_is_prompt()`` on the ORM side.
    from mlflow.entities.model_registry import RegisteredModel
    from mlflow.entities.model_registry.registered_model_tag import RegisteredModelTag
    from mlflow.store.entities import PagedList
    from mlflow.utils.search_utils import SearchUtils

    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)
    role = store.create_role(name="prompt-reader", workspace="team-a")
    store.add_role_permission(role.id, "prompt", "refetched-prompt", READ.name)
    store.assign_role_to_user(store.get_user(username).id, role.id)

    # Initial response is empty + has a next_page_token so the refetch loop
    # runs. Then a fake registry returns one prompt + one registered_model.
    refetched_rows = [
        RegisteredModel(
            name="refetched-prompt",
            tags=[RegisteredModelTag(key=IS_PROMPT_TAG_KEY, value="true")],
        ),
        RegisteredModel(name="refetched-model", tags=[]),
    ]
    # First refetch returns the seed rows; subsequent calls return an empty
    # page so the loop terminates instead of spinning on the same fake page.
    calls = {"count": 0}

    def fake_search(**_kwargs):
        calls["count"] += 1
        return PagedList(refetched_rows if calls["count"] == 1 else [], token=None)

    fake_registry = SimpleNamespace(search_registered_models=fake_search)
    monkeypatch.setattr(auth_module, "_get_model_registry_store", lambda: fake_registry)

    # ``SearchUtils.parse_start_offset_from_page_token`` requires a real
    # base64-encoded JSON token; use the project's helper so the loop's
    # offset-bookkeeping doesn't reject our seed.
    seed_token = SearchUtils.create_page_token(1).decode("utf-8")
    flask_resp = Response(
        json.dumps({"registered_models": [], "next_page_token": seed_token}),
        mimetype="application/json",
    )
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/registered-models/search",
        method="GET",
        query_string={"max_results": "10"},
    ):
        auth_module.filter_search_registered_models(flask_resp)

    out = json.loads(flask_resp.get_data(as_text=True))
    names = [rm["name"] for rm in out.get("registered_models", [])]
    # The refetched prompt row is kept (prompt grant satisfies it); the
    # plain registered_model row is filtered out.
    assert names == ["refetched-prompt"]


def test_delete_can_manage_registered_model_permission_rejects_missing_name(
    workspace_permission_setup,
):
    # ``request.get_json(silent=True)`` returns ``None`` on missing /
    # unparsable bodies; the guard must surface a clean 400 instead of a
    # ``TypeError`` -> 500.
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/registered-models/delete",
        method="DELETE",
        data="",  # empty body
        content_type="application/json",
    ):
        with pytest.raises(MlflowException, match="Missing value for required parameter 'name'"):
            auth_module.delete_can_manage_registered_model_permission(Response(status=200))


def test_rename_registered_model_permission_rejects_missing_fields(
    workspace_permission_setup,
):
    # Missing ``name`` / ``new_name`` must raise INVALID_PARAMETER_VALUE
    # rather than silently forwarding ``None`` to
    # ``rename_grants_for_resource`` where it would corrupt grants.
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/registered-models/rename",
        method="POST",
        json={"name": "foo"},  # no new_name
    ):
        with pytest.raises(MlflowException, match="Missing value for required parameter"):
            auth_module.rename_registered_model_permission(Response(status=200))


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


@pytest.mark.parametrize("path", [GET_TRACE_ARTIFACT, GET_TRACE_ARTIFACT_V3])
def test_trace_artifact_validator_uses_workspace_permissions(workspace_permission_setup, path):
    with auth_module.app.test_request_context(
        path,
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
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)

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
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)

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
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)

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
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)

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
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)

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
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)

    with auth_module.app.test_request_context(
        CREATE_PROMPTLAB_RUN,
        method="POST",
        json={"experiment_id": "exp-2"},
    ):
        assert not auth_module.validate_can_create_promptlab_run()


@pytest.mark.parametrize("path", [GET_TRACE_ARTIFACT, GET_TRACE_ARTIFACT_V3])
def test_trace_artifact_validator_denied_without_workspace_permission(
    workspace_permission_setup, path
):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)

    with auth_module.app.test_request_context(
        path,
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


def test_presigned_upload_logged_model_cross_workspace_access_denied(
    workspace_permission_setup, monkeypatch
):
    tracking_store = _TrackingStore(
        experiment_workspaces={"exp-other-ws": "team-b"},
        run_experiments={},
        trace_experiments={},
        logged_model_experiments={"m-other-ws": "exp-other-ws"},
    )
    monkeypatch.setattr(auth_module, "_get_tracking_store", lambda: tracking_store)

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/artifacts/presigned-upload-url",
        method="POST",
        json={"model_id": "m-other-ws", "path": "model.pkl"},
    ):
        assert not auth_module.validate_can_update_run_or_logged_model()


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

    _set_workspace_permission(store, username, NO_PERMISSIONS.name)
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
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)

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


@pytest.mark.parametrize(
    "source_prompt_uri",
    ["prompts:/other/1", "prompts:/other@prod", "other", "other@latest"],
    ids=["uri-version", "uri-alias", "bare-name", "bare-name-alias"],
)
@pytest.mark.parametrize("denied_tier", ["prompt", "prompt_version"])
def test_create_prompt_optimization_job_vetoes_a_bare_source_prompt_name(
    workspace_permission_setup, monkeypatch, source_prompt_uri, denied_tier
):
    """`load_prompt` normalizes through `parse_prompt_name_or_uri`, which resolves ANY
    non-`prompts:/` string to `prompts:/<name>@latest`. So a bare name reaches the registry exactly
    as a URI does, but the veto matched only the URI form and returned NO requirements for a bare
    name -- skipping both the prompt and prompt_version tiers on a job that loads that prompt and
    registers a new version under it, in a worker with no caller identity.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    _set_workspace_permission(store, username, USE.name)
    rows = [
        ("experiment", "exp-1", EDIT.name),
        (denied_tier, "*" if denied_tier == "prompt_version" else "other", DENY.name),
    ]
    _grant(store, username, "team-a", rows)
    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/prompt-optimization/jobs",
        method="POST",
        json={"experiment_id": "exp-1", "source_prompt_uri": source_prompt_uri},
    ):
        assert auth_module.validate_can_create_prompt_optimization_job() is False


@pytest.mark.parametrize("source_prompt_uri", ["prompts:/", "@prod"])
def test_create_prompt_optimization_job_fails_closed_on_an_unreadable_source_prompt(
    workspace_permission_setup, monkeypatch, source_prompt_uri
):
    """Non-empty but no name could be read. The worker still resolves it somehow, so refuse rather
    than authorize a prompt the auth layer could not identify.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    _set_workspace_permission(store, username, MANAGE.name)
    _grant(store, username, "team-a", [("experiment", "exp-1", MANAGE.name)])
    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/prompt-optimization/jobs",
        method="POST",
        json={"experiment_id": "exp-1", "source_prompt_uri": source_prompt_uri},
    ):
        assert auth_module.validate_can_create_prompt_optimization_job() is False


def test_create_prompt_optimization_job_allows_a_source_prompt_with_no_denial(
    workspace_permission_setup, monkeypatch
):
    """The veto must not turn into a positive requirement: no prompt grant still passes."""
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("experiment", "exp-1", EDIT.name)])
    for uri in ("prompts:/other/1", "other", ""):
        with auth_module.app.test_request_context(
            "/api/3.0/mlflow/prompt-optimization/jobs",
            method="POST",
            json={"experiment_id": "exp-1", "source_prompt_uri": uri},
        ):
            assert auth_module.validate_can_create_prompt_optimization_job() is True, uri


@pytest.mark.parametrize(
    ("filter_string", "run_denied_blocks"),
    [
        ("", False),
        ("name = 'm'", False),
        ("source_path LIKE 'x%'", False),
        ("tags.k = 'v'", False),
        ("run_id = 'run-1'", True),
        ("run_id IN ('run-1','run-2')", True),
        ("run_id != 'run-1'", True),
        ("garbage((", True),
    ],
)
def test_search_model_versions_gates_a_filter_that_selects_a_run(
    workspace_permission_setup, monkeypatch, filter_string, run_denied_blocks
):
    """`_withhold_denied_version_siblings` strips `run_id`/`run_link` from the rows, but which rows
    MATCH is itself the disclosure: `run_id = '<id>'` confirms a version came from that run even
    when the field comes back empty. Same reasoning as `_authorize_trace_search`.

    An unparsable filter counts as selecting, so the most restrictive reading applies.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("run", "*", DENY.name)])
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/model-versions/search",
        method="GET",
        query_string={"filter": filter_string} if filter_string else {},
    ):
        assert auth_module.validate_can_search_model_versions() is not run_denied_blocks


@pytest.mark.parametrize("filter_string", ["run_id = 'run-1'", "garbage(("])
def test_search_model_versions_run_filter_is_veto_only(
    workspace_permission_setup, monkeypatch, filter_string
):
    """No run grant at all still passes: the tier vetoes, it does not become a positive gate."""
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    _set_workspace_permission(store, username, USE.name)
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/model-versions/search",
        method="GET",
        query_string={"filter": filter_string},
    ):
        assert auth_module.validate_can_search_model_versions() is True


@pytest.mark.parametrize(
    ("query", "allowed"),
    [
        ({}, True),
        ({"provider": "openai"}, True),
        ({"secret_id": "secret-2"}, True),
        ({"secret_id": "secret-1"}, False),
    ],
    ids=["no-selector", "provider-only", "other-secret", "denied-secret"],
)
def test_gateway_model_definition_list_gates_a_denied_secret_selector(
    workspace_permission_setup, monkeypatch, query, allowed
):
    """`secret_id` is a membership oracle the row redaction cannot close: the rows drop
    `secret_id`/`secret_name`, but filtering ON it still reveals which endpoints and definitions use
    that secret.

    `gateway_secret` is id grain, so the gate names the exact secret the request named -- a DENY on
    `secret-1` must not refuse a listing filtered on `secret-2`.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("gateway_secret", "secret-1", DENY.name)])
    validator = auth_module.validate_can_list_gateway_model_definitions
    route = "/api/2.0/mlflow/gateway/model-definitions/list"
    with auth_module.app.test_request_context(route, method="GET", query_string=query):
        assert validator() is allowed


@pytest.mark.parametrize(
    ("denied_type", "allowed"),
    [(None, True), ("gateway_model_definition", False), ("gateway_endpoint", True)],
    ids=["no-deny", "created-type-deny", "unrelated-type-deny"],
)
def test_create_gateway_model_definition_honors_the_created_type_veto(
    workspace_permission_setup, monkeypatch, denied_type, allowed
):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    _set_workspace_permission(store, username, USE.name)
    if denied_type:
        _grant(store, username, "team-a", [(denied_type, "*", DENY.name)])
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/gateway/model-definitions/create", method="POST", json={"name": "d"}
    ):
        assert auth_module.validate_can_create_gateway_model_definition() is allowed


@pytest.mark.parametrize(
    ("denied_type", "allowed"),
    [(None, True), ("mcp_server", False), ("mcp_server_version", True)],
    ids=["no-deny", "created-type-deny", "unrelated-type-deny"],
)
def test_create_mcp_server_honors_the_created_type_veto(
    workspace_permission_setup, monkeypatch, denied_type, allowed
):
    """FastAPI hands the validator an identity, so the veto uses it, not a re-authentication."""
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    if denied_type:
        _grant(store, username, "team-a", [(denied_type, "*", DENY.name)])
    assert auth_module.validate_can_create_mcp_server(username) is allowed


@pytest.mark.parametrize(
    ("body", "denied_type", "allowed"),
    [
        ({}, None, True),
        # Usage tracking defaults ON, so an omitted field still auto-creates an experiment.
        ({}, "experiment", False),
        ({"usage_tracking": False}, "experiment", True),
        ({"experiment_id": "7"}, "experiment", False),
        ({}, "gateway_endpoint", False),
        ({}, "run", True),
    ],
    ids=[
        "no-deny",
        "default-tracking-auto-creates",
        "tracking-off",
        "named-experiment",
        "created-type-deny",
        "unrelated-type-deny",
    ],
)
def test_create_gateway_endpoint_vetoes_the_experiment_it_will_trace_into(
    workspace_permission_setup, monkeypatch, body, denied_type, allowed
):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    monkeypatch.setattr(
        auth_module, "_validate_can_use_model_definitions_for_create", lambda configs: True
    )
    _set_workspace_permission(store, username, USE.name)
    if denied_type:
        _grant(store, username, "team-a", [(denied_type, "*", DENY.name)])
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/gateway/endpoints/create", method="POST", json={"name": "e", **body}
    ):
        assert auth_module.validate_can_create_gateway_endpoint() is allowed


@pytest.mark.parametrize(
    ("body", "attached_experiment", "allowed"),
    [
        ({"usage_tracking": True}, None, False),
        # Already attached, so nothing is auto-created and the veto does not apply.
        ({"usage_tracking": True}, "7", True),
        # An omitted flag never reaches the store's auto-create branch.
        ({}, None, True),
        ({"usage_tracking": False}, None, True),
    ],
    ids=["enable-auto-creates", "already-attached", "flag-omitted", "disable"],
)
def test_update_gateway_endpoint_vetoes_an_auto_created_experiment(
    workspace_permission_setup, monkeypatch, body, attached_experiment, allowed
):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    monkeypatch.setattr(auth_module, "_validate_can_use_model_definitions", lambda configs: True)
    monkeypatch.setattr(
        auth_module,
        "_get_gateway_endpoint_permission",
        lambda endpoint_id: auth_module.get_permission(MANAGE.name),
    )
    monkeypatch.setattr(
        auth_module._get_tracking_store(),
        "get_gateway_endpoint",
        lambda endpoint_id=None, name=None: SimpleNamespace(experiment_id=attached_experiment),
        raising=False,
    )
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("experiment", "*", DENY.name)])
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/gateway/endpoints/update",
        method="POST",
        json={"endpoint_id": "ep-1", **body},
    ):
        assert auth_module.validate_can_update_gateway_endpoint() is allowed


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


def test_prompt_optimization_job_validators_workspace_use_blocks_reads_and_writes(
    workspace_permission_setup, monkeypatch
):
    # Job auth gates on parent experiment permission; workspace USE no longer folds.
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]

    mock_job = SimpleNamespace(params='{"experiment_id": "exp-1"}')
    monkeypatch.setattr(auth_module, "get_job", lambda job_id: mock_job)

    _set_workspace_permission(store, username, USE.name)

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/prompt-optimization/jobs/get",
        method="GET",
        query_string={"job_id": "job-1"},
    ):
        assert not auth_module.validate_can_read_prompt_optimization_job()
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


def _version_row(name, is_prompt=False):
    return SimpleNamespace(name=name, _is_prompt=lambda: is_prompt)


def test_graphql_run_reads_honor_a_run_deny(workspace_permission_setup):
    """GraphQL resolved only the run's EXPERIMENT, so (run, "*", DENY) withheld a run over REST
    while the same run stayed readable over GraphQL. Both transports must answer alike.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "*", EDIT.name),
            ("run", "*", DENY.name),
        ],
    )

    assert auth_module._graphql_can_read_run("run-1", username) is False
    with auth_module.app.test_request_context("/graphql", method="POST"):
        assert auth_module._authorize_run_id("run-1", "read") is False
    # The experiment itself is unaffected: the veto is on the run tier, not its parent.
    assert auth_module._graphql_can_read_experiment("exp-1", username) is True


def test_graphql_run_search_filter_honors_a_run_deny(workspace_permission_setup):
    """mlflowSearchRuns scopes RUN rows, so it carries the same run veto REST's
    filter_experiment_ids does. mlflowSearchDatasets scopes datasets (out of scope) and must not.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    # Workspace MANAGE is workspace-admin, which is deliberately not restrictable (it precedes
    # DENY in resolve_permissions), so the tier is only observable below that level.
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "*", EDIT.name),
            ("run", "*", DENY.name),
        ],
    )
    middleware = auth_module.GraphQLAuthorizationMiddleware()

    def _check(field_name):
        with auth_module.app.test_request_context("/graphql", method="POST"):
            return middleware._check_authorization(
                field_name, {"input": SimpleNamespace(experiment_ids=["exp-1"])}, username
            )

    assert _check("mlflowSearchRuns") is False
    assert _check("mlflowSearchDatasets") is True


def _search_model_versions_names(rows):
    payload = json.dumps({"model_versions": rows})
    flask_resp = Response(payload, mimetype="application/json")
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/model-versions/search", method="GET"
    ):
        auth_module.filter_search_model_versions(flask_resp)
    out = json.loads(flask_resp.get_data(as_text=True))
    return [mv["name"] for mv in out.get("model_versions", [])]


def _search_registered_models_names(rows):
    payload = json.dumps({"registered_models": rows, "next_page_token": ""})
    flask_resp = Response(payload, mimetype="application/json")
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/registered-models/search",
        method="GET",
        query_string={"max_results": "100"},
    ):
        auth_module.filter_search_registered_models(flask_resp)
    out = json.loads(flask_resp.get_data(as_text=True))
    return [rm["name"] for rm in out.get("registered_models", [])]


def _run_artifact_proxy(validator, artifact_path, method="GET"):
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow-artifacts/artifacts",
        method=method,
        query_string={"path": artifact_path},
    ):
        return getattr(auth_module, validator)()


@pytest.mark.parametrize(
    ("tier", "artifact_path"),
    [
        ("run", "1/abc123/artifacts/model.pkl"),
        ("logged_model", "1/models/m-abc/artifacts/data.bin"),
        ("trace", "1/traces/tr-1/artifacts/spans.json"),
    ],
)
def test_artifact_proxy_honors_child_tier_deny(workspace_permission_setup, tier, artifact_path):
    """The proxy serves a repository path directly, and the path encodes the child. Checking
    only the leading experiment id let a child DENY be bypassed by the concrete proxy path,
    while the point artifact routes refused it.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "*", MANAGE.name),
            (tier, "*", DENY.name),
        ],
    )

    assert (
        _run_artifact_proxy("validate_can_read_experiment_artifact_proxy", artifact_path) is False
    )
    assert (
        _run_artifact_proxy(
            "validate_can_update_experiment_artifact_proxy", artifact_path, method="PUT"
        )
        is False
    )
    assert (
        _run_artifact_proxy(
            "validate_can_delete_experiment_artifact_proxy", artifact_path, method="DELETE"
        )
        is False
    )
    # FastAPI dispatch must not be the softer path.
    assert (
        auth_module._authorize_fastapi_artifact_proxy_child(
            f"/api/2.0/mlflow-artifacts/artifacts/{artifact_path}", username, None, "read"
        )
        is False
    )


@pytest.mark.parametrize(
    ("tier", "artifact_path"),
    [
        ("trace", "1/%2Ftraces%2Ftr-1%2Fartifacts%2Fspans.json"),
        ("trace", "1/%252Ftraces%252Ftr-1%252Fartifacts%252Fspans.json"),
        ("run", "1/%2Fabc123%2Fartifacts%2Fmodel.pkl"),
        ("logged_model", "1/%2Fmodels%2Fm-abc%2Fartifacts%2Fdata.bin"),
    ],
    ids=["trace-encoded", "trace-double-encoded", "run-encoded", "logged_model-encoded"],
)
def test_artifact_proxy_child_tier_survives_path_encoding(
    workspace_permission_setup, tier, artifact_path
):
    """The auth layer and the handler must classify the same string. Flask decodes view_args once,
    then every proxy handler calls validate_path_is_safe, which decodes AGAIN -- so an encoded path
    reached the child classifier as one opaque segment naming no tier, while the handler resolved it
    to the real child path and served the content.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "*", MANAGE.name),
            (tier, "*", DENY.name),
        ],
    )
    assert (
        _run_artifact_proxy("validate_can_read_experiment_artifact_proxy", artifact_path) is False
    )
    assert (
        _run_artifact_proxy(
            "validate_can_delete_experiment_artifact_proxy", artifact_path, method="DELETE"
        )
        is False
    )
    # FastAPI dispatch must not be the softer path.
    assert (
        auth_module._authorize_fastapi_artifact_proxy_child(
            f"/api/2.0/mlflow-artifacts/artifacts/{artifact_path}", username, None, "read"
        )
        is False
    )


@pytest.mark.parametrize(
    ("experiment_grant", "allowed"),
    [(None, False), ("DENY", False), ("READ", True)],
    ids=["no-experiment-grant", "experiment-deny", "experiment-read"],
)
def test_artifact_proxy_parent_gate_survives_an_encoded_experiment_id(
    workspace_permission_setup, monkeypatch, experiment_grant, allowed
):
    """An unparsed experiment id is not a denial -- it falls through to the workspace grant (or
    `default_permission` with workspaces off), so failing to canonicalize here substituted a
    workspace-wide answer for a per-experiment one. `%31/...` reached the parser as an opaque
    segment while the handler resolved experiment 1.

    The encoded path must now answer exactly as the plain one does, for every grant state.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    # Workspace USE is what the parent gate used to fall back on.
    _set_workspace_permission(store, username, USE.name)
    if experiment_grant:
        _grant(store, username, "team-a", [("experiment", "1", experiment_grant)])

    for path in ("1/plain.txt", "%31/plain.txt"):
        assert (
            _run_artifact_proxy("validate_can_read_experiment_artifact_proxy", path) is allowed
        ), path
    # FastAPI resolves the parent from the URL rather than view_args, so it needs its own proof.
    for path in ("1/plain.txt", "%31/plain.txt"):
        permission = auth_module._get_proxy_artifact_permission(
            f"/api/2.0/mlflow-artifacts/artifacts/{path}", username, None
        )
        assert permission.can_read is allowed, path


def test_artifact_proxy_parent_gate_canonicalizes_the_list_query_path(
    workspace_permission_setup, monkeypatch
):
    """List-artifacts carries the path as ?path=, a separate branch of the FastAPI extractor."""
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("experiment", "1", DENY.name)])

    for query_path in ("1/plain.txt", "%31/plain.txt"):
        permission = auth_module._get_proxy_artifact_permission(
            "/api/2.0/mlflow-artifacts/artifacts", username, query_path
        )
        assert permission.can_read is False, query_path


def test_artifact_proxy_fails_closed_on_a_path_the_handler_would_reject(
    workspace_permission_setup,
):
    """Traversal never reaches a child decision. Refusing costs nothing -- validate_path_is_safe
    raises on exactly these paths in the handler too.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("experiment", "*", MANAGE.name)])
    assert (
        _run_artifact_proxy("validate_can_read_experiment_artifact_proxy", "1/../../etc/passwd")
        is False
    )


def test_artifact_proxy_child_deny_does_not_cross_tiers(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "*", MANAGE.name),
            ("run", "*", DENY.name),
        ],
    )

    assert (
        _run_artifact_proxy(
            "validate_can_read_experiment_artifact_proxy", "1/traces/tr-1/artifacts/f"
        )
        is True
    )
    # An experiment-level path names no child, so the experiment alone governs it.
    assert (
        _run_artifact_proxy("validate_can_read_experiment_artifact_proxy", "1/plain-file.txt")
        is True
    )


def test_artifact_proxy_still_inherits_from_the_experiment(workspace_permission_setup):
    """No child grant: the experiment decides, exactly as before."""
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("experiment", "*", MANAGE.name)])

    assert (
        _run_artifact_proxy(
            "validate_can_read_experiment_artifact_proxy", "1/abc123/artifacts/model.pkl"
        )
        is True
    )


def test_version_point_reads_honor_a_version_deny(workspace_permission_setup):
    """A version DENY must withhold a version whether it is fetched by name or found by searching;
    GetModelVersion and friends consulted only the parent. GetRegisteredModel shares the old
    validator and must stay parent-only, since a version denial should not hide the parent.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("registered_model", "*", READ.name),
            ("registered_model_version", "*", DENY.name),
        ],
    )

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/model-versions/get", query_string={"name": "model-xyz", "version": "3"}
    ):
        assert auth_module.validate_can_read_model_or_prompt_version() is False
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/registered-models/get", query_string={"name": "model-xyz"}
    ):
        assert auth_module._validate_can_read_registered_model_or_prompt() is True


def test_version_point_reads_still_inherit_from_the_parent(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("registered_model", "*", READ.name)])

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/model-versions/get", query_string={"name": "model-xyz", "version": "3"}
    ):
        assert auth_module.validate_can_read_model_or_prompt_version() is True


def test_version_read_filters_honor_a_version_deny(workspace_permission_setup, monkeypatch):
    """The version tier withholds VERSION rows without hiding their parents from a model list --
    which is why the veto lives in a version-specific predicate rather than the shared one that
    also filters registered-model rows.
    """
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("registered_model", "*", READ.name),
            ("registered_model_version", "*", DENY.name),
        ],
    )

    assert _search_model_versions_names([{"name": "model-xyz", "tags": []}]) == []
    # The parent list is untouched by a version denial.
    assert _search_registered_models_names([{"name": "model-xyz", "tags": []}]) == ["model-xyz"]


def test_version_deny_does_not_cross_families(workspace_permission_setup, monkeypatch):
    """A prompt-version denial must not withhold model versions, and vice versa."""
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("registered_model", "*", READ.name),
            ("prompt", "*", READ.name),
            ("prompt_version", "*", DENY.name),
        ],
    )

    names = _search_model_versions_names([
        {"name": "model-xyz", "tags": []},
        {"name": "my-prompt", "tags": [{"key": IS_PROMPT_TAG_KEY, "value": "true"}]},
    ])
    assert names == ["model-xyz"]


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
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)

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
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)

    _assign_role_with_permission(store, username, "team-a", "gateway_endpoint", "READ")

    # _validate_gateway_use_permission looks up the endpoint by name, resolves
    # the endpoint id, then checks ``can_use`` via the permission resolver.
    with auth_module.app.test_request_context("/"):
        assert auth_module._validate_gateway_use_permission("endpoint-1", username) is False


def test_role_grant_use_on_gateway_endpoint_permits_use(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)

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
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)

    _assign_role_with_permission(store, username, "team-a", "gateway_endpoint", granted)

    with auth_module.app.test_request_context("/"):
        assert (
            auth_module._validate_gateway_use_permission("endpoint-1", username) is expected_can_use
        )


# ---- Workspace-wide role grants on gateway resources ----


@pytest.mark.parametrize(
    ("granted", "expected_read", "expected_manage"),
    [
        # USE doesn't fold into resource lookups; MANAGE does.
        ("USE", False, False),
        ("MANAGE", True, True),
    ],
)
def test_role_workspace_wide_grant_folds_for_manage_only_on_gateway_endpoints(
    workspace_permission_setup, granted, expected_read, expected_manage
):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)

    _assign_role_with_permission(store, username, "team-a", "workspace", granted)

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/gateway/endpoints/get",
        method="GET",
        query_string={"endpoint_id": "endpoint-1"},
    ):
        assert auth_module.validate_can_read_gateway_endpoint() is expected_read
        assert auth_module.validate_can_delete_gateway_endpoint() is expected_manage
        assert auth_module.validate_can_manage_gateway_endpoint() is expected_manage


@pytest.mark.parametrize(
    ("granted", "expected_use"),
    [
        # USE doesn't fold into resource lookups; MANAGE does.
        ("USE", False),
        ("MANAGE", True),
    ],
)
def test_role_workspace_wide_grant_invocation_tier_dependent(
    workspace_permission_setup, granted, expected_use
):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)

    _assign_role_with_permission(store, username, "team-a", "workspace", granted)

    with auth_module.app.test_request_context("/"):
        assert auth_module._validate_gateway_use_permission("endpoint-1", username) is expected_use


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
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)

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
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)

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
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)

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
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)

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
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)

    _assign_role_with_permission(store, username, "team-a", "gateway_endpoint", "READ")
    _assign_role_with_permission(store, username, "team-a", "gateway_endpoint", "MANAGE")

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/gateway/endpoints/get",
        method="GET",
        query_string={"endpoint_id": "endpoint-1"},
    ):
        assert auth_module.validate_can_manage_gateway_endpoint() is True


# =============================================================================
# Authorization for MCP server resources
# =============================================================================


@pytest.mark.parametrize(
    ("granted", "expected_read", "expected_update", "expected_delete", "expected_manage"),
    [
        ("READ", True, False, False, False),
        ("USE", True, False, False, False),
        ("EDIT", True, True, False, False),
        ("MANAGE", True, True, True, True),
    ],
)
def test_role_grant_on_mcp_server_gates_capabilities(
    workspace_permission_setup,
    granted,
    expected_read,
    expected_update,
    expected_delete,
    expected_manage,
):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)
    _assign_role_with_permission(store, username, "team-a", "mcp_server", granted)

    perm = auth_module._get_mcp_server_permission("server-1", username)
    assert perm.can_read is expected_read
    assert perm.can_update is expected_update
    assert perm.can_delete is expected_delete
    assert perm.can_manage is expected_manage


@pytest.mark.parametrize(
    ("version_grant", "allowed"),
    [
        (None, True),
        (MANAGE.name, True),
        (READ.name, False),
        (DENY.name, False),
    ],
    ids=["no-version-grant", "version-manage", "version-read", "version-deny"],
)
def test_deleting_an_mcp_server_takes_the_version_tier_along(
    workspace_permission_setup, monkeypatch, version_grant, allowed
):
    """`DELETE /{name}` destroys the server's versions with it -- the ORM pairs `ondelete="CASCADE"`
    with `delete-orphan` -- so it needs the same version-tier delete the experiment and
    registered-model cascades take. `_mcp_path_targets_a_version` is False for the bare server path,
    so the cascade previously ran on the server's `can_delete` alone: a holder of
    `(mcp_server_version, *, DENY)` could destroy through the parent exactly what
    `DELETE /{name}/versions/{v}` refuses.

    No version grant falls back to the server, which the caller already passed `can_delete` on.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    # MCP server names are namespace/slug; the fixture's entries are single-segment.
    auth_module._get_tracking_store()._mcp_server_workspaces["com.test/srv"] = "team-a"
    _set_workspace_permission(store, username, USE.name)
    rows = [("mcp_server", "com.test/srv", MANAGE.name)]
    if version_grant:
        rows.append(("mcp_server_version", "*", version_grant))
    _grant(store, username, "team-a", rows)

    validator = auth_module._get_mcp_server_validator("/api/3.0/mlflow/mcp-servers/com.test/srv")
    request = SimpleNamespace(method="DELETE", state=SimpleNamespace(), query_params={})
    assert asyncio.run(validator(username, request)) is allowed
    # A nested version route is unaffected: it still answers via the veto, not the delete gate.
    nested = auth_module._get_mcp_server_validator(
        "/api/3.0/mlflow/mcp-servers/com.test/srv/versions/1"
    )
    nested_expected = version_grant != DENY.name
    assert asyncio.run(nested(username, request)) is nested_expected


def test_role_in_other_workspace_does_not_grant_mcp_server_access(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)
    _assign_role_with_permission(store, username, "team-b", "mcp_server", "MANAGE")

    perm = auth_module._get_mcp_server_permission("server-1", username)
    assert perm.can_read is False


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


@pytest.mark.parametrize(
    ("actor", "workspaces", "expected"),
    [
        # Super admin lists across any combination unconditionally.
        ("super_admin", ["foo", "bar"], True),
        # Workspace admin must hold a role in *every* requested workspace.
        ("ws_admin_foo", ["foo"], True),
        ("ws_admin_foo", ["foo", "bar"], False),  # not present in bar
        ("ws_admin_foo", ["foo", "foo"], True),  # duplicate is fine
        # Member of foo can list foo, but not foo + bar.
        ("ws_member_foo", ["foo"], True),
        ("ws_member_foo", ["foo", "bar"], False),
        # Outsider can list nothing.
        ("outsider", ["foo"], False),
        ("outsider", ["foo", "bar"], False),
    ],
)
def test_validate_can_list_roles_multi_workspace(role_auth_setup, actor, workspaces, expected):
    role_auth_setup["login_as"](actor)
    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/roles/list",
        method="GET",
        query_string=[("workspace", w) for w in workspaces],
    ):
        assert auth_module.validate_can_list_roles() is expected


# Listing users is scoped to workspace membership (the review-queue assignment UI
# needs the roster; assigning still requires experiment MANAGE), so the roster
# isn't leaked across workspaces. Super admin is omitted: ``_before_request``
# short-circuits via ``sender_is_admin`` before the validator is reached.
@pytest.mark.parametrize(
    ("actor", "workspace", "expected"),
    [
        # Workspace-wide grant carrying can_use (USE/MANAGE) → may list the roster.
        ("ws_admin_foo", "foo", True),
        # Isolation: an admin of another workspace can't list users in this one.
        ("ws_admin_foo", "bar", False),
        ("ws_admin_bar", "foo", False),
        # A plain experiment-level grant is not a workspace-wide grant → denied.
        ("ws_member_foo", "foo", False),
        # No grant anywhere.
        ("outsider", "foo", False),
    ],
)
def test_validate_can_list_users_workspace_scoped(role_auth_setup, actor, workspace, expected):
    role_auth_setup["login_as"](actor)
    token = workspace_context.set_server_request_workspace(workspace)
    try:
        with auth_module.app.test_request_context("/api/2.0/mlflow/users/list", method="GET"):
            assert auth_module.validate_can_list_users() is expected
    finally:
        workspace_context._WORKSPACE.reset(token)


def test_validate_can_list_users_allows_any_user_without_workspaces(role_auth_setup, monkeypatch):
    # With workspaces disabled there is no isolation boundary, so any authenticated
    # user may list the roster (single-tenant). ``role_auth_setup`` enables
    # workspaces; override it back off for this case.
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")
    role_auth_setup["login_as"]("outsider")
    with auth_module.app.test_request_context("/api/2.0/mlflow/users/list", method="GET"):
        assert auth_module.validate_can_list_users() is True


def test_list_users_handler_eager_loads_scoped_roles(role_auth_setup):
    # Workspace admin: bulk response includes per-user roles, scoped to
    # workspaces the requester administers (plus self, unscoped).
    role_auth_setup["login_as"]("ws_admin_foo")
    with auth_module.app.test_request_context("/api/2.0/mlflow/users/list", method="GET"):
        response = auth_module.list_users()
    by_username = {u["username"]: u for u in response.get_json()["users"]}

    # Self: own admin role visible.
    assert {(r["workspace"], r["name"]) for r in by_username["ws_admin_foo"]["roles"]} == {
        ("foo", "admin-foo")
    }
    # Cross-user in foo: filtered to foo only (member-foo lives in foo).
    assert {(r["workspace"], r["name"]) for r in by_username["ws_member_foo"]["roles"]} == {
        ("foo", "member-foo")
    }
    # Cross-user outside requester's admin set: roles hidden.
    assert by_username["ws_admin_bar"]["roles"] == []
    assert by_username["outsider"]["roles"] == []


def test_list_users_handler_super_admin_sees_every_role(role_auth_setup):
    role_auth_setup["login_as"]("super_admin")
    with auth_module.app.test_request_context("/api/2.0/mlflow/users/list", method="GET"):
        response = auth_module.list_users()
    by_username = {u["username"]: u for u in response.get_json()["users"]}

    assert {(r["workspace"], r["name"]) for r in by_username["ws_admin_foo"]["roles"]} == {
        ("foo", "admin-foo")
    }
    assert {(r["workspace"], r["name"]) for r in by_username["ws_admin_bar"]["roles"]} == {
        ("bar", "admin-bar")
    }
    assert {(r["workspace"], r["name"]) for r in by_username["ws_member_foo"]["roles"]} == {
        ("foo", "member-foo")
    }


@pytest.mark.parametrize(
    ("actor", "expected"),
    [
        ("ws_admin_foo", True),
        ("ws_admin_bar", True),
        ("ws_member_foo", False),
        ("outsider", False),
    ],
)
def test_validate_can_create_user(role_auth_setup, actor, expected):
    role_auth_setup["login_as"](actor)
    with auth_module.app.test_request_context("/api/2.0/mlflow/users/create", method="POST"):
        assert auth_module.validate_can_create_user() is expected


def test_validate_can_delete_user_stays_super_admin_only(role_auth_setup):
    # Regression: the create-user widening must not have leaked into delete.
    for actor in ("ws_admin_foo", "outsider"):
        role_auth_setup["login_as"](actor)
        with auth_module.app.test_request_context("/api/2.0/mlflow/users/delete", method="DELETE"):
            assert auth_module.validate_can_delete_user() is False


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


def test_role_permission_resolver_honors_default_workspace_autogrant(monkeypatch):
    """Resource-level resolution must fall back to ``default_permission`` for an
    ungranted user in the configured default workspace when
    ``grant_default_workspace_access=true``. Without this, deployments that
    relied on the implicit auto-grant pre-simplification suddenly see
    ``NO_PERMISSIONS`` for resources in the default workspace.
    """
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    monkeypatch.setattr(
        auth_module,
        "auth_config",
        auth_module.auth_config._replace(
            default_permission=READ.name,
            grant_default_workspace_access=True,
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
        def get_user(self, username):
            return SimpleNamespace(id=42, username=username)

        def list_grants(self, user_id, workspace, resource_types):
            return []

    monkeypatch.setattr(auth_module, "store", DummyStore(), raising=False)
    monkeypatch.setattr(
        auth_module,
        "_get_resource_workspace",
        lambda *args, **kwargs: default_workspace,
    )

    role_perm = auth_module._role_permission_for(
        username="alice",
        resource_type="experiment",
        resource_key="exp-1",
        workspace_lookup_id="exp-1",
        workspace_fetcher=lambda _id: SimpleNamespace(),
        workspace_label="experiment",
    )
    perm = auth_module._get_role_permission_or_default(role_perm)
    assert perm.name == READ.name


def test_role_permission_resolver_denies_in_non_default_workspace(monkeypatch):
    """The auto-grant only applies to the configured default workspace. An
    ungranted user in any other workspace must still get ``NO_PERMISSIONS``,
    even if ``grant_default_workspace_access`` is enabled.
    """
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    monkeypatch.setattr(
        auth_module,
        "auth_config",
        auth_module.auth_config._replace(
            default_permission=READ.name,
            grant_default_workspace_access=True,
        ),
        raising=False,
    )

    monkeypatch.setattr(auth_module, "_get_workspace_store", lambda: None, raising=False)
    monkeypatch.setattr(
        auth_module,
        "get_default_workspace_optional",
        lambda *args, **kwargs: (SimpleNamespace(name="team-default"), True),
        raising=False,
    )

    class DummyStore:
        def get_user(self, username):
            return SimpleNamespace(id=42, username=username)

        def list_grants(self, user_id, workspace, resource_types):
            return []

    monkeypatch.setattr(auth_module, "store", DummyStore(), raising=False)
    monkeypatch.setattr(
        auth_module,
        "_get_resource_workspace",
        lambda *args, **kwargs: "other-workspace",
    )

    role_perm = auth_module._role_permission_for(
        username="alice",
        resource_type="experiment",
        resource_key="exp-1",
        workspace_lookup_id="exp-1",
        workspace_fetcher=lambda _id: SimpleNamespace(),
        workspace_label="experiment",
    )
    perm = auth_module._get_role_permission_or_default(role_perm)
    assert perm.name == NO_PERMISSIONS.name


def _grant(store, username, workspace, rows):
    """Assign ``username`` a fresh role in ``workspace`` carrying ``rows``.

    ``rows`` are ``(resource_type, resource_pattern, permission)`` triples.
    """
    role = store.create_role(f"role-{random_str(10)}", workspace)
    for resource_type, resource_pattern, permission in rows:
        store.add_role_permission(role.id, resource_type, resource_pattern, permission)
    store.assign_role_to_user(store.get_user(username).id, role.id)
    return role


def test_legacy_resolver_lets_deny_beat_a_positive_grant(workspace_permission_setup):
    """The regression `4af3cf834` fixed: the store folded grants with ``max`` and
    ``PERMISSION_PRIORITY[DENY]`` is -1, so a ``DENY`` sharing a role with any positive grant was
    silently discarded. Both rows below match the ``(experiment, exp-1)`` key -- a wildcard pattern
    matches every id -- so the fold decides between them, and ``DENY`` must win.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    # The fixture pre-grants workspace MANAGE, which is stored as a synthetic (workspace, *, MANAGE)
    # role grant and triggers the admin bypass. Drop to the member tier so the experiment rows are
    # what decide.
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "*", READ.name),
            ("experiment", "exp-1", DENY.name),
        ],
    )

    denied = auth_module._get_experiment_permission("exp-1", username)
    assert denied.name == DENY.name
    assert not denied.can_read

    # exp-2 is matched only by the wildcard row, so it keeps the positive grant. This is what
    # makes the assertion above a fold result rather than a blanket failure.
    assert auth_module._get_experiment_permission("exp-2", username).can_read


def test_legacy_resolver_keeps_the_workspace_admin_bypass(workspace_permission_setup):
    """``_role_grant_for_resource`` has to apply ``is_workspace_admin_grant`` itself:
    ``fold_grants_for_key`` ignores rows of a different ``resource_type``, so a workspace-wide
    MANAGE would otherwise never fold into a resource query and admins would lose access.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("workspace", "*", MANAGE.name),
            ("experiment", "exp-1", DENY.name),
        ],
    )

    permission = auth_module._get_experiment_permission("exp-1", username)
    assert permission.name == MANAGE.name
    assert permission.can_manage


def test_legacy_resolver_never_loads_a_child_deny(workspace_permission_setup):
    """Pins the limit `4af3cf834`'s own message records, so the gap stays visible.

    The legacy callers resolve one resource_type and do not pass the parent, so a ``(run, *, DENY)``
    row is never loaded on those paths -- it cannot veto anything resolved through the experiment
    tier. Follow-up item 1 (the §5e baseline) is what makes a child tier participate.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    # Without this the fixture's workspace MANAGE would grant read on its own, and the assertion
    # below would hold whether or not the run DENY was loaded.
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "*", READ.name),
            ("run", "*", DENY.name),
        ],
    )

    assert auth_module._get_experiment_permission("exp-1", username).can_read


# =============================================================================
# The §5e sub-resource baseline: a parent or intermediate veto must not be
# bypassable by a grant on a higher-priority tier. See follow-up item 1.
# =============================================================================


def test_parent_deny_is_not_bypassed_by_a_child_grant(workspace_permission_setup):
    """Hole A. ``fallback_if_no_grant`` means *only* if no grant, so a sufficient run grant ends
    the chain and the experiment is never consulted -- letting a child grant override the
    operator's DENY on the parent. The baseline's positive experiment READ closes it.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "exp-1", DENY.name),
            ("run", "*", MANAGE.name),
        ],
    )

    assert not auth_module._authorize_run_id("run-1", "delete")
    assert not auth_module._authorize_run_id("run-1", "read")


def test_child_wildcard_alone_does_not_confer_access(workspace_permission_setup):
    """The escalation the baseline bounds: run grain is wildcard-only, so ``(run, *, MANAGE)``
    with no experiment grant would otherwise confer delete on every run in every experiment in
    the workspace.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("run", "*", MANAGE.name)])

    assert not auth_module._authorize_run_id("run-1", "delete")


def test_parent_grant_still_inherits_to_the_child_tier(workspace_permission_setup):
    """The baseline must not deny anyone the parent tier allowed -- inheritance keeps working."""
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("experiment", "exp-1", MANAGE.name)])

    assert auth_module._authorize_run_id("run-1", "read")
    assert auth_module._authorize_run_id("run-1", "delete")


def test_trace_deny_is_not_bypassed_by_an_assessment_grant(workspace_permission_setup):
    """Hole B, the three-level chain assessment -> trace -> experiment.

    A sufficient assessment grant ends the chain at the first key, so the operator's trace DENY
    is never consulted. The parent READ baseline does NOT close this -- the experiment grant is
    positive here -- so the intermediate trace tier needs its own veto requirement.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "exp-1", EDIT.name),
            ("trace", "*", DENY.name),
            ("assessment", "*", MANAGE.name),
        ],
    )

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/traces/trace-1/assessments/a-1",
        method="PATCH",
        json={"trace_id": "trace-1"},
    ):
        assert not auth_module.validate_can_update_assessment()


def test_assessment_grant_still_works_without_a_trace_deny(workspace_permission_setup):
    """The trace veto must cost nothing when the operator has not denied the trace tier."""
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "exp-1", READ.name),
            ("assessment", "*", MANAGE.name),
        ],
    )

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/traces/trace-1/assessments/a-1",
        method="PATCH",
        json={"trace_id": "trace-1"},
    ):
        assert auth_module.validate_can_update_assessment()


# =============================================================================
# The read predicate (design doc §5f, follow-up items 2 and 6): a list row and a
# point request must reach the same decision.
# =============================================================================


def test_read_predicate_honors_a_wildcard_deny(workspace_permission_setup):
    """``(experiment, *, DENY)`` alone listed EVERYTHING: the predicate skipped any row failing
    ``can_read``, so DENY fell through to ``default_permission``.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("experiment", "*", DENY.name)])

    predicate = auth_module._role_based_read_predicate(username, "experiment")
    assert not predicate("exp-1")
    assert not predicate("exp-2")


def test_read_predicate_lets_a_specific_deny_override_a_wildcard_read(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "*", READ.name),
            ("experiment", "exp-1", DENY.name),
        ],
    )

    predicate = auth_module._role_based_read_predicate(username, "experiment")
    assert not predicate("exp-1")
    assert predicate("exp-2")


def test_read_predicate_agrees_with_the_point_route(workspace_permission_setup):
    """The property that matters: no grant configuration may make a row visible in a listing but
    unreadable at its point route, or the reverse.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "*", READ.name),
            ("experiment", "exp-1", DENY.name),
        ],
    )

    predicate = auth_module._role_based_read_predicate(username, "experiment")
    for experiment_id in ("exp-1", "exp-2"):
        assert predicate(experiment_id) == (
            auth_module._get_experiment_permission(experiment_id, username).can_read
        ), experiment_id


def test_read_predicate_child_deny_hides_every_row(workspace_permission_setup):
    """A veto stated as an ordinary requirement: listing logged models keys on the EXPERIMENT,
    so without it ``(logged_model, *, DENY)`` was bypassable via ``POST /logged-models/search``.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "*", READ.name),
            ("logged_model", "*", DENY.name),
        ],
    )

    predicate = auth_module._role_based_read_predicate(
        username,
        "experiment",
        also_require=[Requirement("logged_model", "*", ACTION_NOT_DENIED)],
    )
    assert not predicate("exp-1")
    # …while the experiment tier itself stays readable.
    assert auth_module._role_based_read_predicate(username, "experiment")("exp-1")


def test_read_predicate_child_grant_is_never_positive(workspace_permission_setup):
    """An ACTION_NOT_DENIED requirement is satisfied by a grant but never CONFERS read, so a
    wildcard sub-resource grant must not make rows visible the row tier does not allow.

    The workspace grant is removed outright, not merely downgraded: the row requirement falls back
    to the workspace tier, so leaving even USE there would confer read on its own and the
    assertion would pass for the wrong reason.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)
    _grant(store, username, "team-a", [("logged_model", "*", MANAGE.name)])

    predicate = auth_module._role_based_read_predicate(
        username,
        "experiment",
        also_require=[Requirement("logged_model", "*", ACTION_NOT_DENIED)],
    )
    assert not predicate("exp-1")


def test_read_predicate_keeps_the_workspace_admin_bypass(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _grant(
        store,
        username,
        "team-a",
        [
            ("workspace", "*", MANAGE.name),
            ("experiment", "*", DENY.name),
        ],
    )

    predicate = auth_module._role_based_read_predicate(username, "experiment")
    assert predicate("exp-1")


# =============================================================================
# Bulk routes (design doc §5g): many resources in one request, one requirement
# pair per distinct parent. Follow-up item 4.
# =============================================================================


def test_bulk_trace_read_honors_a_trace_deny(workspace_permission_setup):
    """``SearchTraces`` resolved each experiment with ``_get_experiment_permission``, so the trace
    tier was never consulted and ``(trace, *, DENY)`` did not stop it.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "*", READ.name),
            ("trace", "*", DENY.name),
        ],
    )

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/traces", query_string=[("experiment_ids", "exp-1")]
    ):
        assert not auth_module.validate_can_search_traces()


def test_bulk_trace_read_inherits_from_the_experiment(workspace_permission_setup):
    """No trace grant: the experiment tier still governs, for every id."""
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("experiment", "*", READ.name)])

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/traces",
        query_string=[("experiment_ids", "exp-1"), ("experiment_ids", "exp-2")],
    ):
        assert auth_module.validate_can_search_traces()


def test_bulk_trace_read_is_all_or_nothing_on_the_parent(workspace_permission_setup):
    """Master's documented all-or-nothing over distinct parents, preserved: one unreadable
    experiment fails the whole request rather than being filtered out.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("experiment", "exp-1", READ.name)])

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/traces",
        query_string=[("experiment_ids", "exp-1"), ("experiment_ids", "exp-2")],
    ):
        assert not auth_module.validate_can_search_traces()


def test_bulk_metric_history_honors_a_run_deny(workspace_permission_setup):
    """The same shape one tier over: bulk metric history resolves RUNS, so the run tier vetoes."""
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "*", READ.name),
            ("run", "*", DENY.name),
        ],
    )

    with auth_module.app.test_request_context(
        "/ajax-api/2.0/mlflow/metrics/get-history-bulk", query_string=[("run_id", "run-1")]
    ):
        assert not auth_module.validate_can_read_metric_history_bulk()


def test_bulk_metric_history_inherits_from_the_experiment(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("experiment", "*", READ.name)])

    with auth_module.app.test_request_context(
        "/ajax-api/2.0/mlflow/metrics/get-history-bulk",
        query_string=[("run_id", "run-1"), ("run_id", "run-2")],
    ):
        assert auth_module.validate_can_read_metric_history_bulk()


def test_list_user_role_permissions_workspace_is_default_when_workspaces_disabled(
    tmp_path, monkeypatch
):
    # ``_UserRolePermissionRow.workspace`` is typed ``str``, not ``str | None``.
    # When workspaces are disabled, ``_get_active_workspace_name()`` returns the
    # string ``"default"`` (``DEFAULT_WORKSPACE_NAME``) and every role write
    # threads that through ``SqlRole.workspace`` (``nullable=False``). Pins
    # that LIST surfaces ``"default"`` rather than ``None`` in this mode.
    from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

    monkeypatch.delenv(MLFLOW_ENABLE_WORKSPACES.name, raising=False)
    db_uri = f"sqlite:///{tmp_path / 'auth-store.db'}"
    auth_store = SqlAlchemyStore()
    auth_store.init_db(db_uri)
    monkeypatch.setattr(auth_module, "store", auth_store, raising=False)

    username = "alice"
    auth_store.create_user(username, "supersecurepassword", is_admin=False)
    auth_store.grant_user_resource_permission(username, "experiment", "exp-1", READ.name)

    is_admin, rows = auth_module._list_user_role_permissions(username)
    assert is_admin is False
    assert len(rows) == 1
    row = rows[0]
    assert isinstance(row.workspace, str)
    assert row.workspace == DEFAULT_WORKSPACE_NAME
    assert row.resource_type == "experiment"
    assert row.resource_pattern == "exp-1"
    assert row.permission == READ.name
    # Direct grants land on the synthetic ``__user_<id>__`` role.
    assert row.role_name == f"__user_{auth_store.get_user(username).id}__"


def test_list_user_role_permissions_aggregates_synthetic_and_custom_roles(tmp_path, monkeypatch):
    # Direct grants flow into the synthetic ``__user_<id>__`` role; a workspace
    # admin can also assign the user to a named, hand-authored role. LIST must
    # surface BOTH so the FE can split the "Direct permissions" tab (synthetic
    # rows) from the "Role permissions" tab (everything else) by ``role_name``.
    monkeypatch.delenv(MLFLOW_ENABLE_WORKSPACES.name, raising=False)
    db_uri = f"sqlite:///{tmp_path / 'auth-store.db'}"
    auth_store = SqlAlchemyStore()
    auth_store.init_db(db_uri)
    monkeypatch.setattr(auth_module, "store", auth_store, raising=False)

    username = "alice"
    user = auth_store.create_user(username, "supersecurepassword", is_admin=False)
    auth_store.grant_user_resource_permission(username, "experiment", "exp-1", READ.name)

    custom_role = auth_store.create_role(name="ml-reviewer", workspace="default")
    auth_store.add_role_permission(custom_role.id, "registered_model", "*", READ.name)
    auth_store.assign_role_to_user(user.id, custom_role.id)

    is_admin, rows = auth_module._list_user_role_permissions(username)
    assert is_admin is False
    assert len(rows) == 2

    by_role = {r.role_name: r for r in rows}
    synthetic_name = f"__user_{user.id}__"
    assert synthetic_name in by_role
    assert "ml-reviewer" in by_role

    syn_row = by_role[synthetic_name]
    assert syn_row.resource_type == "experiment"
    assert syn_row.resource_pattern == "exp-1"
    assert syn_row.permission == READ.name

    custom_row = by_role["ml-reviewer"]
    assert custom_row.role_id == custom_role.id
    assert custom_row.resource_type == "registered_model"
    assert custom_row.resource_pattern == "*"
    assert custom_row.permission == READ.name

    # Distinct rows really do come from distinct roles (no aliasing on role_id).
    assert syn_row.role_id != custom_row.role_id


def test_list_user_role_permissions_empty_for_user_with_no_roles(tmp_path, monkeypatch):
    # A freshly-created non-admin user with no grants must surface as
    # ``(is_admin=False, rows=[])`` — not raise, and not leak rows from other users.
    monkeypatch.delenv(MLFLOW_ENABLE_WORKSPACES.name, raising=False)
    db_uri = f"sqlite:///{tmp_path / 'auth-store.db'}"
    auth_store = SqlAlchemyStore()
    auth_store.init_db(db_uri)
    monkeypatch.setattr(auth_module, "store", auth_store, raising=False)

    auth_store.create_user("alice", "supersecurepassword", is_admin=False)
    # Second user with a grant — must not leak into alice's rows.
    auth_store.create_user("bob", "supersecurepassword", is_admin=False)
    auth_store.grant_user_resource_permission("bob", "experiment", "exp-1", READ.name)

    is_admin, rows = auth_module._list_user_role_permissions("alice")
    assert is_admin is False
    assert rows == []


def test_list_user_role_permissions_admin_flag_propagates(tmp_path, monkeypatch):
    # ``is_admin`` comes from the User row, not from any grants — admins can
    # have zero permission rows and still surface ``is_admin=True``. The FE
    # relies on this to skip permission-fetch entirely for super admins.
    monkeypatch.delenv(MLFLOW_ENABLE_WORKSPACES.name, raising=False)
    db_uri = f"sqlite:///{tmp_path / 'auth-store.db'}"
    auth_store = SqlAlchemyStore()
    auth_store.init_db(db_uri)
    monkeypatch.setattr(auth_module, "store", auth_store, raising=False)

    auth_store.create_user("rootuser", "supersecurepassword", is_admin=True)

    is_admin, rows = auth_module._list_user_role_permissions("rootuser")
    assert is_admin is True
    assert rows == []


def test_list_user_role_permissions_multiple_perms_on_one_role(tmp_path, monkeypatch):
    # One role with two RolePermission rows must produce two LIST rows that
    # share role_id / role_name / workspace but differ on resource_type /
    # permission. Pins the flatten-on-role.permissions behavior.
    monkeypatch.delenv(MLFLOW_ENABLE_WORKSPACES.name, raising=False)
    db_uri = f"sqlite:///{tmp_path / 'auth-store.db'}"
    auth_store = SqlAlchemyStore()
    auth_store.init_db(db_uri)
    monkeypatch.setattr(auth_module, "store", auth_store, raising=False)

    user = auth_store.create_user("alice", "supersecurepassword", is_admin=False)
    role = auth_store.create_role(name="ml-power-user", workspace="default")
    auth_store.add_role_permission(role.id, "experiment", "*", READ.name)
    auth_store.add_role_permission(role.id, "registered_model", "*", MANAGE.name)
    auth_store.assign_role_to_user(user.id, role.id)

    is_admin, rows = auth_module._list_user_role_permissions("alice")
    assert is_admin is False
    assert len(rows) == 2
    assert {(r.resource_type, r.permission) for r in rows} == {
        ("experiment", READ.name),
        ("registered_model", MANAGE.name),
    }
    assert {r.role_id for r in rows} == {role.id}
    assert {r.role_name for r in rows} == {"ml-power-user"}


def test_list_user_role_permissions_workspace_reflects_role_workspace(tmp_path, monkeypatch):
    # With workspaces enabled, ``row.workspace`` is the *role's* workspace —
    # i.e., the workspace where the resource lives — not always
    # ``DEFAULT_WORKSPACE_NAME``. Pin against silent collapsing of the workspace
    # field for non-default workspaces.
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    db_uri = f"sqlite:///{tmp_path / 'auth-store.db'}"
    auth_store = SqlAlchemyStore()
    auth_store.init_db(db_uri)
    monkeypatch.setattr(auth_module, "store", auth_store, raising=False)

    user = auth_store.create_user("alice", "supersecurepassword", is_admin=False)
    role_a = auth_store.create_role(name="viewer", workspace="team-a")
    auth_store.add_role_permission(role_a.id, "experiment", "*", READ.name)
    auth_store.assign_role_to_user(user.id, role_a.id)

    role_b = auth_store.create_role(name="viewer", workspace="team-b")
    auth_store.add_role_permission(role_b.id, "experiment", "*", READ.name)
    auth_store.assign_role_to_user(user.id, role_b.id)

    _, rows = auth_module._list_user_role_permissions("alice")
    assert {r.workspace for r in rows} == {"team-a", "team-b"}


def _list_user_permissions_response(monkeypatch, requester: str, target: str) -> dict[str, object]:
    """Invoke ``list_user_permissions`` as ``requester`` for ``target`` and return JSON."""
    monkeypatch.setattr(
        auth_module, "authenticate_request", lambda: SimpleNamespace(username=requester)
    )
    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/users/permissions/list",
        method="GET",
        query_string={"username": target},
    ):
        response = auth_module.list_user_permissions()
    return json.loads(response.get_data(as_text=True))


def test_list_user_permissions_admin_sees_rows_across_all_workspaces(tmp_path, monkeypatch):
    # Platform admins must see EVERY role-derived row for the target user,
    # regardless of which workspace the row sits in. This is the path the
    # admin UI's "User detail" page hits when an admin clicks any user.
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    db_uri = f"sqlite:///{tmp_path / 'auth-store.db'}"
    auth_store = SqlAlchemyStore()
    auth_store.init_db(db_uri)
    monkeypatch.setattr(auth_module, "store", auth_store, raising=False)

    auth_store.create_user("root", "supersecurepassword", is_admin=True)
    target = auth_store.create_user("alice", "supersecurepassword", is_admin=False)
    role_a = auth_store.create_role(name="viewer", workspace="team-a")
    auth_store.add_role_permission(role_a.id, "experiment", "*", READ.name)
    auth_store.assign_role_to_user(target.id, role_a.id)
    role_b = auth_store.create_role(name="viewer", workspace="team-b")
    auth_store.add_role_permission(role_b.id, "experiment", "*", READ.name)
    auth_store.assign_role_to_user(target.id, role_b.id)

    payload = _list_user_permissions_response(monkeypatch, requester="root", target="alice")
    assert payload["is_admin"] is False  # is_admin reflects the *target*, not the caller.
    assert {p["workspace"] for p in payload["permissions"]} == {"team-a", "team-b"}


def test_list_user_permissions_self_sees_all_rows(tmp_path, monkeypatch):
    # Self-view sees everything the user holds, across every workspace, with no
    # admin-workspace filtering. ``is_admin`` here mirrors the caller because
    # caller == target.
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    db_uri = f"sqlite:///{tmp_path / 'auth-store.db'}"
    auth_store = SqlAlchemyStore()
    auth_store.init_db(db_uri)
    monkeypatch.setattr(auth_module, "store", auth_store, raising=False)

    alice = auth_store.create_user("alice", "supersecurepassword", is_admin=False)
    role_a = auth_store.create_role(name="viewer", workspace="team-a")
    auth_store.add_role_permission(role_a.id, "experiment", "*", READ.name)
    auth_store.assign_role_to_user(alice.id, role_a.id)
    role_b = auth_store.create_role(name="viewer", workspace="team-b")
    auth_store.add_role_permission(role_b.id, "experiment", "*", READ.name)
    auth_store.assign_role_to_user(alice.id, role_b.id)

    payload = _list_user_permissions_response(monkeypatch, requester="alice", target="alice")
    assert payload["is_admin"] is False
    assert {p["workspace"] for p in payload["permissions"]} == {"team-a", "team-b"}


def test_list_user_permissions_workspace_admin_scoped_to_own_workspaces(tmp_path, monkeypatch):
    # A workspace admin of team-a viewing alice (who has rows in team-a + team-b)
    # must see ONLY team-a rows. The team-b row would leak cross-workspace state
    # the requester has no claim over. Security regression guard.
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    db_uri = f"sqlite:///{tmp_path / 'auth-store.db'}"
    auth_store = SqlAlchemyStore()
    auth_store.init_db(db_uri)
    monkeypatch.setattr(auth_module, "store", auth_store, raising=False)

    auth_store.create_user("wp-admin", "supersecurepassword", is_admin=False)
    auth_store.set_workspace_permission("team-a", "wp-admin", MANAGE.name)

    alice = auth_store.create_user("alice", "supersecurepassword", is_admin=False)
    role_a = auth_store.create_role(name="viewer", workspace="team-a")
    auth_store.add_role_permission(role_a.id, "experiment", "*", READ.name)
    auth_store.assign_role_to_user(alice.id, role_a.id)
    role_b = auth_store.create_role(name="viewer", workspace="team-b")
    auth_store.add_role_permission(role_b.id, "experiment", "*", READ.name)
    auth_store.assign_role_to_user(alice.id, role_b.id)

    payload = _list_user_permissions_response(monkeypatch, requester="wp-admin", target="alice")
    workspaces = {p["workspace"] for p in payload["permissions"]}
    assert workspaces == {"team-a"}, workspaces


def test_list_current_user_permissions_returns_caller_rows_and_admin_flag(tmp_path, monkeypatch):
    # ``/users/current/permissions`` is the unauthenticated-safe self path the
    # FE uses on bootstrap to decide which nav items to render. Caller == target
    # implicitly; ``is_admin`` reflects the caller; rows cover every workspace.
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    db_uri = f"sqlite:///{tmp_path / 'auth-store.db'}"
    auth_store = SqlAlchemyStore()
    auth_store.init_db(db_uri)
    monkeypatch.setattr(auth_module, "store", auth_store, raising=False)

    alice = auth_store.create_user("alice", "supersecurepassword", is_admin=False)
    role = auth_store.create_role(name="viewer", workspace="team-a")
    auth_store.add_role_permission(role.id, "experiment", "*", READ.name)
    auth_store.assign_role_to_user(alice.id, role.id)

    monkeypatch.setattr(
        auth_module, "authenticate_request", lambda: SimpleNamespace(username="alice")
    )
    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/users/current/permissions", method="GET"
    ):
        response = auth_module.list_current_user_permissions()
    payload = json.loads(response.get_data(as_text=True))
    assert payload["is_admin"] is False
    assert len(payload["permissions"]) == 1
    assert payload["permissions"][0]["workspace"] == "team-a"
    assert payload["permissions"][0]["role_name"] == "viewer"


def test_default_permission_floors_lesser_role_grant(monkeypatch):
    monkeypatch.setattr(
        auth_module,
        "auth_config",
        auth_module.auth_config._replace(default_permission=EDIT.name),
        raising=False,
    )
    perm = auth_module._get_role_permission_or_default(lambda: READ)
    assert perm.name == EDIT.name


def test_default_permission_does_not_downgrade_higher_role_grant(monkeypatch):
    monkeypatch.setattr(
        auth_module,
        "auth_config",
        auth_module.auth_config._replace(default_permission=READ.name),
        raising=False,
    )
    perm = auth_module._get_role_permission_or_default(lambda: MANAGE)
    assert perm.name == MANAGE.name


def test_default_permission_does_not_override_explicit_no_permissions(monkeypatch):
    # NO_PERMISSIONS is the explicit-deny carve-out — must survive the floor.
    monkeypatch.setattr(
        auth_module,
        "auth_config",
        auth_module.auth_config._replace(default_permission=MANAGE.name),
        raising=False,
    )
    perm = auth_module._get_role_permission_or_default(lambda: NO_PERMISSIONS)
    assert perm.name == NO_PERMISSIONS.name


def test_default_permission_kicks_in_when_no_grant_matches(monkeypatch):
    # None (no grant matched, workspaces disabled) → fall through to default.
    monkeypatch.setattr(
        auth_module,
        "auth_config",
        auth_module.auth_config._replace(default_permission=EDIT.name),
        raising=False,
    )
    perm = auth_module._get_role_permission_or_default(lambda: None)
    assert perm.name == EDIT.name


def test_user_can_create_in_default_workspace_via_autogrant(monkeypatch):
    """``_user_can_create_in_workspace`` must honor
    ``grant_default_workspace_access`` so an ungranted user in the default
    workspace can still create when ``default_permission.can_use`` is true.
    Regression guard for the legacy-endpoint simplification, which dropped
    the auto-grant fallback when ``_workspace_permission`` was retired.
    """
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    auth = SimpleNamespace(username="alice")
    monkeypatch.setattr(auth_module, "authenticate_request", lambda: auth)
    monkeypatch.setattr(
        auth_module,
        "auth_config",
        auth_module.auth_config._replace(
            default_permission=USE.name,
            grant_default_workspace_access=True,
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
        def get_user(self, username):
            return SimpleNamespace(id=42, username=username)

        def get_role_permission_for_resource(self, *args, **kwargs):
            return None

    monkeypatch.setattr(auth_module, "store", DummyStore(), raising=False)

    # Default workspace + autogrant + USE → allowed.
    with workspace_context.WorkspaceContext(default_workspace):
        assert auth_module._user_can_create_in_workspace()

    # Same config but a non-default workspace → still denied.
    with workspace_context.WorkspaceContext("team-other"):
        assert not auth_module._user_can_create_in_workspace()


def test_user_cannot_create_via_autogrant_when_default_permission_lacks_use(monkeypatch):
    """The auto-grant create-gate gates on ``default_permission.can_use``. If
    the operator pinned ``default_permission=READ`` (read-only access), the
    fallback must not allow create.
    """
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    auth = SimpleNamespace(username="alice")
    monkeypatch.setattr(auth_module, "authenticate_request", lambda: auth)
    monkeypatch.setattr(
        auth_module,
        "auth_config",
        auth_module.auth_config._replace(
            default_permission=READ.name,
            grant_default_workspace_access=True,
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
        def get_user(self, username):
            return SimpleNamespace(id=42, username=username)

        def get_role_permission_for_resource(self, *args, **kwargs):
            return None

    monkeypatch.setattr(auth_module, "store", DummyStore(), raising=False)

    with workspace_context.WorkspaceContext(default_workspace):
        assert not auth_module._user_can_create_in_workspace()


def test_role_based_read_predicate_matches_the_point_route_on_no_permissions_rows(monkeypatch):
    """A ``NO_PERMISSIONS`` row must mean the same thing in a listing as at a point route.

    It used to not: the predicate skipped any row failing ``can_read`` and fell through to
    ``default_permission``, so such a row LISTED while ``_get_experiment_permission`` denied it
    (v2 preserves ``NO_PERMISSIONS`` as the workspace-boundary signal rather than maxing it against
    the default, which is what master did). Asserting the two agree, rather than asserting a
    particular answer, keeps them tied together if either side changes.

    The row type is ungrantable in both versions -- ``RESOURCE_GRANTABLE_PERMISSIONS`` omits it --
    so this configuration is only reachable through a fake like this one, or a legacy DB row.
    """
    monkeypatch.delenv(MLFLOW_ENABLE_WORKSPACES.name, raising=False)
    monkeypatch.setattr(
        auth_module,
        "auth_config",
        auth_module.auth_config._replace(default_permission=READ.name),
        raising=False,
    )

    class DummyStore:
        def get_user(self, username):
            return SimpleNamespace(id=42, username=username)

        def list_grants(self, user_id, workspace, resource_types):
            return [
                RoleGrantRow("experiment", "*", NO_PERMISSIONS.name),
                RoleGrantRow("experiment", "exp-allowed", READ.name),
                RoleGrantRow("experiment", "exp-explicit-deny", NO_PERMISSIONS.name),
            ]

    monkeypatch.setattr(auth_module, "store", DummyStore(), raising=False)

    predicate = auth_module._role_based_read_predicate("alice", "experiment")
    for experiment_id in ("exp-allowed", "exp-other", "exp-explicit-deny"):
        assert predicate(experiment_id) == (
            auth_module._get_experiment_permission(experiment_id, "alice").can_read
        ), experiment_id
    # A specific positive grant still reads, so the parity above is not vacuous.
    assert predicate("exp-allowed")


# =============================================================================
# Unified per-user permission convenience APIs — validator dispatcher tests
# =============================================================================


def _scorer_resource_id(experiment_id: str, scorer_name: str) -> str:
    from mlflow.server.auth.sqlalchemy_store import SqlAlchemyStore

    return SqlAlchemyStore._scorer_pattern(experiment_id, scorer_name)


@pytest.mark.parametrize(
    ("resource_type", "resource_id"),
    [
        ("experiment", "exp-1"),
        ("registered_model", "model-xyz"),
        ("scorer", "exp-1/score-1"),
        ("gateway_secret", "secret-1"),
        ("gateway_endpoint", "endpoint-1"),
        ("gateway_model_definition", "model-def-1"),
    ],
)
def test_validate_can_manage_resource_workspace_manage_allows(
    workspace_permission_setup, resource_type, resource_id
):
    # Workspace MANAGE (the fixture's default) grants per-resource MANAGE on every
    # resource type in the workspace — the dispatcher must route the request
    # through the same code path the legacy validators use.
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, MANAGE.name)

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/users/permissions/grant",
        method="POST",
        json={
            "username": username,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "permission": "READ",
        },
    ):
        assert auth_module.validate_can_manage_resource()


def test_validate_can_manage_resource_per_resource_manage_allows(
    workspace_permission_setup,
):
    # Per-resource MANAGE delegation: a user with MANAGE on exp-1 (and nothing
    # else) can grant other users access to exp-1, mirroring the legacy
    # ``validate_can_manage_experiment`` semantics.
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    user_id = store.get_user(username).id

    _set_workspace_permission(store, username, NO_PERMISSIONS.name)
    role = store.create_role(name="exp-1-manager", workspace="team-a")
    store.add_role_permission(role.id, "experiment", "exp-1", MANAGE.name)
    store.assign_role_to_user(user_id, role.id)

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/users/permissions/grant",
        method="POST",
        json={
            "username": username,
            "resource_type": "experiment",
            "resource_id": "exp-1",
            "permission": "READ",
        },
    ):
        assert auth_module.validate_can_manage_resource()


def test_validate_can_manage_resource_other_resource_denied(workspace_permission_setup):
    # MANAGE on exp-1 doesn't extend to exp-2.
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    user_id = store.get_user(username).id

    _set_workspace_permission(store, username, NO_PERMISSIONS.name)
    role = store.create_role(name="exp-1-manager", workspace="team-a")
    store.add_role_permission(role.id, "experiment", "exp-1", MANAGE.name)
    store.assign_role_to_user(user_id, role.id)

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/users/permissions/grant",
        method="POST",
        json={
            "username": username,
            "resource_type": "experiment",
            "resource_id": "exp-2",
            "permission": "READ",
        },
    ):
        assert not auth_module.validate_can_manage_resource()


def test_validate_can_manage_resource_no_grant_denied(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/users/permissions/grant",
        method="POST",
        json={
            "username": username,
            "resource_type": "experiment",
            "resource_id": "exp-1",
            "permission": "READ",
        },
    ):
        assert not auth_module.validate_can_manage_resource()


def test_validate_can_manage_resource_workspace_use_insufficient(workspace_permission_setup):
    # Workspace USE (regular member) doesn't grant per-resource MANAGE — only
    # the resource owner's grant tier does.
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/users/permissions/grant",
        method="POST",
        json={
            "username": username,
            "resource_type": "experiment",
            "resource_id": "exp-1",
            "permission": "READ",
        },
    ):
        assert not auth_module.validate_can_manage_resource()


def test_validate_can_manage_resource_scorer_dispatch(workspace_permission_setup):
    # The scorer ``resource_id`` is the compound pattern
    # ``"<experiment_id>/<url_quote(scorer_name)>"``. The dispatcher must split
    # off the experiment_id prefix for workspace resolution.
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, MANAGE.name)

    pattern = _scorer_resource_id("exp-1", "score-1")
    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/users/permissions/grant",
        method="POST",
        json={
            "username": username,
            "resource_type": "scorer",
            "resource_id": pattern,
            "permission": "READ",
        },
    ):
        assert auth_module.validate_can_manage_resource()


def test_validate_can_manage_resource_scorer_missing_delimiter_raises(workspace_permission_setup):
    # ``resource_id`` without the ``/`` delimiter cannot be split into
    # ``experiment_id`` for workspace resolution.
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, MANAGE.name)

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/users/permissions/grant",
        method="POST",
        json={
            "username": username,
            "resource_type": "scorer",
            "resource_id": "missing-delimiter",
            "permission": "READ",
        },
    ):
        with pytest.raises(MlflowException, match="Expected '<experiment_id>/<scorer_name>'"):
            auth_module.validate_can_manage_resource()


def test_validate_can_manage_resource_workspace_resource_type_rejected(
    workspace_permission_setup,
):
    # ``workspace`` is intentionally excluded from the unified API — workspace
    # grants live behind set_workspace_permission / delete_workspace_permission.
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, MANAGE.name)

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/users/permissions/grant",
        method="POST",
        json={
            "username": username,
            "resource_type": "workspace",
            "resource_id": "*",
            "permission": "MANAGE",
        },
    ):
        with pytest.raises(MlflowException, match="is not supported by the per-user"):
            auth_module.validate_can_manage_resource()


def test_validate_can_get_user_permission_self_check_allowed(
    workspace_permission_setup,
):
    # A non-admin user can always check their own permissions, even without
    # any workspace presence.
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/users/permissions/get",
        method="GET",
        query_string={
            "username": username,
            "resource_type": "experiment",
            "resource_id": "exp-1",
        },
    ):
        assert auth_module.validate_can_get_user_permission()


def test_validate_can_get_user_permission_admin_short_circuits(
    workspace_permission_setup, monkeypatch
):
    # Platform admins bypass the workspace check.
    store = workspace_permission_setup["store"]
    admin_username = "platform-admin"
    store.create_user(admin_username, "supersecurepassword", is_admin=True)

    monkeypatch.setattr(
        auth_module,
        "authenticate_request",
        lambda: SimpleNamespace(username=admin_username),
    )

    target = workspace_permission_setup["username"]
    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/users/permissions/get",
        method="GET",
        query_string={
            "username": target,
            "resource_type": "experiment",
            "resource_id": "exp-1",
        },
    ):
        assert auth_module.validate_can_get_user_permission()


def test_validate_can_get_user_permission_admin_probes_other_workspace(
    workspace_permission_setup, monkeypatch
):
    # Platform admins bypass the workspace check globally — even for resources in
    # workspaces they have no role in. Pins the is_admin short-circuit on the
    # cross-workspace path that ``_cross_workspace_probe_denied`` denies for non-admins.
    store = workspace_permission_setup["store"]
    admin_username = "platform-admin"
    store.create_user(admin_username, "supersecurepassword", is_admin=True)

    monkeypatch.setattr(
        auth_module,
        "authenticate_request",
        lambda: SimpleNamespace(username=admin_username),
    )

    target = workspace_permission_setup["username"]
    # team-b experiment — admin has no membership in team-b.
    auth_module._get_tracking_store()._experiment_workspaces["exp-team-b"] = "team-b"

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/users/permissions/get",
        method="GET",
        query_string={
            "username": target,
            "resource_type": "experiment",
            "resource_id": "exp-team-b",
        },
    ):
        assert auth_module.validate_can_get_user_permission()


def test_validate_can_get_user_permission_cross_user_requires_admin(
    workspace_permission_setup,
):
    # A non-admin requester without workspace MANAGE in the resource's workspace
    # cannot check another user's permissions. ``alice`` holds USE in team-a (not
    # MANAGE), so the workspace-admin gate fails even though the resource lives
    # in alice's workspace.
    store = workspace_permission_setup["store"]
    requester = workspace_permission_setup["username"]
    target = "bob"
    store.create_user(target, "supersecurepassword", is_admin=False)
    _set_workspace_permission(store, requester, USE.name)

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/users/permissions/get",
        method="GET",
        query_string={
            "username": target,
            "resource_type": "experiment",
            "resource_id": "exp-1",
        },
    ):
        assert not auth_module.validate_can_get_user_permission()


def test_validate_can_get_user_permission_wp_admin_scoped_to_resource_workspace(
    workspace_permission_setup,
):
    # A workspace admin in team-a can check another user's permissions on resources
    # in team-a. The scoping is by **resource workspace** (not by target-user
    # presence), so the target need not have a role in team-a for the gate to allow.
    store = workspace_permission_setup["store"]
    requester = workspace_permission_setup["username"]
    _set_workspace_permission(store, requester, MANAGE.name)

    target = "bob"
    store.create_user(target, "supersecurepassword", is_admin=False)

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/users/permissions/get",
        method="GET",
        query_string={
            "username": target,
            "resource_type": "experiment",
            "resource_id": "exp-1",
        },
    ):
        assert auth_module.validate_can_get_user_permission()


def test_validate_can_get_user_permission_cross_workspace_probe_denied(
    workspace_permission_setup,
):
    # Security gate: a workspace admin of team-a must NOT be able to probe a
    # target user's permissions on a resource in team-b. Closes the
    # cross-workspace information-disclosure gap.
    store = workspace_permission_setup["store"]
    requester = workspace_permission_setup["username"]
    _set_workspace_permission(store, requester, MANAGE.name)

    target = "bob"
    store.create_user(target, "supersecurepassword", is_admin=False)
    auth_module._get_tracking_store()._experiment_workspaces["exp-team-b"] = "team-b"

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/users/permissions/get",
        method="GET",
        query_string={
            "username": target,
            "resource_type": "experiment",
            "resource_id": "exp-team-b",
        },
    ):
        assert not auth_module.validate_can_get_user_permission()


def test_validate_can_get_user_permission_unknown_resource_denied(
    workspace_permission_setup,
):
    # If the resource can't be resolved to a workspace (e.g. it doesn't exist),
    # the gate must deny — otherwise a caller could probe across all workspaces
    # by using a non-existent ID.
    store = workspace_permission_setup["store"]
    requester = workspace_permission_setup["username"]
    _set_workspace_permission(store, requester, MANAGE.name)

    target = "bob"
    store.create_user(target, "supersecurepassword", is_admin=False)

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/users/permissions/get",
        method="GET",
        query_string={
            "username": target,
            "resource_type": "experiment",
            "resource_id": "exp-does-not-exist",
        },
    ):
        assert not auth_module.validate_can_get_user_permission()


def test_mcp_server_delete_grants_workspace_isolated(tmp_path, monkeypatch):
    """Deleting grants for an MCP server in one workspace must not affect
    same-named servers in another workspace.
    """
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")

    db_uri = f"sqlite:///{tmp_path / 'auth-iso.db'}"
    auth_store = SqlAlchemyStore()
    auth_store.init_db(db_uri)

    username = "alice"
    auth_store.create_user(username, "supersecurepassword", is_admin=False)
    server_name = "com.test/shared-name"

    # Grant MANAGE in both workspaces on the same server name.
    with workspace_context.WorkspaceContext("team-a"):
        auth_store.grant_user_permission(username, "mcp_server", server_name, MANAGE.name)

    with workspace_context.WorkspaceContext("team-b"):
        auth_store.grant_user_permission(username, "mcp_server", server_name, MANAGE.name)

    # Delete grants in team-a only.
    with workspace_context.WorkspaceContext("team-a"):
        auth_store.delete_grants_for_resource("mcp_server", server_name, workspace_scoped=True)

    # team-a grant is gone.
    with workspace_context.WorkspaceContext("team-a"):
        user = auth_store.get_user(username)
        perm = auth_store.get_role_permission_for_resource(
            user.id, "mcp_server", server_name, "team-a"
        )
        assert perm is None

    # team-b grant survives.
    with workspace_context.WorkspaceContext("team-b"):
        perm = auth_store.get_role_permission_for_resource(
            user.id, "mcp_server", server_name, "team-b"
        )
        assert perm is not None
        assert perm.name == MANAGE.name

    auth_store.engine.dispose()


def test_list_mcp_server_permissions_scoped_to_active_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")

    db_uri = f"sqlite:///{tmp_path / 'auth-list-ws.db'}"
    auth_store = SqlAlchemyStore()
    auth_store.init_db(db_uri)

    username = "alice"
    auth_store.create_user(username, "supersecurepassword", is_admin=False)

    with workspace_context.WorkspaceContext("team-a"):
        auth_store.grant_user_permission(username, "mcp_server", "com.test/a", MANAGE.name)
    with workspace_context.WorkspaceContext("team-b"):
        auth_store.grant_user_permission(username, "mcp_server", "com.test/b", READ.name)

    with workspace_context.WorkspaceContext("team-a"):
        perms = auth_store.list_mcp_server_permissions(username)
        assert sorted(p.name for p in perms) == ["com.test/a"]

    with workspace_context.WorkspaceContext("team-b"):
        perms = auth_store.list_mcp_server_permissions(username)
        assert sorted(p.name for p in perms) == ["com.test/b"]

    auth_store.engine.dispose()


# =============================================================================
# The review-queue LIST filter must honour the queue tier the detail gate uses
# (findings 2 + 7, tracker item 2c).
# =============================================================================


def _list_review_queues_response(rows):
    import json as _json

    from mlflow.protos.review_queues_pb2 import ListReviewQueues
    from mlflow.utils.proto_json_utils import message_to_json, parse_dict

    message = ListReviewQueues.Response()
    parse_dict({"review_queues": rows}, message)
    return SimpleNamespace(json=_json.loads(message_to_json(message)), data=None)


def _run_list_filter(rows):
    import json as _json

    from mlflow.protos.review_queues_pb2 import ListReviewQueues
    from mlflow.utils.proto_json_utils import parse_dict

    resp = _list_review_queues_response(rows)
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/review-queues/list", query_string={"experiment_id": "exp-1"}
    ):
        auth_module.filter_list_review_queues(resp)
    if resp.data is None:
        # The filter returned without narrowing: every row stayed visible.
        return [q["queue_id"] for q in rows]
    out = ListReviewQueues.Response()
    parse_dict(_json.loads(resp.data), out)
    return [q.queue_id for q in out.review_queues]


def _run_trace_artifact(monkeypatch, experiment_id, request_id="tr-1"):
    with auth_module.app.test_request_context(
        "/ajax-api/2.0/mlflow/get-trace-artifact", query_string={"request_id": request_id}
    ):
        monkeypatch.setattr(
            auth_module._get_tracking_store(),
            "get_trace_info",
            lambda _tid: SimpleNamespace(experiment_id=experiment_id),
            raising=False,
        )
        return auth_module.validate_can_read_trace_artifact()


def test_trace_artifact_download_honors_a_trace_deny(workspace_permission_setup, monkeypatch):
    """The artifact IS the trace payload. Resolving the experiment alone let (trace, *, DENY)
    block GetTrace, batch, search and tags while this route still served the spans.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "*", EDIT.name),
            ("trace", "*", DENY.name),
        ],
    )

    assert _run_trace_artifact(monkeypatch, "exp-1") is False


def test_trace_artifact_download_inherits_the_experiment(workspace_permission_setup, monkeypatch):
    """No trace grant: the experiment tier still decides, as it always did."""
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("experiment", "*", READ.name)])

    assert _run_trace_artifact(monkeypatch, "exp-1") is True


def _run_scorer_point_route(validator_name, experiment_id, scorer_name):
    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/scorers/get",
        query_string={"experiment_id": experiment_id, "name": scorer_name},
    ):
        return getattr(auth_module, validator_name)()


def test_scorer_point_routes_honor_a_scorer_version_deny(workspace_permission_setup):
    """ListScorers withholds rows on a version DENY, but GetScorer / ListScorerVersions return the
    same serialized_scorer while checking only the parent scorer. DeleteScorer likewise.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "*", READ.name),
            ("scorer", "*", MANAGE.name),
            ("scorer_version", "*", DENY.name),
        ],
    )

    assert _run_scorer_point_route("validate_can_read_scorer", "1", "s1") is False
    assert _run_scorer_point_route("validate_can_delete_scorer", "1", "s1") is False
    # Update is not a disclosure surface and keeps the scorer tier alone.
    assert _run_scorer_point_route("validate_can_update_scorer", "1", "s1") is True


def test_scorer_point_routes_unchanged_without_a_version_grant(workspace_permission_setup):
    """No version grant: the veto passes and the scorer tier decides, exactly as before."""
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "*", READ.name),
            ("scorer", "*", MANAGE.name),
        ],
    )

    assert _run_scorer_point_route("validate_can_read_scorer", "1", "s1") is True
    assert _run_scorer_point_route("validate_can_delete_scorer", "1", "s1") is True


def _run_queue_by_name(monkeypatch, experiment_id, queue_users):
    queue = SimpleNamespace(
        experiment_id=experiment_id, users=list(queue_users), created_by=None, queue_type="CUSTOM"
    )
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/review-queues/get-by-name",
        query_string={"experiment_id": experiment_id, "name": "Q"},
    ):
        # Extend the fixture's tracking store rather than replacing it: the workspace
        # resolver reads get_experiment off the same object.
        monkeypatch.setattr(
            auth_module._get_tracking_store(),
            "get_review_queue_by_name",
            lambda _exp, name: queue,
            raising=False,
        )
        return auth_module.validate_can_view_review_queue_by_name()


def test_queue_by_name_honors_a_queue_deny(workspace_permission_setup, monkeypatch):
    """GetReviewQueue 403s on a queue DENY; GetReviewQueueByName resolved the EXPERIMENT tier,
    so the same queue opened by name was allowed. Assigned membership made it reachable.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "*", EDIT.name),
            ("review_queue", "*", DENY.name),
        ],
    )

    assert _run_queue_by_name(monkeypatch, "exp-1", [username]) is False


def test_queue_by_name_inherits_the_experiment_without_a_queue_grant(
    workspace_permission_setup, monkeypatch
):
    """No queue grant: the experiment tier still decides, so membership continues to open it."""
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("experiment", "*", EDIT.name)])

    assert _run_queue_by_name(monkeypatch, "exp-1", [username]) is True


def _run_get_or_create_queue(experiment_id):
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/review-queues/get-or-create-user-queue",
        query_string={"experiment_id": experiment_id},
    ):
        return auth_module.validate_can_get_or_create_user_queue()


def test_get_or_create_user_queue_honors_a_queue_deny(workspace_permission_setup):
    """Get-or-create was experiment UPDATE alone -- the one create path with no child veto, so a
    queue DENY could still be made to create and return a personal queue.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "*", EDIT.name),
            ("review_queue", "*", DENY.name),
        ],
    )

    assert _run_get_or_create_queue("exp-1") is False


def test_get_or_create_user_queue_allowed_without_a_queue_grant(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("experiment", "*", EDIT.name)])

    assert _run_get_or_create_queue("exp-1") is True


def test_review_queue_list_filter_honors_a_queue_deny(workspace_permission_setup):
    """``(review_queue, *, DENY)`` 403s the detail gate, but the list filter -- still on the
    experiment tier -- returned every row, leaking name, assigned users and created_by for
    queues the caller cannot open.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "*", EDIT.name),
            ("review_queue", "*", DENY.name),
        ],
    )

    rows = [{"queue_id": "q1", "users": [username]}, {"queue_id": "q2", "users": ["bob"]}]
    assert _run_list_filter(rows) == []


def test_review_queue_list_filter_unchanged_without_a_queue_grant(workspace_permission_setup):
    """No regression: absent a queue grant the list tier stays exactly as broad as master's --
    an experiment EDITor sees every row, including queues they neither own nor belong to.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("experiment", "*", EDIT.name)])

    rows = [{"queue_id": "q1", "users": ["bob"]}, {"queue_id": "q2", "users": ["carol"]}]
    assert _run_list_filter(rows) == ["q1", "q2"]


def test_review_queue_list_filter_read_only_still_sees_only_assigned(workspace_permission_setup):
    """The other half of no-regression: a READ-only caller keeps seeing only their own rows."""
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("experiment", "*", READ.name)])

    rows = [{"queue_id": "q1", "users": [username]}, {"queue_id": "q2", "users": ["bob"]}]
    assert _run_list_filter(rows) == ["q1"]


def test_review_queue_list_filter_honors_a_queue_manage_grant(workspace_permission_setup):
    """The inverse gap: a queue MANAGE grant lets the detail gate open any queue, so the list
    must stop hiding them behind a merely-READ experiment tier.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "*", READ.name),
            ("review_queue", "*", MANAGE.name),
        ],
    )

    rows = [{"queue_id": "q1", "users": ["bob"]}, {"queue_id": "q2", "users": ["carol"]}]
    assert _run_list_filter(rows) == ["q1", "q2"]


def _run_scorer_list_filter(rows):
    import json as _json

    from mlflow.protos.service_pb2 import ListScorers
    from mlflow.utils.proto_json_utils import message_to_json, parse_dict

    message = ListScorers.Response()
    parse_dict({"scorers": rows}, message)
    resp = SimpleNamespace(json=_json.loads(message_to_json(message)), data=None)
    with auth_module.app.test_request_context("/api/2.0/mlflow/scorers/list"):
        auth_module.filter_list_scorers(resp)
    if resp.data is None:
        return [r["scorer_name"] for r in rows]
    out = ListScorers.Response()
    parse_dict(_json.loads(resp.data), out)
    return [s.scorer_name for s in out.scorers]


def test_scorer_list_filter_honors_a_scorer_deny(workspace_permission_setup):
    """The scorer tier is per-id grain, so a DENY can name ONE scorer. It must drop that row and
    leave its sibling, which the pre-`e5b3004d4` predicate could not do -- it discarded DENY.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "*", READ.name),
            ("scorer", "*", READ.name),
            ("scorer", "1/blocked", DENY.name),
        ],
    )

    # Scorer.experiment_id is an int32 in the proto, so ids here are numeric. The predicate
    # matches grant keys and never fetches an experiment, so any id works.
    rows = [
        {"experiment_id": 1, "scorer_name": "blocked"},
        {"experiment_id": 1, "scorer_name": "allowed"},
    ]
    assert _run_scorer_list_filter(rows) == ["allowed"]


def test_scorer_list_filter_honors_a_scorer_version_deny(workspace_permission_setup):
    """Every listed row is a ScorerVersion, so a version-tier DENY empties the list.

    The version tier is wildcard-only, so this is one constant decision for the whole response --
    it cannot name a single row. Positive grants on the tiers above it do not lift it.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "*", READ.name),
            ("scorer", "*", READ.name),
            ("scorer_version", "*", DENY.name),
        ],
    )

    rows = [
        {"experiment_id": 1, "scorer_name": "s1"},
        {"experiment_id": 2, "scorer_name": "s2"},
    ]
    assert _run_scorer_list_filter(rows) == []


def test_scorer_list_filter_keeps_rows_without_a_scorer_version_grant(workspace_permission_setup):
    """No version grant: the veto passes, so the tiers above decide. The compatibility case."""
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "*", READ.name),
            ("scorer", "*", READ.name),
        ],
    )

    rows = [{"experiment_id": 1, "scorer_name": "s1"}]
    assert _run_scorer_list_filter(rows) == ["s1"]


def test_scorer_list_filter_honors_an_experiment_deny(workspace_permission_setup):
    """The other tier: denying the experiment drops its scorers even with a scorer grant."""
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "*", READ.name),
            ("experiment", "1", DENY.name),
            ("scorer", "*", READ.name),
        ],
    )

    rows = [
        {"experiment_id": 1, "scorer_name": "s1"},
        {"experiment_id": 2, "scorer_name": "s2"},
    ]
    assert _run_scorer_list_filter(rows) == ["s2"]


# ==========================================================================================
# Trace assessment redaction (design doc 5h; review finding 9a)
# ==========================================================================================


def _run_multi_trace_redaction(proto_name, handler_name, payload):
    import json as _json

    from mlflow.utils.proto_json_utils import message_to_json, parse_dict

    proto = getattr(__import__("mlflow.protos.service_pb2", fromlist=[proto_name]), proto_name)
    message = proto.Response()
    parse_dict(payload, message)
    resp = SimpleNamespace(json=_json.loads(message_to_json(message)), data=None)
    with auth_module.app.test_request_context("/api/3.0/mlflow/traces"):
        getattr(auth_module, handler_name)(resp)
    out = proto.Response()
    parse_dict(_json.loads(resp.data) if resp.data is not None else resp.json, out)
    return out


def _guardrail_config_payload(experiment_id=1, scorer_name="safety"):
    return {
        "configs": [
            {
                "endpoint_id": "endpoint-1",
                "guardrail_id": "g-1",
                "guardrail": {
                    "guardrail_id": "g-1",
                    "name": "safety-guard",
                    # ScorerVersion.experiment_id is int32 in this proto, so the gate keys on
                    # its stringified form -- experiment "1" in the fixture's workspace map.
                    "scorer": {
                        "experiment_id": experiment_id,
                        "scorer_name": scorer_name,
                        "scorer_version": 2,
                        "serialized_scorer": "SECRET-SCORER-BODY",
                    },
                },
            }
        ]
    }


def _run_guardrail_redaction(payload):
    import json as _json

    from mlflow.protos.service_pb2 import ListEndpointGuardrailConfigs
    from mlflow.utils.proto_json_utils import message_to_json, parse_dict

    message = ListEndpointGuardrailConfigs.Response()
    parse_dict(payload, message)
    resp = SimpleNamespace(json=_json.loads(message_to_json(message)), data=None)
    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/gateway/guardrails/list-for-endpoint",
        query_string={"endpoint_id": "endpoint-1"},
    ):
        auth_module.redact_list_guardrail_config_scorers(resp)
    out = ListEndpointGuardrailConfigs.Response()
    parse_dict(_json.loads(resp.data) if resp.data is not None else resp.json, out)
    return out


def test_guardrail_configs_withhold_a_denied_scorer(workspace_permission_setup):
    """Guardrail.scorer is a full ScorerVersion, serialized_scorer included, behind an
    endpoint-only gate. The scorer is a PASSENGER -- the caller asked for the endpoint's guardrail
    configs -- so the row stays and only the scorer is withheld.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "*", EDIT.name),
            ("scorer", "*", DENY.name),
        ],
    )

    out = _run_guardrail_redaction(_guardrail_config_payload())

    assert len(out.configs) == 1, "the config row itself must survive"
    assert out.configs[0].guardrail.guardrail_id == "g-1"
    assert out.configs[0].guardrail.name == "safety-guard"
    assert not out.configs[0].guardrail.HasField("scorer")


def test_guardrail_configs_withhold_on_scorer_version_deny(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "*", EDIT.name),
            ("scorer_version", "*", DENY.name),
        ],
    )

    out = _run_guardrail_redaction(_guardrail_config_payload())

    assert not out.configs[0].guardrail.HasField("scorer")


def test_guardrail_configs_keep_a_permitted_scorer(workspace_permission_setup):
    """Without a scorer denial nothing is withheld, so the endpoint UI is unchanged."""
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("experiment", "*", EDIT.name)])

    out = _run_guardrail_redaction(_guardrail_config_payload())

    assert out.configs[0].guardrail.scorer.serialized_scorer == "SECRET-SCORER-BODY"


def _run_add_guardrail(monkeypatch, scorer_name="safety", experiment_id=1):
    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/gateway/guardrails/add-to-endpoint",
        method="POST",
        json={"endpoint_id": "endpoint-1", "guardrail_id": "g-1"},
    ):
        # Patch onto the real store instance: replacing _get_tracking_store wholesale breaks the
        # workspace resolver, which reads other methods off the same object.
        monkeypatch.setattr(
            auth_module._get_tracking_store(),
            "get_gateway_guardrail",
            lambda guardrail_id: SimpleNamespace(
                guardrail_id=guardrail_id,
                scorer=SimpleNamespace(experiment_id=experiment_id, scorer_name=scorer_name),
            ),
            raising=False,
        )
        return auth_module.validate_can_add_guardrail_to_gateway_endpoint()


def test_add_guardrail_vetoes_a_denied_scorer(workspace_permission_setup, monkeypatch):
    """Attaching a guardrail puts its scorer on the endpoint's traffic and echoes the scorer back,
    so the scorer vetoes -- the same treatment validate_can_invoke_scorer gives a scorer it runs.
    The endpoint tier stays the positive gate.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("gateway_endpoint", "*", EDIT.name),
            ("scorer", "*", DENY.name),
        ],
    )

    assert _run_add_guardrail(monkeypatch) is False


def test_add_guardrail_allowed_without_a_scorer_denial(workspace_permission_setup, monkeypatch):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("gateway_endpoint", "*", EDIT.name)])

    assert _run_add_guardrail(monkeypatch) is True


def test_add_guardrail_denies_an_unresolvable_guardrail(workspace_permission_setup, monkeypatch):
    """A guardrail id that does not resolve denies uniformly, so the response is not an oracle
    for which guardrail ids exist -- the same reasoning _run_requirement applies to runs.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("gateway_endpoint", "*", EDIT.name)])

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/gateway/guardrails/add-to-endpoint",
        method="POST",
        json={"endpoint_id": "endpoint-1", "guardrail_id": "missing"},
    ):

        def _raise(guardrail_id):
            raise MlflowException("not found", error_code=RESOURCE_DOES_NOT_EXIST)

        monkeypatch.setattr(
            auth_module._get_tracking_store(), "get_gateway_guardrail", _raise, raising=False
        )
        assert auth_module.validate_can_add_guardrail_to_gateway_endpoint() is False


def _info_row(experiment_id, trace_id, names):
    return {
        "trace_id": trace_id,
        "trace_location": {"mlflow_experiment": {"experiment_id": experiment_id}},
        "assessments": [{"assessment_name": n} for n in names],
    }


def test_batch_trace_infos_redaction_honors_assessment_deny(workspace_permission_setup):
    """BatchGetTraceInfos returns trace_infos[] of TraceInfoV3 directly, assessments always set."""
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "*", EDIT.name),
            ("assessment", "*", DENY.name),
        ],
    )

    out = _run_multi_trace_redaction(
        "BatchGetTraceInfos",
        "redact_batch_trace_info_assessments",
        {"trace_infos": [_info_row("exp-1", "t1", ["a1"]), _info_row("exp-2", "t2", ["a2"])]},
    )
    assert [len(i.assessments) for i in out.trace_infos] == [0, 0]
    # The rows themselves survive: only the assessments are withheld.
    assert [i.trace_id for i in out.trace_infos] == ["t1", "t2"]


def test_search_traces_v3_redaction_inherits_the_experiment(workspace_permission_setup):
    """No assessment grant: the experiment governs, so assessments stay. The compatibility case."""
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("experiment", "*", READ.name)])

    out = _run_multi_trace_redaction(
        "SearchTracesV3",
        "redact_search_traces_v3_assessments",
        {"traces": [_info_row("exp-1", "t1", ["a1", "a2"])]},
    )
    assert [a.assessment_name for a in out.traces[0].assessments] == ["a1", "a2"]


def test_batch_get_traces_redaction_reaches_nested_trace_info(workspace_permission_setup):
    """BatchGetTraces wraps each TraceInfoV3 in a Trace, so the assessments sit one level down."""
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "*", EDIT.name),
            ("assessment", "*", DENY.name),
        ],
    )

    out = _run_multi_trace_redaction(
        "BatchGetTraces",
        "redact_batch_trace_assessments",
        {"traces": [{"trace_info": _info_row("exp-1", "t1", ["a1"])}]},
    )
    assert len(out.traces[0].trace_info.assessments) == 0


def _run_submit_optimization(source_prompt_uri):
    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/prompt-optimization-jobs/create",
        json={
            "experiment_id": "exp-1",
            "source_prompt_uri": source_prompt_uri,
            "config": {"scorers": []},
        },
    ):
        return auth_module.validate_can_create_prompt_optimization_job()


@pytest.mark.parametrize("uri", ["prompts:/other/3", "prompts:/other@prod", "prompts:/other"])
def test_optimization_job_honors_a_prompt_deny(workspace_permission_setup, uri):
    """The identity-less worker loads source_prompt_uri and registers a NEW version under it, so
    an experiment editor could append a version to a prompt they cannot update. All three URI
    spellings resolve to the same prompt name.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "*", EDIT.name),
            ("prompt", "other", DENY.name),
        ],
    )

    assert _run_submit_optimization(uri) is False


def test_optimization_job_honors_a_prompt_version_deny(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "*", EDIT.name),
            ("prompt_version", "*", DENY.name),
        ],
    )

    assert _run_submit_optimization("prompts:/other/3") is False


def test_optimization_job_unchanged_without_prompt_grants(workspace_permission_setup):
    """No prompt grants, and a request naming no prompt at all: both behave as before."""
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("experiment", "*", EDIT.name)])

    assert _run_submit_optimization("prompts:/other/3") is True
    assert _run_submit_optimization("") is True


def _run_get_assessment(monkeypatch, experiment_id):
    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/traces/t1/assessments/a1",
        query_string={"trace_id": "t1", "assessment_id": "a1"},
    ):
        monkeypatch.setattr(
            auth_module._get_tracking_store(),
            "get_trace_info",
            lambda _tid: SimpleNamespace(experiment_id=experiment_id),
            raising=False,
        )
        return auth_module.validate_can_get_assessment()


def test_get_assessment_denies_rather_than_redacts(workspace_permission_setup, monkeypatch):
    """The assessment is the SUBJECT of this route, so a DENY refuses it outright."""
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "*", EDIT.name),
            ("assessment", "*", DENY.name),
        ],
    )

    assert _run_get_assessment(monkeypatch, "exp-1") is False


def test_get_assessment_inherits_the_experiment(workspace_permission_setup, monkeypatch):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("experiment", "*", READ.name)])

    assert _run_get_assessment(monkeypatch, "exp-1") is True


def _run_model_version_artifact(name="model-xyz", version="3"):
    with auth_module.app.test_request_context(
        "/model-versions/get-artifact",
        query_string={"name": name, "version": version, "path": "MLmodel"},
    ):
        return auth_module.validate_can_read_model_version_artifact()


def test_model_version_artifact_honors_a_version_deny(workspace_permission_setup):
    """The artifact IS the version's content -- the handler streams
    get_model_version_download_uri(name, version) -- yet the route whose entire subject is a
    version resolved only the registered model.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("registered_model", "*", READ.name),
            ("registered_model_version", "*", DENY.name),
        ],
    )

    assert _run_model_version_artifact() is False


def test_model_version_artifact_unchanged_without_a_version_grant(workspace_permission_setup):
    """No version grant: the registered model tier decides, exactly as before."""
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("registered_model", "*", READ.name)])

    assert _run_model_version_artifact() is True


def _run_search_traces(filter_string):
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/traces",
        query_string={"experiment_ids": "exp-1", "filter": filter_string},
    ):
        return auth_module.validate_can_search_traces()


def _run_filter_correlation(filter1, camel_case=False):
    body = (
        {"experimentIds": ["exp-1"], "filterString1": filter1, "filterString2": "name = 'x'"}
        if camel_case
        else {
            "experiment_ids": ["exp-1"],
            "filter_string1": filter1,
            "filter_string2": "name = 'x'",
        }
    )
    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/traces/calculate-filter-correlation", json=body
    ):
        return auth_module.validate_can_read_traces_by_experiment_ids()


def _deny_assessments(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "*", EDIT.name),
            ("assessment", "*", DENY.name),
        ],
    )


def test_search_traces_refuses_an_assessment_backed_filter(workspace_permission_setup):
    """Redaction cannot cover a filter: which rows MATCH is the disclosure, so stripping
    assessments from the returned rows still answers "how many traces scored 'no'".
    """
    _deny_assessments(workspace_permission_setup)

    assert _run_search_traces("feedback.safety = 'no'") is False
    assert _run_search_traces("expectation.expected = 'x'") is False
    # Numeric comparators reach the same table.
    assert _run_search_traces("feedback.score > 0.5") is False


def test_search_traces_unaffected_by_non_assessment_filters(workspace_permission_setup):
    """The gate must land only on filters that actually reach the assessments table, or an
    assessment DENY would cost the caller trace search entirely -- the case redaction exists for.
    """
    _deny_assessments(workspace_permission_setup)

    assert _run_search_traces("") is True
    assert _run_search_traces("status = 'OK'") is True
    assert _run_search_traces("name = 'x' AND timestamp_ms > 0") is True
    assert _run_search_traces("tags.foo = 'bar'") is True
    # issue.id selects on the assessment NAME, not a value, and `issue` is not a type we govern.
    assert _run_search_traces("issue.id = 'i1'") is True
    # `prompt` maps to the linked-prompts tag: an unvalidated author assertion, not prompt
    # content, so it stays ungated exactly as on master.
    assert _run_search_traces("prompt = 'prompts:/p/1'") is True


def test_search_traces_fails_closed_on_an_unparsable_filter(workspace_permission_setup):
    """Only reachable for filters the handler would 400 anyway; the cost is 403 instead."""
    _deny_assessments(workspace_permission_setup)

    assert _run_search_traces("(((") is False


def test_search_traces_assessment_filter_allowed_without_a_grant(workspace_permission_setup):
    """No assessment grant: the experiment governs, so nothing master allowed is newly denied."""
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("experiment", "*", READ.name)])

    assert _run_search_traces("feedback.safety = 'no'") is True


@pytest.mark.parametrize("tier", ["assessment", "run", "logged_model"])
def test_filter_correlation_gates_camel_case_filters(workspace_permission_setup, tier):
    """The handler parses this request with ParseDict, which accepts lowerCamelCase aliases too, so
    a gate reading raw snake_case keys never sees a filter spelled `filterString1` -- while the
    handler executes it. Reading the same proto the handler reads closes every spelling at once.
    """
    _deny_tier(workspace_permission_setup, tier)
    selector = {
        "assessment": "feedback.safety = 'no'",
        "run": "run_id = 'run-1'",
        "logged_model": "metadata.`mlflow.modelId` = 'model-1'",
    }[tier]

    assert _run_filter_correlation(selector) is False
    assert _run_filter_correlation(selector, camel_case=True) is False
    # An unrelated filter stays allowed in both spellings.
    assert _run_filter_correlation("status = 'OK'") is True
    assert _run_filter_correlation("status = 'OK'", camel_case=True) is True


def _run_search_traces_v3(locations, filter_string=""):
    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/traces/search",
        method="POST",
        json={"locations": locations, "filter": filter_string},
    ):
        return auth_module.validate_can_search_traces_v3()


def _snake_location(experiment_id):
    return {"mlflow_experiment": {"experiment_id": experiment_id}}


def _camel_location(experiment_id):
    return {"mlflowExperiment": {"experimentId": experiment_id}}


def test_search_traces_v3_sees_mixed_alias_locations(workspace_permission_setup):
    """The handler parses locations with ParseDict, which accepts lowerCamelCase PER LIST ELEMENT.
    A body mixing one permitted snake_case location with one denied camelCase location therefore
    hid the second experiment from a raw-JSON walk while the handler searched both.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "exp-1", READ.name),
            ("experiment", "exp-2", DENY.name),
        ],
    )

    assert _run_search_traces_v3([_snake_location("exp-1")]) is True
    assert _run_search_traces_v3([_snake_location("exp-2")]) is False
    # The denied experiment must not become invisible by changing its spelling, in either order.
    assert _run_search_traces_v3([_snake_location("exp-1"), _camel_location("exp-2")]) is False
    assert _run_search_traces_v3([_camel_location("exp-2"), _snake_location("exp-1")]) is False
    # A wholly camelCase permitted request is honoured rather than refused.
    assert _run_search_traces_v3([_camel_location("exp-1")]) is True


def test_start_trace_v3_accepts_camel_case_locations(workspace_permission_setup):
    """A structural match on raw JSON refused the lowerCamelCase spelling the handler accepts."""
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("experiment", "*", EDIT.name)])

    def _run(location):
        with auth_module.app.test_request_context(
            "/api/3.0/mlflow/traces",
            method="POST",
            json={"trace": {"trace_info": {"trace_location": location}}},
        ):
            return auth_module.validate_can_start_trace_v3()

    assert _run(_snake_location("exp-1")) is True
    assert _run(_camel_location("exp-1")) is True
    # A body naming no experiment still denies.
    assert _run({}) is False


_PROMPT_TAGS = [{"key": "mlflow.prompt.is_prompt", "value": "true"}]


@pytest.mark.parametrize(
    ("validator", "tier", "path", "body"),
    [
        (
            "validate_can_create_experiment",
            "experiment",
            "/api/2.0/mlflow/experiments/create",
            {"name": "x"},
        ),
        (
            "validate_can_create_registered_model",
            "registered_model",
            "/api/2.0/mlflow/registered-models/create",
            {"name": "x"},
        ),
        # The route is shared; the request's own tags decide which family is created.
        (
            "validate_can_create_registered_model",
            "prompt",
            "/api/2.0/mlflow/registered-models/create",
            {"name": "x", "tags": _PROMPT_TAGS},
        ),
        (
            "validate_can_create_gateway_secret",
            "gateway_secret",
            "/api/3.0/mlflow/gateway/secrets/create",
            {"name": "x"},
        ),
    ],
)
def test_workspace_creates_honor_a_created_type_deny(
    workspace_permission_setup, validator, tier, path, body
):
    """§5d gives the created type a veto. The child creates had it via
    `_authorize_create_in_experiment`; the workspace-scoped creates did not, so a DENY holder kept
    creating resources while being refused every other operation on one.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [(tier, "*", DENY.name)])

    with auth_module.app.test_request_context(path, method="POST", json=body):
        assert getattr(auth_module, validator)() is False


@pytest.mark.parametrize(
    ("denied_tier", "body"),
    [
        # A prompt DENY must not block creating a plain registered model...
        ("prompt", {"name": "x"}),
        # ...nor a registered_model DENY block creating a prompt.
        ("registered_model", {"name": "x", "tags": _PROMPT_TAGS}),
    ],
)
def test_registered_model_create_vetoes_only_the_family_it_creates(
    workspace_permission_setup, denied_tier, body
):
    """CREATE is the one shared route where the body IS the truth: these tags are the tags the
    handler persists, so there is nothing for them to contradict. Vetoing both families would
    refuse a caller denied only the family they are not creating.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [(denied_tier, "*", DENY.name)])
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/registered-models/create", method="POST", json=body
    ):
        assert auth_module.validate_can_create_registered_model() is True


def test_workspace_creates_allowed_without_a_deny(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/experiments/create", method="POST", json={"name": "x"}
    ):
        assert auth_module.validate_can_create_experiment() is True


@pytest.mark.parametrize(
    ("handler", "path"),
    [
        ("redact_get_registered_model_versions", "/api/2.0/mlflow/registered-models/get"),
        ("redact_update_registered_model_versions", "/api/2.0/mlflow/registered-models/update"),
    ],
)
def test_single_model_responses_redact_embedded_versions(
    workspace_permission_setup, monkeypatch, handler, path
):
    """Get, Update and Rename all return the model via to_mlflow_entity(), which populates
    latest_versions -- so each is a route to version data, not just Get.
    """
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("registered_model", "*", READ.name),
            ("registered_model_version", "*", DENY.name),
        ],
    )
    payload = json.dumps({
        "registered_model": {
            "name": "model-xyz",
            "tags": [],
            "latest_versions": [{"name": "model-xyz"}],
        }
    })
    flask_resp = Response(payload, mimetype="application/json")
    with auth_module.app.test_request_context(path, method="POST", json={"name": "model-xyz"}):
        getattr(auth_module, handler)(flask_resp)
    out = json.loads(flask_resp.get_data(as_text=True))
    # message_to_json omits an emptied repeated field rather than serialising [].
    assert out["registered_model"].get("latest_versions", []) == []
    assert out["registered_model"]["name"] == "model-xyz"


def test_delete_alias_requires_version_read_and_falls_back_to_the_model(
    workspace_permission_setup, monkeypatch
):
    """Removing an alias un-publishes whatever version it pointed at, so it carries the same version
    requirement as setting one -- even though the request names no version field.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)

    def _delete_alias():
        with auth_module.app.test_request_context(
            "/api/2.0/mlflow/registered-models/alias",
            method="DELETE",
            json={"name": "model-xyz", "alias": "champion"},
        ):
            return auth_module.validate_can_delete_model_or_prompt_version_alias()

    # Parent MANAGE (delete needs it), no version grant -> falls back to the parent as master did.
    _grant(store, username, "team-a", [("registered_model", "model-xyz", MANAGE.name)])
    assert _delete_alias() is True

    _grant(
        store,
        username,
        "team-a",
        [
            ("registered_model", "model-xyz", MANAGE.name),
            ("registered_model_version", "*", DENY.name),
        ],
    )
    assert _delete_alias() is False


def test_set_alias_requires_version_read_and_falls_back_to_the_model(
    workspace_permission_setup, monkeypatch
):
    """The alias is what publishes a version under a friendly name, so a version DENY blocks it --
    but with no version grant the parent governs, exactly as master did.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)

    def _set_alias():
        with auth_module.app.test_request_context(
            "/api/2.0/mlflow/registered-models/alias",
            method="POST",
            json={"name": "model-xyz", "alias": "champion", "version": "3"},
        ):
            return auth_module.validate_can_set_model_or_prompt_version_alias()

    # Parent EDIT, no version grant -> falls back to the parent, which master already required.
    _grant(store, username, "team-a", [("registered_model", "model-xyz", EDIT.name)])
    assert _set_alias() is True

    # A version DENY blocks publishing even though the parent still permits the alias map update.
    _grant(
        store,
        username,
        "team-a",
        [
            ("registered_model", "model-xyz", EDIT.name),
            ("registered_model_version", "*", DENY.name),
        ],
    )
    assert _set_alias() is False


def test_create_model_version_gates_a_model_id_hidden_in_the_source_uri(
    workspace_permission_setup, monkeypatch
):
    """A `models:/m-<id>` source is dereferenced by the store to derive run_id, so it is a third way
    to bind a version to another user's logged model -- naming neither run_id nor model_id.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("registered_model", "model-xyz", EDIT.name)])
    seen = []

    def _fake_logged_model_read(model_id, action):
        seen.append((model_id, action))
        return False

    monkeypatch.setattr(auth_module, "_authorize_logged_model_id", _fake_logged_model_read)

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/model-versions/create",
        method="POST",
        json={"name": "model-xyz", "source": "models:/m-someone-elses"},
    ):
        allowed = auth_module.validate_can_create_model_version()

    assert allowed is False
    assert seen == [("m-someone-elses", "read")]


def test_create_model_version_ignores_a_registry_source_uri(
    workspace_permission_setup, monkeypatch
):
    """`models:/<name>/<version>` names a registry entry, not a logged model, so it dereferences
    nothing and must not be pushed through the logged-model check.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("registered_model", "model-xyz", EDIT.name)])
    monkeypatch.setattr(
        auth_module,
        "_authorize_logged_model_id",
        lambda *a: pytest.fail("a registry source must not be treated as a logged model"),
    )
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/model-versions/create",
        method="POST",
        json={"name": "model-xyz", "source": "models:/other-model/1"},
    ):
        assert auth_module.validate_can_create_model_version() is True


def _version_payload():
    return {
        "model_version": {
            "name": "model-xyz",
            "version": "3",
            "run_id": "r-1",
            "run_link": "http://host/#/experiments/1/runs/r-1",
            "model_id": "m-1",
            "model_metrics": [{"key": "acc", "value": 0.9}],
            "source": "models:/m-1",
        }
    }


def test_model_version_withholds_run_content_on_a_run_deny(workspace_permission_setup, monkeypatch):
    """A ModelVersion names the run that produced it (run_id, and run_link which is a URL to it)."""
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("run", "*", DENY.name)])

    flask_resp = Response(json.dumps(_version_payload()), mimetype="application/json")
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/model-versions/get",
        method="GET",
        query_string={"name": "model-xyz", "version": "3"},
    ):
        auth_module.redact_model_version_siblings(flask_resp)
    version = json.loads(flask_resp.get_data(as_text=True))["model_version"]

    assert "run_id" not in version
    assert "run_link" not in version
    # The logged-model tier is separate and untouched.
    assert version["model_id"] == "m-1"
    # `source` is the version's OWN artifact location, not a sibling's, and is gated at create.
    assert version["source"] == "models:/m-1"


def test_model_version_withholds_model_content_on_a_logged_model_deny(
    workspace_permission_setup, monkeypatch
):
    """model_params / model_metrics are the logged model's own values surfacing on the version."""
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("logged_model", "*", DENY.name)])

    flask_resp = Response(json.dumps(_version_payload()), mimetype="application/json")
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/model-versions/get",
        method="GET",
        query_string={"name": "model-xyz", "version": "3"},
    ):
        auth_module.redact_model_version_siblings(flask_resp)
    version = json.loads(flask_resp.get_data(as_text=True))["model_version"]

    assert "model_id" not in version
    assert "model_metrics" not in version
    assert version["run_id"] == "r-1"


@pytest.mark.parametrize(
    ("tier", "gone", "kept"),
    [("run", ("run_id", "run_link"), "model_id"), ("logged_model", ("model_id",), "run_id")],
    ids=["run-deny", "logged_model-deny"],
)
def test_search_registered_models_latest_versions_lose_denied_siblings(
    workspace_permission_setup, monkeypatch, tier, gone, kept
):
    """The search filter withheld whole latest_versions rows on a version DENY but never made the
    second pass the point routes make, so a surviving row still carried the denied sibling's ids.
    """
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store, username, "team-a", [("registered_model", "*", MANAGE.name), (tier, "*", DENY.name)]
    )

    payload = {
        "registered_models": [
            {
                "name": "model-xyz",
                "latest_versions": [
                    {
                        "name": "model-xyz",
                        "version": "3",
                        "run_id": "r-1",
                        "run_link": "http://x/r-1",
                        "model_id": "m-1",
                    }
                ],
            }
        ]
    }
    flask_resp = Response(json.dumps(payload), mimetype="application/json")
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/registered-models/search", method="GET"
    ):
        with workspace_context.WorkspaceContext("team-a"):
            auth_module.filter_search_registered_models(flask_resp)
    body = json.loads(flask_resp.get_data(as_text=True))
    version = body["registered_models"][0]["latest_versions"][0]

    assert version["version"] == "3"
    for field in gone:
        assert field not in version
    # Each tier owns its own fields, so one DENY must not strip the other's.
    assert version[kept] is not None


def test_registered_model_latest_versions_also_lose_denied_siblings(
    workspace_permission_setup, monkeypatch
):
    """A latest_versions row the caller may read still carried the denied run's id."""
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("run", "*", DENY.name)])
    payload = {
        "registered_model": {
            "name": "model-xyz",
            "latest_versions": [{"name": "model-xyz", "version": "3", "run_id": "r-1"}],
        }
    }

    flask_resp = Response(json.dumps(payload), mimetype="application/json")
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/registered-models/get", method="GET", query_string={"name": "model-xyz"}
    ):
        auth_module.redact_get_registered_model_versions(flask_resp)
    model = json.loads(flask_resp.get_data(as_text=True))["registered_model"]

    assert model["latest_versions"][0]["version"] == "3"
    assert "run_id" not in model["latest_versions"][0]


def test_get_run_withholds_model_links_on_a_logged_model_deny(
    workspace_permission_setup, monkeypatch
):
    """GetLoggedModel applies the model tier to these ids, so a run must not hand them out."""
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("logged_model", "*", DENY.name)])
    payload = {
        "run": {
            "info": {"run_id": "r-1", "experiment_id": "exp-1"},
            "inputs": {"model_inputs": [{"model_id": "m-1"}]},
            "outputs": {"model_outputs": [{"model_id": "m-2", "step": 1}]},
        }
    }

    flask_resp = Response(json.dumps(payload), mimetype="application/json")
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/runs/get", method="GET", query_string={"run_id": "r-1"}
    ):
        auth_module.redact_get_run_model_links(flask_resp)
    run = json.loads(flask_resp.get_data(as_text=True))["run"]

    assert "model_inputs" not in run.get("inputs", {})
    assert "model_outputs" not in run.get("outputs", {})
    # The run itself is the subject and survives.
    assert run["info"]["run_id"] == "r-1"


def test_get_run_keeps_model_links_without_a_deny(workspace_permission_setup, monkeypatch):
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    payload = {
        "run": {"info": {"run_id": "r-1"}, "inputs": {"model_inputs": [{"model_id": "m-1"}]}}
    }
    flask_resp = Response(json.dumps(payload), mimetype="application/json")
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/runs/get", method="GET", query_string={"run_id": "r-1"}
    ):
        auth_module.redact_get_run_model_links(flask_resp)
    run = json.loads(flask_resp.get_data(as_text=True))["run"]
    assert run["inputs"]["model_inputs"][0]["model_id"] == "m-1"


def test_every_metadata_bearing_trace_response_is_registered_for_redaction():
    """Coverage invariant, not a behaviour test: the per-handler tests call the function directly,
    so they stay green if a route is never wired. Asserted over the protos rather than a fixed list
    so a new trace response carrying metadata fails here instead of silently leaking.
    """
    from mlflow.protos import service_pb2

    def names_metadata(descriptor, depth=0, seen=None):
        seen = seen or set()
        if descriptor.full_name in seen or depth > 3:
            return False
        seen = seen | {descriptor.full_name}
        for field in descriptor.fields:
            if field.name in ("request_metadata", "trace_metadata"):
                return True
            if field.message_type and names_metadata(field.message_type, depth + 1, seen):
                return True
        return False

    # A proto whose route is served by `_not_implemented` returns 501 and never emits a body, so
    # it has nothing to redact -- SearchUnifiedTraces is declared as an rpc and routed, but only to
    # that stub, which is also why the auth layer maps its path to a None validator.
    from mlflow.protos import databricks_pb2
    from mlflow.server import handlers

    stubbed = {
        path
        for path, handler, _ in handlers.get_endpoints()
        if handler.__name__ == "_not_implemented"
    }

    def is_stubbed(proto_name):
        for service in service_pb2.DESCRIPTOR.services_by_name.values():
            for method in service.methods:
                if method.input_type.name != proto_name:
                    continue
                declared = [
                    f"/api/2.0{endpoint.path}".replace("{", "<").replace("}", ">")
                    for endpoint in method.GetOptions().Extensions[databricks_pb2.rpc].endpoints
                ]
                return bool(declared) and all(path in stubbed for path in declared)
        return False

    unwired = []
    for name in dir(service_pb2):
        proto = getattr(service_pb2, name)
        response = getattr(proto, "Response", None)
        if response is None or not names_metadata(response.DESCRIPTOR):
            continue
        if proto in auth_module.AFTER_REQUEST_PATH_HANDLERS or is_stubbed(name):
            continue
        unwired.append(name)

    assert unwired == []
    assert is_stubbed("SearchUnifiedTraces")


@pytest.mark.parametrize(
    ("handler", "payload", "metadata_at"),
    [
        (
            "redact_batch_trace_assessments",
            {
                "traces": [
                    {
                        "trace_info": {
                            "trace_id": "t-1",
                            "trace_metadata": {"mlflow.sourceRun": "r-1", "other": "keep"},
                        }
                    }
                ]
            },
            lambda body: body["traces"][0]["trace_info"]["trace_metadata"],
        ),
        (
            "redact_batch_trace_info_assessments",
            {
                "trace_infos": [
                    {
                        "trace_id": "t-1",
                        "trace_metadata": {"mlflow.sourceRun": "r-1", "other": "keep"},
                    }
                ]
            },
            lambda body: body["trace_infos"][0]["trace_metadata"],
        ),
        (
            "redact_start_trace_metadata",
            {
                "trace_info": {
                    "request_id": "t-1",
                    "request_metadata": [
                        {"key": "mlflow.sourceRun", "value": "r-1"},
                        {"key": "other", "value": "keep"},
                    ],
                }
            },
            lambda body: {e["key"]: e["value"] for e in body["trace_info"]["request_metadata"]},
        ),
        (
            "redact_end_trace_metadata",
            {
                "trace_info": {
                    "request_id": "t-1",
                    "request_metadata": [
                        {"key": "mlflow.sourceRun", "value": "r-1"},
                        {"key": "other", "value": "keep"},
                    ],
                }
            },
            lambda body: {e["key"]: e["value"] for e in body["trace_info"]["request_metadata"]},
        ),
    ],
    ids=["batch-get-traces", "batch-get-trace-infos", "start-trace-v2", "end-trace"],
)
def test_every_trace_route_strips_a_denied_runs_id_from_metadata(
    workspace_permission_setup, monkeypatch, handler, payload, metadata_at
):
    """The sibling strip was composed into three of the five assessment handlers and neither V2
    write route, so BatchGetTraces/BatchGetTraceInfos/StartTrace/EndTrace still returned
    `mlflow.sourceRun` under a run DENY that GetTrace withheld.
    """
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("run", "*", DENY.name)])

    flask_resp = Response(json.dumps(payload), mimetype="application/json")
    with auth_module.app.test_request_context("/api/3.0/mlflow/traces", method="POST", json={}):
        with workspace_context.WorkspaceContext("team-a"):
            getattr(auth_module, handler)(flask_resp)
    md = metadata_at(json.loads(flask_resp.get_data(as_text=True)))

    assert "mlflow.sourceRun" not in md
    assert md["other"] == "keep"


def test_trace_metadata_strips_only_the_denied_sibling_tier(
    workspace_permission_setup, monkeypatch
):
    """v2 already refuses FILTERING a trace search by metadata.mlflow.sourceRun on the run tier;
    returning the value is the other half. Each key maps to its own tier, so a run DENY must not
    strip the model id.
    """
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("run", "*", DENY.name)])
    payload = {
        "trace": {
            "trace_info": {
                "trace_id": "t-1",
                "trace_metadata": {
                    "mlflow.sourceRun": "r-1",
                    "mlflow.modelId": "m-1",
                    "other": "keep",
                },
            }
        }
    }

    flask_resp = Response(json.dumps(payload), mimetype="application/json")
    with auth_module.app.test_request_context("/api/3.0/mlflow/traces", method="POST", json={}):
        auth_module.redact_start_trace_v3_metadata(flask_resp)
    md = json.loads(flask_resp.get_data(as_text=True))["trace"]["trace_info"]["trace_metadata"]

    assert "mlflow.sourceRun" not in md
    assert md["mlflow.modelId"] == "m-1"
    assert md["other"] == "keep"


def test_trace_metadata_handles_the_v2_repeated_spelling(workspace_permission_setup, monkeypatch):
    """TraceInfo carries repeated request_metadata entries, not a map."""
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("logged_model", "*", DENY.name)])
    payload = {
        "trace_info": {
            "request_id": "t-1",
            "request_metadata": [
                {"key": "mlflow.modelId", "value": "m-1"},
                {"key": "keep", "value": "v"},
            ],
        }
    }

    flask_resp = Response(json.dumps(payload), mimetype="application/json")
    with auth_module.app.test_request_context("/api/2.0/mlflow/traces/t-1/info", method="GET"):
        auth_module.redact_trace_info_metadata(flask_resp)
    entries = json.loads(flask_resp.get_data(as_text=True))["trace_info"]["request_metadata"]
    keys = [e["key"] for e in entries]

    assert "mlflow.modelId" not in keys
    assert "keep" in keys


def test_attach_model_response_redacts_a_denied_secret(workspace_permission_setup, monkeypatch):
    """Attach requires can_use on the DEFINITION, which says nothing about the secret it names."""
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("gateway_secret", "sec-1", DENY.name)])
    payload = {
        "mapping": {
            "mapping_id": "map-1",
            "endpoint_id": "ep-1",
            "model_definition_id": "md-1",
            "model_definition": {
                "model_definition_id": "md-1",
                "name": "md",
                "secret_id": "sec-1",
                "secret_name": "s",
                "provider": "openai",
            },
        }
    }

    flask_resp = Response(json.dumps(payload), mimetype="application/json")
    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/gateway/endpoints/models/attach", method="POST", json={}
    ):
        auth_module.redact_attached_model_mapping(flask_resp)
    mapping = json.loads(flask_resp.get_data(as_text=True))["mapping"]

    assert "secret_id" not in mapping["model_definition"]
    assert "secret_name" not in mapping["model_definition"]
    # The definition itself was authorized by can_use and survives.
    assert mapping["model_definition"]["provider"] == "openai"


@pytest.mark.parametrize(
    ("run_grant", "expected"),
    [(None, True), ("READ", True), ("DENY", False)],
    ids=["no-run-grant-falls-back-to-experiment", "run-read", "run-deny"],
)
def test_create_logged_model_requires_read_on_a_named_source_run(
    workspace_permission_setup, monkeypatch, run_grant, expected
):
    """Binding a new logged model to another user's run would launder artifact access through the
    model, so a named `source_run_id` needs run READ -- the same check the version create performs.
    """
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    rows = [("experiment", "exp-1", MANAGE.name)]
    if run_grant:
        rows.append(("run", "*", run_grant))
    _grant(store, username, "team-a", rows)

    tracking = auth_module._get_tracking_store()
    monkeypatch.setattr(
        tracking,
        "get_run",
        lambda run_id: SimpleNamespace(info=SimpleNamespace(experiment_id="exp-1")),
        raising=False,
    )
    body = {"experiment_id": "exp-1", "name": "m", "source_run_id": "run-1"}
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/logged-models", method="POST", json=body
    ):
        with workspace_context.WorkspaceContext("team-a"):
            assert auth_module.validate_can_create_logged_model() is expected


def test_create_logged_model_without_a_source_run_is_unaffected(
    workspace_permission_setup, monkeypatch
):
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    # A run DENY must not block a create that names no run.
    _grant(
        store, username, "team-a", [("experiment", "exp-1", MANAGE.name), ("run", "*", DENY.name)]
    )

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/logged-models",
        method="POST",
        json={"experiment_id": "exp-1", "name": "m"},
    ):
        with workspace_context.WorkspaceContext("team-a"):
            assert auth_module.validate_can_create_logged_model() is True


@pytest.mark.parametrize(
    ("run_grant", "run_id_kept"),
    [(None, True), ("READ", True), ("DENY", False)],
    ids=["no-run-grant", "run-read", "run-deny"],
)
def test_logged_model_metrics_withhold_a_denied_runs_id(
    workspace_permission_setup, monkeypatch, run_grant, run_id_kept
):
    """The mirror of the run case: the logged model is the subject here, so its own `model_id`
    stays and `run_id` is the sibling that can be denied.
    """
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    if run_grant:
        _grant(store, username, "team-a", [("run", "*", run_grant)])

    payload = {
        "model": {
            "info": {"model_id": "m-1", "experiment_id": "exp-1"},
            "data": {"metrics": [_metric_row()]},
        }
    }
    flask_resp = Response(json.dumps(payload), mimetype="application/json")
    with auth_module.app.test_request_context("/api/2.0/mlflow/logged-models/m-1", method="GET"):
        with workspace_context.WorkspaceContext("team-a"):
            auth_module.redact_get_logged_model_run_ids(flask_resp)
    metric = json.loads(flask_resp.get_data(as_text=True))["model"]["data"]["metrics"][0]

    assert bool(metric.get("run_id")) is run_id_kept
    # The model is the subject and the caller passed its read check, so its own id stays.
    assert metric["model_id"] == "m-1"


def _metric_row(model_id="m-1", run_id="run-1"):
    return {
        "key": "acc",
        "value": 0.9,
        "timestamp": 1,
        "step": 0,
        "model_id": model_id,
        "run_id": run_id,
    }


@pytest.mark.parametrize(
    ("model_grant", "model_id_kept"),
    [(None, True), ("READ", True), ("DENY", False)],
    ids=["no-model-grant", "model-read", "model-deny"],
)
def test_get_run_metrics_withhold_a_denied_models_id(
    workspace_permission_setup, monkeypatch, model_grant, model_id_kept
):
    """A metric is dual-homed: it belongs to the run AND names the logged model it was logged
    against, so it is a second route to a model id beyond model_inputs/model_outputs.
    """
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    if model_grant:
        _grant(store, username, "team-a", [("logged_model", "*", model_grant)])

    payload = {
        "run": {
            "info": {"run_id": "run-1", "experiment_id": "exp-1"},
            "data": {"metrics": [_metric_row()]},
        }
    }
    flask_resp = Response(json.dumps(payload), mimetype="application/json")
    with auth_module.app.test_request_context("/api/2.0/mlflow/runs/get", method="GET"):
        with workspace_context.WorkspaceContext("team-a"):
            auth_module.redact_get_run_model_links(flask_resp)
    metric = json.loads(flask_resp.get_data(as_text=True))["run"]["data"]["metrics"][0]

    assert bool(metric.get("model_id")) is model_id_kept
    # The run is the route's subject and the caller passed its read check, so its own id stays.
    assert metric["run_id"] == "run-1"
    assert metric["value"] == 0.9


def test_metric_history_withholds_a_denied_models_id(workspace_permission_setup, monkeypatch):
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("logged_model", "*", DENY.name)])

    flask_resp = Response(
        json.dumps({"metrics": [_metric_row(), _metric_row(model_id="")]}),
        mimetype="application/json",
    )
    with auth_module.app.test_request_context("/api/2.0/mlflow/metrics/get-history", method="GET"):
        with workspace_context.WorkspaceContext("team-a"):
            auth_module.redact_metric_history_model_ids(flask_resp)
    metrics = json.loads(flask_resp.get_data(as_text=True))["metrics"]

    # The invariant is that no row discloses a model id. A row that named none serializes as an
    # empty string rather than being dropped, which is equally non-disclosing.
    assert not any(m.get("model_id") for m in metrics)
    assert [m["key"] for m in metrics] == ["acc", "acc"]


_ID_GRAIN_TYPES = [
    "experiment",
    "registered_model",
    "prompt",
    "scorer",
    "gateway_secret",
    "gateway_endpoint",
    "gateway_model_definition",
    "mcp_server",
]


@pytest.mark.parametrize("resource_type", _ID_GRAIN_TYPES)
@pytest.mark.parametrize(
    "rows",
    [
        # A per-id DENY, and a per-id DENY that must beat a wildcard positive grant.
        [("{t}", "res-1", "DENY")],
        [("{t}", "*", "MANAGE"), ("{t}", "res-1", "DENY")],
        [("{t}", "*", "DENY")],
    ],
    ids=["id-deny", "id-deny-beats-wildcard-manage", "wildcard-deny"],
)
def test_id_grain_types_honor_deny_in_response_filtering(
    workspace_permission_setup, monkeypatch, resource_type, rows
):
    """Every WILDCARD_AND_ID type must honor DENY on the read predicate each list filter uses.

    `_role_based_read_predicate` is what `filter_search_experiments`,
    `filter_list_gateway_model_definitions`, `filter_list_gateway_endpoints`,
    `filter_list_gateway_secrets`, `filter_search_registered_models`, `filter_list_scorers` and the
    MCP filters all build their per-row decision from, so one assertion covers the response side
    for the whole family.
    """
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [(r[0].format(t=resource_type), r[1], r[2]) for r in rows])

    with auth_module.app.test_request_context("/api/2.0/mlflow/experiments/get", method="GET"):
        with workspace_context.WorkspaceContext("team-a"):
            predicate = auth_module._role_based_read_predicate(username, resource_type)
            assert predicate("res-1") is False


@pytest.mark.parametrize(
    ("validator", "path", "body", "action"),
    [
        (
            "validate_can_read_gateway_model_definition",
            "/api/3.0/mlflow/gateway/model-definitions/get",
            {"model_definition_id": "res-1"},
            "read",
        ),
        (
            "validate_can_delete_gateway_model_definition",
            "/api/3.0/mlflow/gateway/model-definitions/delete",
            {"model_definition_id": "res-1"},
            "delete",
        ),
        (
            "validate_can_read_gateway_endpoint",
            "/api/3.0/mlflow/gateway/endpoints/get",
            {"endpoint_id": "res-1"},
            "read",
        ),
        (
            "validate_can_read_gateway_secret",
            "/api/3.0/mlflow/gateway/secrets/get",
            {"secret_id": "res-1"},
            "read",
        ),
    ],
)
def test_gateway_point_validators_honor_a_per_id_deny(
    workspace_permission_setup, monkeypatch, validator, path, body, action
):
    """The gateway validators resolve through the legacy `Permission` path, which shares
    `fold_grants_for_key` with the requirement model -- so a per-id DENY beats a wildcard MANAGE
    there exactly as it does on a framework route.
    """
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    tier = {
        "model_definition": "gateway_model_definition",
        "endpoint": "gateway_endpoint",
        "secret": "gateway_secret",
    }[next(k for k in ("model_definition", "endpoint", "secret") if f"{k}_id" in body)]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [(tier, "*", MANAGE.name), (tier, "res-1", DENY.name)])

    with auth_module.app.test_request_context(path, method="POST", json=body):
        assert getattr(auth_module, validator)() is False


def _definition_payload(secret_id="sec-1"):
    return {
        "model_definition": {
            "model_definition_id": "md-1",
            "name": "md",
            "secret_id": secret_id,
            "secret_name": "my-secret",
            "provider": "openai",
        }
    }


@pytest.mark.parametrize(
    ("handler", "path"),
    [
        (
            "redact_get_gateway_model_definition_secrets",
            "/api/3.0/mlflow/gateway/model-definitions/get",
        ),
        (
            "redact_update_gateway_model_definition_secrets",
            "/api/3.0/mlflow/gateway/model-definitions/update",
        ),
    ],
)
def test_model_definition_responses_redact_a_denied_secret(
    workspace_permission_setup, monkeypatch, handler, path
):
    """The definition is the SUBJECT here and is already gated by its own tier, so it survives; the
    secret it names is a passenger of a different grantable type and is withheld.
    """
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("gateway_secret", "sec-1", DENY.name)])

    flask_resp = Response(json.dumps(_definition_payload()), mimetype="application/json")
    with auth_module.app.test_request_context(path, method="GET"):
        getattr(auth_module, handler)(flask_resp)
    definition = json.loads(flask_resp.get_data(as_text=True))["model_definition"]

    assert "secret_id" not in definition
    assert "secret_name" not in definition
    # The definition itself is the subject, not a passenger: it is not withheld.
    assert definition["model_definition_id"] == "md-1"
    assert definition["provider"] == "openai"


def test_list_model_definitions_redacts_secrets_in_rows_it_keeps(
    workspace_permission_setup, monkeypatch
):
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("gateway_model_definition", "*", READ.name),
            ("gateway_secret", "sec-1", DENY.name),
        ],
    )
    payload = {
        "model_definitions": [
            {"model_definition_id": "md-1", "secret_id": "sec-1", "secret_name": "a"},
            {"model_definition_id": "md-2", "secret_id": "sec-2", "secret_name": "b"},
        ]
    }
    flask_resp = Response(json.dumps(payload), mimetype="application/json")
    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/gateway/model-definitions/list", method="GET"
    ):
        auth_module.filter_list_gateway_model_definitions(flask_resp)
    rows = json.loads(flask_resp.get_data(as_text=True))["model_definitions"]

    assert len(rows) == 2
    assert "secret_id" not in rows[0]
    assert rows[1]["secret_id"] == "sec-2"


def test_create_endpoint_vetoes_a_denied_experiment(workspace_permission_setup, monkeypatch):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    monkeypatch.setattr(
        auth_module, "_validate_can_use_model_definitions_for_create", lambda configs: True
    )
    _grant(store, username, "team-a", [("experiment", "exp-1", DENY.name)])

    def _create(experiment_id):
        with auth_module.app.test_request_context(
            "/api/3.0/mlflow/gateway/endpoints/create",
            method="POST",
            json={"name": "ep", "experiment_id": experiment_id},
        ):
            return auth_module.validate_can_create_gateway_endpoint()

    assert _create("exp-1") is False
    assert _create("exp-2") is True


def test_create_endpoint_response_redacts_denied_definitions(
    workspace_permission_setup, monkeypatch
):
    """CreateGatewayEndpoint's only after-request handler is grant bookkeeping, so the redaction
    composes there -- the endpoint is the caller's own but its definitions are not.
    """
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("gateway_model_definition", "md-1", DENY.name)])
    monkeypatch.setattr(store, "grant_user_permission", lambda *a, **k: None, raising=False)

    flask_resp = Response(json.dumps(_endpoint_payload()), mimetype="application/json")
    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/gateway/endpoints/create", method="POST", json={"name": "ep"}
    ):
        auth_module.set_can_manage_gateway_endpoint_permission(flask_resp)
    mapping = json.loads(flask_resp.get_data(as_text=True))["endpoint"]["model_mappings"][0]

    assert "model_definition" not in mapping
    assert "model_definition_id" not in mapping


def _endpoint_payload(definition_id="md-1", secret_id="sec-1"):
    return {
        "endpoint": {
            "endpoint_id": "ep-1",
            "model_mappings": [
                {
                    "mapping_id": "map-1",
                    "endpoint_id": "ep-1",
                    "model_definition_id": definition_id,
                    "model_definition": {
                        "model_definition_id": definition_id,
                        "name": "md",
                        "secret_id": secret_id,
                        "secret_name": "my-secret",
                        "provider": "openai",
                    },
                }
            ],
        }
    }


def _run_endpoint_redaction(handler, payload):
    flask_resp = Response(json.dumps(payload), mimetype="application/json")
    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/gateway/endpoints/get",
        method="GET",
        query_string={"endpoint_id": "ep-1"},
    ):
        getattr(auth_module, handler)(flask_resp)
    return json.loads(flask_resp.get_data(as_text=True))["endpoint"]["model_mappings"][0]


@pytest.mark.parametrize(
    "handler",
    [
        "redact_get_gateway_endpoint_model_definitions",
        "redact_update_gateway_endpoint_model_definitions",
    ],
)
def test_endpoint_responses_redact_a_denied_model_definition(
    workspace_permission_setup, monkeypatch, handler
):
    """GatewayEndpoint.model_mappings embeds a whole GatewayModelDefinition, so Get, Update and List
    are each a route to it. A denied definition takes its id with it, since the id still names it.
    """
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("gateway_model_definition", "md-1", DENY.name)])

    mapping = _run_endpoint_redaction(handler, _endpoint_payload())
    assert "model_definition" not in mapping
    assert "model_definition_id" not in mapping
    # The mapping row itself survives, so the response shape stays valid.
    assert mapping["mapping_id"] == "map-1"


def test_endpoint_responses_redact_only_the_secret_when_the_secret_is_denied(
    workspace_permission_setup, monkeypatch
):
    """A readable definition whose SECRET is denied keeps everything except the two secret fields --
    gateway_secret is its own grantable type named inside the embedded definition.
    """
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("gateway_secret", "sec-1", DENY.name)])

    mapping = _run_endpoint_redaction(
        "redact_get_gateway_endpoint_model_definitions", _endpoint_payload()
    )
    definition = mapping["model_definition"]
    assert "secret_id" not in definition
    assert "secret_name" not in definition
    assert definition["provider"] == "openai"
    assert definition["model_definition_id"] == "md-1"


def test_endpoint_responses_untouched_without_a_deny(workspace_permission_setup, monkeypatch):
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)

    mapping = _run_endpoint_redaction(
        "redact_get_gateway_endpoint_model_definitions", _endpoint_payload()
    )
    assert mapping["model_definition"]["secret_id"] == "sec-1"
    assert mapping["model_definition_id"] == "md-1"


def test_detach_model_vetoes_a_denied_model_definition(workspace_permission_setup, monkeypatch):
    """Detach NAMES a model definition but neither uses nor destroys it, so it vetoes rather than
    requiring can_use as attach does -- requiring more would refuse callers master allows.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    monkeypatch.setattr(auth_module, "_get_gateway_endpoint_permission", lambda endpoint_id: MANAGE)
    _grant(store, username, "team-a", [("gateway_model_definition", "md-1", DENY.name)])

    def _detach(definition_id):
        with auth_module.app.test_request_context(
            "/api/3.0/mlflow/gateway/endpoints/models/detach",
            method="POST",
            json={"endpoint_id": "ep-1", "model_definition_id": definition_id},
        ):
            return auth_module.validate_can_detach_model_from_gateway_endpoint()

    assert _detach("md-1") is False
    # A definition with no DENY is unaffected: master's endpoint-only check still decides.
    assert _detach("md-2") is True


def test_update_endpoint_vetoes_a_denied_experiment(workspace_permission_setup, monkeypatch):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    monkeypatch.setattr(auth_module, "_get_gateway_endpoint_permission", lambda endpoint_id: MANAGE)
    monkeypatch.setattr(auth_module, "_validate_can_use_model_definitions", lambda configs: True)
    _grant(store, username, "team-a", [("experiment", "exp-1", DENY.name)])

    def _update(experiment_id):
        with auth_module.app.test_request_context(
            "/api/3.0/mlflow/gateway/endpoints/update",
            method="POST",
            json={"endpoint_id": "ep-1", "experiment_id": experiment_id},
        ):
            return auth_module.validate_can_update_gateway_endpoint()

    assert _update("exp-1") is False
    assert _update("exp-2") is True


def _get_registered_model_versions(rows, name="model-xyz", tags=None):
    payload = json.dumps({
        "registered_model": {"name": name, "tags": tags or [], "latest_versions": rows}
    })
    flask_resp = Response(payload, mimetype="application/json")
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/registered-models/get", method="GET", query_string={"name": name}
    ):
        auth_module.redact_get_registered_model_versions(flask_resp)
    out = json.loads(flask_resp.get_data(as_text=True))
    return out.get("registered_model", {}).get("latest_versions", [])


def test_latest_versions_are_redacted_by_a_version_deny(workspace_permission_setup, monkeypatch):
    """RegisteredModel embeds ModelVersion rows, so a registered-model response is a second route to
    version data that SearchModelVersions and GetModelVersion already gate. The versions are
    passengers on a model the caller may read, so they are redacted and the row survives.
    """
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("registered_model", "*", READ.name),
            ("registered_model_version", "*", DENY.name),
        ],
    )

    assert _get_registered_model_versions([{"name": "model-xyz", "version": "3"}]) == []
    # The model row itself is still returned -- only the embedded versions are withheld.
    payload = json.dumps({
        "registered_model": {
            "name": "model-xyz",
            "tags": [],
            "latest_versions": [{"name": "model-xyz"}],
        }
    })
    flask_resp = Response(payload, mimetype="application/json")
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/registered-models/get", method="GET", query_string={"name": "model-xyz"}
    ):
        auth_module.redact_get_registered_model_versions(flask_resp)
    assert json.loads(flask_resp.get_data(as_text=True))["registered_model"]["name"] == "model-xyz"


def test_latest_versions_survive_without_a_version_deny(workspace_permission_setup, monkeypatch):
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("registered_model", "*", READ.name)])

    assert len(_get_registered_model_versions([{"name": "model-xyz", "version": "3"}])) == 1


def test_search_registered_models_redacts_embedded_versions(
    workspace_permission_setup, monkeypatch
):
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("registered_model", "*", READ.name),
            ("registered_model_version", "*", DENY.name),
        ],
    )
    payload = json.dumps({
        "registered_models": [
            {"name": "model-xyz", "tags": [], "latest_versions": [{"name": "model-xyz"}]}
        ],
        "next_page_token": "",
    })
    flask_resp = Response(payload, mimetype="application/json")
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/registered-models/search",
        method="GET",
        query_string={"max_results": "100"},
    ):
        auth_module.filter_search_registered_models(flask_resp)
    out = json.loads(flask_resp.get_data(as_text=True))
    assert [rm["name"] for rm in out["registered_models"]] == ["model-xyz"]
    assert out["registered_models"][0].get("latest_versions", []) == []


def _run_delete_experiment(experiment_id="exp-1"):
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/experiments/delete", method="POST", json={"experiment_id": experiment_id}
    ):
        return auth_module.validate_can_delete_experiment()


@pytest.mark.parametrize("tier", ["run", "trace", "logged_model", "assessment", "review_queue"])
def test_experiment_delete_requires_delete_on_what_it_contains(workspace_permission_setup, tier):
    """Deleting an experiment withdraws its contents, so each contained tier carries `delete`. A
    child grant that cannot delete withholds the cascade even from an experiment MANAGE holder --
    tier override means the narrower grant decides, which is the intended reading.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "*", MANAGE.name),
            (tier, "*", EDIT.name),
        ],
    )

    assert _run_delete_experiment() is False


def test_experiment_delete_allowed_when_children_are_deletable(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "*", MANAGE.name),
            ("run", "*", MANAGE.name),
        ],
    )

    assert _run_delete_experiment() is True


def test_experiment_delete_falls_back_to_the_parent(workspace_permission_setup):
    """No child grant at all: the experiment decides, exactly as master does."""
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("experiment", "*", MANAGE.name)])

    assert _run_delete_experiment() is True


def test_registered_model_delete_requires_delete_on_its_versions(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("registered_model", "*", MANAGE.name),
            ("registered_model_version", "*", EDIT.name),
        ],
    )

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/registered-models/delete", method="POST", json={"name": "model-xyz"}
    ):
        assert auth_module.validate_can_delete_registered_model_or_prompt_cascade() is False
        # The alias route destroys no version and keeps the plain parent check.
        assert auth_module._validate_can_delete_registered_model_or_prompt() is True


def _run_version_route(validator, name="model-xyz", method="POST"):
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/model-versions/update",
        method=method,
        json={"name": name, "version": "3", "description": "d"},
    ):
        return getattr(auth_module, validator)()


def _use_prompt_registry(monkeypatch, prompt_names):
    monkeypatch.setattr(
        auth_module,
        "_get_model_registry_store",
        lambda: _RegistryStore({"model-xyz": "team-a", "my-prompt": "team-a"}, prompt_names),
    )


def test_version_mutations_honor_a_version_deny(workspace_permission_setup):
    """UpdateModelVersion, TransitionModelVersionStage, SetModelVersionTag, DeleteModelVersion and
    DeleteModelVersionTag all name an existing version and mutate exactly that version, but each
    resolved only the classified parent -- so the version tier could not restrict version writes.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("registered_model", "model-xyz", EDIT.name),
            ("registered_model_version", "*", DENY.name),
        ],
    )

    assert _run_version_route("validate_can_update_model_or_prompt_version") is False
    assert _run_version_route("validate_can_delete_model_or_prompt_version") is False
    # The parent's own routes are untouched by a version denial.
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/registered-models/update", method="POST", json={"name": "model-xyz"}
    ):
        assert auth_module._validate_can_update_registered_model_or_prompt() is True


def test_version_tier_confers_authority_without_registry_management(workspace_permission_setup):
    """The point of the tier: READ on the registry entry plus EDIT on the version tier allows
    version work, matching how (experiment READ + run EDIT) authorizes run updates.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("registered_model", "model-xyz", READ.name),
            ("registered_model_version", "*", EDIT.name),
        ],
    )

    assert _run_version_route("validate_can_update_model_or_prompt_version") is True


def test_version_mutations_still_inherit_from_the_parent(workspace_permission_setup):
    """No version grant: the registry entry governs, exactly as before."""
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)
    _grant(store, username, "team-a", [("registered_model", "model-xyz", EDIT.name)])

    assert _run_version_route("validate_can_update_model_or_prompt_version") is True


def test_version_grant_cannot_outrun_a_parent_deny(workspace_permission_setup):
    """Version grain is wildcard-only, so without the parent READ baseline one version grant would
    reach every version in the workspace -- and would end the fallback chain before the parent DENY
    was consulted.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("registered_model", "model-xyz", DENY.name),
            ("registered_model_version", "*", EDIT.name),
        ],
    )

    assert _run_version_route("validate_can_update_model_or_prompt_version") is False


def test_prompt_versions_resolve_to_the_prompt_version_tier(
    workspace_permission_setup, monkeypatch
):
    """A prompt IS a registered model carrying a tag, so the tier is known only by fetching. A
    prompt's versions must answer to prompt_version, and neither family may be mutated through the
    other's tier.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)
    _use_prompt_registry(monkeypatch, {"my-prompt"})
    _grant(
        store,
        username,
        "team-a",
        [
            ("prompt", "my-prompt", READ.name),
            ("registered_model_version", "*", EDIT.name),
        ],
    )

    # The model-version tier must NOT authorize a prompt version.
    assert (
        _run_version_route("validate_can_update_model_or_prompt_version", name="my-prompt") is False
    )

    _grant(store, username, "team-a", [("prompt_version", "*", EDIT.name)])
    assert (
        _run_version_route("validate_can_update_model_or_prompt_version", name="my-prompt") is True
    )


def test_prompt_version_deny_does_not_block_model_versions(workspace_permission_setup, monkeypatch):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, NO_PERMISSIONS.name)
    _use_prompt_registry(monkeypatch, {"my-prompt"})
    _grant(
        store,
        username,
        "team-a",
        [
            ("registered_model", "model-xyz", EDIT.name),
            ("prompt_version", "*", DENY.name),
        ],
    )

    assert _run_version_route("validate_can_update_model_or_prompt_version") is True


def _deny_tier(workspace_permission_setup, tier):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "*", EDIT.name),
            (tier, "*", DENY.name),
        ],
    )


def test_search_traces_refuses_a_run_backed_filter(workspace_permission_setup):
    """A run_id filter reveals the run's existence and its association with these traces through
    which rows come back, so a run DENY has to refuse it. The parser normalizes every spelling to
    the same request_metadata comparison.
    """
    _deny_tier(workspace_permission_setup, "run")

    assert _run_search_traces("run_id = 'run-1'") is False
    assert _run_search_traces("attributes.run_id = 'run-1'") is False
    assert _run_search_traces("metadata.`mlflow.sourceRun` = 'run-1'") is False
    # Broad operators need no special handling: run grain is wildcard-only, so one key decides.
    assert _run_search_traces("run_id LIKE '%run%'") is False
    # Unrelated filters are untouched.
    assert _run_search_traces("status = 'OK'") is True
    assert _run_search_traces("feedback.safety = 'no'") is True


def test_search_traces_refuses_a_logged_model_backed_filter(workspace_permission_setup):
    _deny_tier(workspace_permission_setup, "logged_model")

    assert _run_search_traces("metadata.`mlflow.modelId` = 'model-1'") is False
    assert _run_search_traces("run_id = 'run-1'") is True


def test_search_traces_run_filter_allowed_without_a_run_grant(workspace_permission_setup):
    """No run grant: the experiment governs, so nothing master allowed is newly denied."""
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("experiment", "*", READ.name)])

    assert _run_search_traces("run_id = 'run-1'") is True


def test_filter_correlation_refuses_an_assessment_backed_filter(workspace_permission_setup):
    """npmi and the four counts are computed over whatever the filters select."""
    _deny_assessments(workspace_permission_setup)

    assert _run_filter_correlation("feedback.safety = 'no'") is False
    assert _run_filter_correlation("status = 'OK'") is True


def _run_query_trace_metrics(view_type):
    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/traces/metrics",
        json={
            "experiment_ids": ["exp-1"],
            "view_type": view_type,
            "metric_name": "assessment_count",
            "aggregations": [{"aggregation_type": "COUNT"}],
        },
    ):
        return auth_module.validate_can_query_trace_metrics()


def test_query_trace_metrics_denies_the_assessments_view(workspace_permission_setup):
    """The assessments view returns a verdict histogram and a pass-rate average -- numbers derived
    from assessments, with no field to redact and no row to drop.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "*", EDIT.name),
            ("assessment", "*", DENY.name),
        ],
    )

    assert _run_query_trace_metrics("ASSESSMENTS") is False
    # The other views aggregate nothing from assessments, so they are untouched.
    assert _run_query_trace_metrics("TRACES") is True
    assert _run_query_trace_metrics("SPANS") is True


def test_query_trace_metrics_assessments_view_allowed_without_a_grant(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("experiment", "*", READ.name)])

    assert _run_query_trace_metrics("ASSESSMENTS") is True


def _run_trace_redaction(experiment_id="exp-1", assessment_names=("a1", "a2")):
    """Run ``redact_trace_assessments`` over a GetTrace response and return the names kept."""
    import json as _json

    from mlflow.protos.service_pb2 import GetTrace
    from mlflow.utils.proto_json_utils import message_to_json, parse_dict

    message = GetTrace.Response()
    parse_dict(
        {
            "trace": {
                "trace_info": {
                    "trace_id": "trace-1",
                    "trace_location": {"mlflow_experiment": {"experiment_id": experiment_id}},
                    "assessments": [{"assessment_name": n} for n in assessment_names],
                }
            }
        },
        message,
    )
    resp = SimpleNamespace(json=_json.loads(message_to_json(message)), data=None)
    with auth_module.app.test_request_context("/api/2.0/mlflow/traces/trace-1"):
        auth_module.redact_trace_assessments(resp)
    out = GetTrace.Response()
    parse_dict(_json.loads(resp.data) if resp.data is not None else resp.json, out)
    return [a.assessment_name for a in out.trace.trace_info.assessments], out


def test_trace_assessments_redacted_on_assessment_deny(workspace_permission_setup):
    """A DENY on the assessment tier withholds the assessments but keeps the trace."""
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "exp-1", READ.name),
            ("assessment", "*", DENY.name),
        ],
    )

    kept, response = _run_trace_redaction()
    assert kept == []
    # The trace is still returned -- redaction, not denial.
    assert response.trace.trace_info.trace_id == "trace-1"


def test_trace_assessments_kept_when_inherited_from_the_experiment(workspace_permission_setup):
    """No assessment grant: the experiment governs through the fallback, so they stay.

    This is the compatibility case -- a deployment that never grants the assessment tier must see
    exactly what it saw before.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("experiment", "exp-1", READ.name)])

    kept, _ = _run_trace_redaction()
    assert kept == ["a1", "a2"]


def test_trace_assessments_kept_on_explicit_assessment_read(workspace_permission_setup):
    """An explicit positive grant on the assessment tier keeps them."""
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "exp-1", READ.name),
            ("assessment", "*", READ.name),
        ],
    )

    kept, _ = _run_trace_redaction()
    assert kept == ["a1", "a2"]


def test_retention_gate_keeps_each_tier_separate(workspace_permission_setup):
    """The contract: one call, one query, a boolean PER resource -- not a conjunction.

    This is what a response filter needs and ``authorize`` cannot give it: keep the trace, drop
    the assessments. Mixing a permitted tier with a denied one in a single call must return both
    answers rather than collapsing to False.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(
        store,
        username,
        "team-a",
        [
            ("experiment", "exp-1", READ.name),
            ("assessment", "*", DENY.name),
        ],
    )

    experiment = (auth_module.RESOURCE_TYPE_EXPERIMENT, "exp-1")
    with auth_module.app.test_request_context("/"):
        decisions = auth_module.retention_gate(
            username,
            experiment,
            [
                Requirement(
                    auth_module.RESOURCE_TYPE_TRACE, "*", "read", fallback_if_no_grant=(experiment,)
                ),
                Requirement(
                    auth_module.RESOURCE_TYPE_ASSESSMENT,
                    "*",
                    "read",
                    fallback_if_no_grant=(experiment,),
                ),
            ],
        )

    assert decisions.retains(auth_module.RESOURCE_TYPE_TRACE) is True
    assert decisions.retains(auth_module.RESOURCE_TYPE_ASSESSMENT) is False


def test_retention_gate_memoizes_and_fails_closed(workspace_permission_setup):
    """Repeated requirements over one resource collapse to a single entry.

    Item 9's duplicate-requirement concern costs nothing here. Separately: an unresolvable
    workspace yields False for every requirement, so a caller that withholds on False fails
    closed with no special case.
    """
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, USE.name)
    _grant(store, username, "team-a", [("experiment", "exp-1", READ.name)])

    experiment = (auth_module.RESOURCE_TYPE_EXPERIMENT, "exp-1")
    template = Requirement(
        auth_module.RESOURCE_TYPE_ASSESSMENT, "*", "read", fallback_if_no_grant=(experiment,)
    )
    with auth_module.app.test_request_context("/"):
        gate = auth_module.retention_gate(username, experiment, [template])
        assert gate.retains(auth_module.RESOURCE_TYPE_ASSESSMENT) is True
        # Asked again, answered from the memo rather than refolded.
        assert gate.retains(auth_module.RESOURCE_TYPE_ASSESSMENT) is True

        # A type no template covered is a programming error, not a False.
        with pytest.raises(KeyError, match="No requirement template"):
            gate.retains(auth_module.RESOURCE_TYPE_SCORER, "exp-1/x")

    # Two templates on one type cannot be told apart by retains().
    with auth_module.app.test_request_context("/"):
        with pytest.raises(ValueError, match="Two requirement templates"):
            auth_module.retention_gate(username, experiment, [template, template])

    # Unknown experiment -> the anchor workspace cannot be resolved -> everything withheld.
    with auth_module.app.test_request_context("/"):
        closed = auth_module.retention_gate(
            username,
            (auth_module.RESOURCE_TYPE_EXPERIMENT, "does-not-exist"),
            [template],
        )
    assert closed.retains(auth_module.RESOURCE_TYPE_ASSESSMENT) is False
