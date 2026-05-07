from contextlib import contextmanager

import pytest
import requests

from mlflow import MlflowException
from mlflow.environment_variables import (
    MLFLOW_AUTH_CONFIG_PATH,
    MLFLOW_FLASK_SERVER_SECRET_KEY,
    MLFLOW_TRACKING_PASSWORD,
    MLFLOW_TRACKING_USERNAME,
)
from mlflow.protos.databricks_pb2 import (
    PERMISSION_DENIED,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
    UNAUTHENTICATED,
    ErrorCode,
)
from mlflow.server.auth.client import AuthServiceClient
from mlflow.utils.os import is_windows

from tests.helper_functions import random_str
from tests.server.auth.auth_test_utils import (
    ADMIN_PASSWORD,
    ADMIN_USERNAME,
    User,
    write_isolated_auth_config,
)
from tests.tracking.integration_test_utils import _init_server


@pytest.fixture(autouse=True)
def clear_credentials(monkeypatch):
    monkeypatch.delenv(MLFLOW_TRACKING_USERNAME.name, raising=False)
    monkeypatch.delenv(MLFLOW_TRACKING_PASSWORD.name, raising=False)


@pytest.fixture
def client(tmp_path):
    auth_config_path = write_isolated_auth_config(tmp_path)
    path = tmp_path.joinpath("sqlalchemy.db").as_uri()
    backend_uri = ("sqlite://" if is_windows() else "sqlite:////") + path[len("file://") :]

    with _init_server(
        backend_uri=backend_uri,
        root_artifact_uri=tmp_path.joinpath("artifacts").as_uri(),
        app="mlflow.server.auth:create_app",
        extra_env={
            MLFLOW_FLASK_SERVER_SECRET_KEY.name: "my-secret-key",
            MLFLOW_AUTH_CONFIG_PATH.name: str(auth_config_path),
        },
        server_type="flask",
    ) as url:
        yield AuthServiceClient(url)


@contextmanager
def assert_unauthenticated():
    with pytest.raises(MlflowException, match=r"You are not authenticated.") as exception_context:
        yield
    assert exception_context.value.error_code == ErrorCode.Name(UNAUTHENTICATED)


@contextmanager
def assert_unauthorized():
    with pytest.raises(MlflowException, match=r"Permission denied.") as exception_context:
        yield
    assert exception_context.value.error_code == ErrorCode.Name(PERMISSION_DENIED)


def _create_admin_controlled_user(client, monkeypatch):
    """Create a non-admin user via the admin account and return (username, password)."""
    username = random_str()
    password = random_str()
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.create_user(username, password)
    return username, password


def _make_user_wp_admin(client, monkeypatch, username, workspace):
    """Create a role with workspace-scope MANAGE in ``workspace`` and assign it to ``username``."""
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        role = client.create_role(workspace=workspace, name=f"wp-admin-{random_str()}")
        client.add_role_permission(role.id, "workspace", "*", "MANAGE")
        client.assign_role(username, role.id)
    return role


# ---- Role CRUD ----


def test_create_role_requires_admin(client, monkeypatch):
    username, password = _create_admin_controlled_user(client, monkeypatch)

    with assert_unauthenticated():
        client.create_role(workspace="ws1", name="viewer")

    with User(username, password, monkeypatch), assert_unauthorized():
        client.create_role(workspace="ws1", name="viewer")

    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        role = client.create_role(workspace="ws1", name="viewer", description="read-only")
    assert role.name == "viewer"
    assert role.workspace == "ws1"
    assert role.description == "read-only"


def test_create_role_duplicate(client, monkeypatch):
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.create_role(workspace="ws1", name="viewer")
        with pytest.raises(MlflowException, match="already exists") as exc:
            client.create_role(workspace="ws1", name="viewer")
    assert exc.value.error_code == ErrorCode.Name(RESOURCE_ALREADY_EXISTS)


def test_get_role(client, monkeypatch):
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        created = client.create_role(workspace="ws1", name="viewer")
        fetched = client.get_role(created.id)
    assert fetched.id == created.id
    assert fetched.name == "viewer"


def test_get_role_not_found(client, monkeypatch):
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        with pytest.raises(MlflowException, match="not found") as exc:
            client.get_role(99999)
    assert exc.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_list_roles(client, monkeypatch):
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.create_role(workspace="ws1", name="viewer")
        client.create_role(workspace="ws1", name="editor")
        client.create_role(workspace="ws2", name="viewer")
        ws1 = client.list_roles("ws1")
        ws2 = client.list_roles("ws2")
    assert {r.name for r in ws1} == {"viewer", "editor"}
    assert {r.name for r in ws2} == {"viewer"}


def test_update_role(client, monkeypatch):
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        role = client.create_role(workspace="ws1", name="old")
        updated = client.update_role(role.id, name="new", description="updated")
    assert updated.name == "new"
    assert updated.description == "updated"


def test_update_role_rejects_empty_update(client, monkeypatch):
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        role = client.create_role(workspace="ws1", name="viewer")
        with pytest.raises(MlflowException, match="At least one of 'name' or 'description'"):
            client.update_role(role.id)


def test_delete_role(client, monkeypatch):
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        role = client.create_role(workspace="ws1", name="viewer")
        client.delete_role(role.id)
        with pytest.raises(MlflowException, match="not found"):
            client.get_role(role.id)


# ---- Role permission CRUD ----


def test_role_permission_crud(client, monkeypatch):
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        role = client.create_role(workspace="ws1", name="viewer")
        rp = client.add_role_permission(role.id, "experiment", "42", "READ")
        assert rp.resource_type == "experiment"
        assert rp.resource_pattern == "42"
        assert rp.permission == "READ"

        perms = client.list_role_permissions(role.id)
        assert len(perms) == 1

        updated = client.update_role_permission(rp.id, "EDIT")
        assert updated.permission == "EDIT"

        client.remove_role_permission(rp.id)
        assert client.list_role_permissions(role.id) == []


def test_add_role_permission_invalid_resource_type(client, monkeypatch):
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        role = client.create_role(workspace="ws1", name="viewer")
        with pytest.raises(MlflowException, match="Invalid resource type"):
            client.add_role_permission(role.id, "bogus", "42", "READ")


def test_add_workspace_resource_type(client, monkeypatch):
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        role = client.create_role(workspace="ws1", name="ws-admin")
        rp = client.add_role_permission(role.id, "workspace", "*", "MANAGE")
    assert rp.resource_type == "workspace"
    assert rp.permission == "MANAGE"


# ---- User-role assignment ----


def test_assign_unassign_role(client, monkeypatch):
    username, _ = _create_admin_controlled_user(client, monkeypatch)

    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        role = client.create_role(workspace="ws1", name="viewer")
        assignment = client.assign_role(username, role.id)
        assert assignment.role_id == role.id

        user_roles = client.list_user_roles(username)
        assert [r.id for r in user_roles] == [role.id]

        role_users = client.list_role_users(role.id)
        assert len(role_users) == 1

        client.unassign_role(username, role.id)
        assert client.list_user_roles(username) == []


def test_assign_nonexistent_role(client, monkeypatch):
    username, _ = _create_admin_controlled_user(client, monkeypatch)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        with pytest.raises(MlflowException, match="not found"):
            client.assign_role(username, 99999)


def test_assign_nonexistent_user(client, monkeypatch):
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        role = client.create_role(workspace="ws1", name="viewer")
        with pytest.raises(MlflowException, match="not found"):
            client.assign_role("nonexistent-user-" + random_str(), role.id)


# ---- Authorization: WP admin vs plain user vs super admin ----


def test_wp_admin_can_manage_roles_in_own_workspace(client, monkeypatch):
    wp_admin, wp_admin_pw = _create_admin_controlled_user(client, monkeypatch)
    _make_user_wp_admin(client, monkeypatch, wp_admin, "ws1")

    # WP admin can create and manage roles in their workspace.
    with User(wp_admin, wp_admin_pw, monkeypatch):
        role = client.create_role(workspace="ws1", name="editor")
        client.add_role_permission(role.id, "experiment", "*", "EDIT")
        assert len(client.list_roles("ws1")) >= 2  # the wp-admin role + this one


def test_wp_admin_cannot_manage_other_workspace(client, monkeypatch):
    wp_admin, wp_admin_pw = _create_admin_controlled_user(client, monkeypatch)
    _make_user_wp_admin(client, monkeypatch, wp_admin, "ws1")

    # WP admin of ws1 cannot create roles in ws2.
    with User(wp_admin, wp_admin_pw, monkeypatch), assert_unauthorized():
        client.create_role(workspace="ws2", name="editor")


def test_list_user_roles_self(client, monkeypatch):
    username, password = _create_admin_controlled_user(client, monkeypatch)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        role = client.create_role(workspace="ws1", name="viewer")
        client.assign_role(username, role.id)

    # Users can list their own role assignments.
    with User(username, password, monkeypatch):
        roles = client.list_user_roles(username)
    assert [r.id for r in roles] == [role.id]


def test_list_user_roles_not_self_requires_admin(client, monkeypatch):
    target, _ = _create_admin_controlled_user(client, monkeypatch)
    requester, requester_pw = _create_admin_controlled_user(client, monkeypatch)

    # A plain user cannot view another user's roles.
    with User(requester, requester_pw, monkeypatch), assert_unauthorized():
        client.list_user_roles(target)


def test_list_user_roles_wp_admin_sees_only_own_workspace_scope(client, monkeypatch):
    # Target has roles in two workspaces (ws1, ws2). Requester is WP admin of ws1 only.
    # The response should be filtered to ws1 roles — ws2 membership must not leak.
    target, _ = _create_admin_controlled_user(client, monkeypatch)
    wp_admin, wp_admin_pw = _create_admin_controlled_user(client, monkeypatch)
    _make_user_wp_admin(client, monkeypatch, wp_admin, "ws1")

    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        ws1_role = client.create_role(workspace="ws1", name="ws1-viewer")
        ws2_role = client.create_role(workspace="ws2", name="ws2-viewer")
        client.assign_role(target, ws1_role.id)
        client.assign_role(target, ws2_role.id)

    with User(wp_admin, wp_admin_pw, monkeypatch):
        visible = client.list_user_roles(target)

    visible_names = {r.name for r in visible}
    assert "ws1-viewer" in visible_names
    assert "ws2-viewer" not in visible_names


def test_list_user_roles_super_admin_sees_all(client, monkeypatch):
    target, _ = _create_admin_controlled_user(client, monkeypatch)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        ws1_role = client.create_role(workspace="ws1", name="r1")
        ws2_role = client.create_role(workspace="ws2", name="r2")
        client.assign_role(target, ws1_role.id)
        client.assign_role(target, ws2_role.id)
        visible = client.list_user_roles(target)

    assert {r.name for r in visible} == {"r1", "r2"}


# ---- Cross-workspace admin list ----


def test_list_all_roles_admin_only(client, monkeypatch):
    username, password = _create_admin_controlled_user(client, monkeypatch)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.create_role(workspace="ws1", name="r1")
        client.create_role(workspace="ws2", name="r2")
        all_roles = client.list_all_roles()
    names = {r.name for r in all_roles}
    assert {"r1", "r2"}.issubset(names)

    with User(username, password, monkeypatch), assert_unauthorized():
        client.list_all_roles()


# ---- /api/ and /ajax-api/ path parity ----


@pytest.mark.parametrize("api_prefix", ["api", "ajax-api"])
def test_rbac_endpoints_reachable_at_both_path_prefixes(client, monkeypatch, api_prefix):
    # The MLflow frontend hits /ajax-api/ paths; the Python client hits /api/ paths.
    # Every RBAC route must be reachable at both (see handlers._get_paths).
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        created = client.create_role(workspace="path-parity", name=f"r-{api_prefix}")

    # GET list endpoint.
    resp = requests.get(
        f"{client.tracking_uri}/{api_prefix}/3.0/mlflow/roles/list",
        params={"workspace": "path-parity"},
        auth=(ADMIN_USERNAME, ADMIN_PASSWORD),
    )
    assert resp.status_code == 200, resp.text
    names = {r["name"] for r in resp.json()["roles"]}
    assert created.name in names

    # GET single role by id.
    resp = requests.get(
        f"{client.tracking_uri}/{api_prefix}/3.0/mlflow/roles/get",
        params={"role_id": str(created.id)},
        auth=(ADMIN_USERNAME, ADMIN_PASSWORD),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["role"]["id"] == created.id
