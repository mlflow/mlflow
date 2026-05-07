import pytest

from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.exceptions import MlflowException
from mlflow.server.auth.entities import WorkspacePermission
from mlflow.server.auth.permissions import EDIT, MANAGE, READ, USE
from mlflow.server.auth.sqlalchemy_store import SqlAlchemyStore
from mlflow.utils.workspace_context import WorkspaceContext
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

from tests.helper_functions import random_str
from tests.server.auth.test_sqlalchemy_store import _rmp_maker, _user_maker

pytest_plugins = ["tests.server.auth.test_sqlalchemy_store"]


pytestmark = pytest.mark.notrackingurimock


@pytest.fixture
def store(tmp_sqlite_uri):
    store = SqlAlchemyStore()
    store.init_db(tmp_sqlite_uri)
    return store


def test_set_workspace_permission_creates_and_updates(store):
    workspace = "team-alpha"
    username = random_str()
    user = store.create_user(username, random_str())

    perm = store.set_workspace_permission(workspace, username, USE.name)
    assert isinstance(perm, WorkspacePermission)
    assert perm.workspace == workspace
    assert perm.user_id == user.id
    assert perm.permission == USE.name

    updated = store.set_workspace_permission(workspace, username, MANAGE.name)
    assert updated.permission == MANAGE.name


def test_set_workspace_permission_rejects_resource_only_tiers(store):
    """Workspace grants accept only USE / MANAGE under the simplified model;
    READ / EDIT / NO_PERMISSIONS are rejected up front by the scope-aware
    validator.
    """
    workspace = "team-strict"
    username = random_str()
    store.create_user(username, random_str())

    for invalid in (READ.name, EDIT.name, "NO_PERMISSIONS"):
        with pytest.raises(MlflowException, match="resource_type='workspace'"):
            store.set_workspace_permission(workspace, username, invalid)


def test_get_workspace_permission_precedence(store):
    workspace = "team-beta"
    username = random_str()
    store.create_user(username, random_str())

    assert store.get_workspace_permission(workspace, username) is None

    store.set_workspace_permission(workspace, username, USE.name)
    perm = store.get_workspace_permission(workspace, username)
    assert perm == USE


def test_get_role_workspace_permission_returns_none_when_no_role(store):
    # No role assignment at all → None (callers fall back to legacy /
    # default lookups).
    username = random_str()
    store.create_user(username, random_str())

    assert store.get_role_workspace_permission("ws1", username) is None


def test_get_role_workspace_permission_returns_workspace_wide_grant(store):
    # Role with (resource_type='workspace', resource_pattern='*', MANAGE)
    # in ws1 surfaces as a MANAGE grant for that workspace.
    username = random_str()
    user = store.create_user(username, random_str())

    role = store.create_role(name="ws-admin", workspace="ws1")
    store.add_role_permission(role.id, "workspace", "*", MANAGE.name)
    store.assign_role_to_user(user.id, role.id)

    assert store.get_role_workspace_permission("ws1", username) == MANAGE


def test_get_role_workspace_permission_ignores_resource_specific_grants(store):
    # A role with only resource-specific grants (e.g. experiment:* READ) must
    # NOT register as a workspace-wide grant — that would over-promote a
    # tab-scoped role into a workspace-level capability.
    username = random_str()
    user = store.create_user(username, random_str())

    role = store.create_role(name="exp-reader", workspace="ws1")
    store.add_role_permission(role.id, "experiment", "*", READ.name)
    store.assign_role_to_user(user.id, role.id)

    assert store.get_role_workspace_permission("ws1", username) is None


def test_get_role_workspace_permission_returns_admin_or_member_grant(store):
    # Under the simplified two-tier workspace model, the unified
    # ``('workspace', '*')`` slot accepts either ``USE`` (regular workspace
    # member) or ``MANAGE`` (workspace admin). ``get_role_workspace_permission``
    # walks every grant on this slot for roles the user is assigned to and
    # returns the max — covering both tiers.
    username = random_str()
    user = store.create_user(username, random_str())

    admin_role = store.create_role(name="ws-admin", workspace="ws1")
    store.add_role_permission(admin_role.id, "workspace", "*", MANAGE.name)
    store.assign_role_to_user(user.id, admin_role.id)

    assert store.get_role_workspace_permission("ws1", username) == MANAGE

    # Also pin the USE case: a separate user with only the workspace-member
    # tier should resolve to USE, not None.
    member_username = random_str()
    member = store.create_user(member_username, random_str())
    member_role = store.create_role(name="ws-member", workspace="ws1")
    store.add_role_permission(member_role.id, "workspace", "*", USE.name)
    store.assign_role_to_user(member.id, member_role.id)

    assert store.get_role_workspace_permission("ws1", member_username) == USE


def test_get_role_workspace_permission_scopes_to_requested_workspace(store):
    # A workspace-wide role grant in ws1 must not leak into ws2's permission
    # check — roles are bound to a single workspace.
    username = random_str()
    user = store.create_user(username, random_str())

    role_ws1 = store.create_role(name="ws-admin", workspace="ws1")
    store.add_role_permission(role_ws1.id, "workspace", "*", MANAGE.name)
    store.assign_role_to_user(user.id, role_ws1.id)

    assert store.get_role_workspace_permission("ws2", username) is None


def test_list_workspace_permissions(store):
    workspace = "team-gamma"
    other_workspace = "team-delta"
    username = random_str()
    other_username = random_str()
    user = store.create_user(username, random_str())
    other_user = store.create_user(other_username, random_str())

    p1 = store.set_workspace_permission(workspace, username, USE.name)
    p2 = store.set_workspace_permission(workspace, other_username, MANAGE.name)
    p3 = store.set_workspace_permission(other_workspace, username, MANAGE.name)

    perms = store.list_workspace_permissions(workspace)
    actual = {(perm.workspace, perm.user_id, perm.permission) for perm in perms}
    expected = {
        (p1.workspace, user.id, p1.permission),
        (p2.workspace, other_user.id, p2.permission),
    }
    assert actual == expected

    perms_other = store.list_workspace_permissions(other_workspace)
    assert {(perm.workspace, perm.user_id, perm.permission) for perm in perms_other} == {
        (p3.workspace, user.id, p3.permission)
    }


def test_delete_workspace_permission(store):
    workspace = "workspace-delete"
    username = random_str()
    store.create_user(username, random_str())

    store.set_workspace_permission(workspace, username, USE.name)

    store.delete_workspace_permission(workspace, username)
    assert store.get_workspace_permission(workspace, username) is None

    with pytest.raises(
        MlflowException,
        match=(
            "Workspace permission does not exist for "
            f"workspace='{workspace}', username='{username}'"
        ),
    ):
        store.delete_workspace_permission(workspace, username)


def test_delete_workspace_permissions_for_workspace(store):
    workspace = "workspace-delete-all"
    other_workspace = "workspace-keep"
    username = random_str()
    store.create_user(username, random_str())

    store.set_workspace_permission(workspace, username, USE.name)
    store.set_workspace_permission(other_workspace, username, MANAGE.name)

    store.delete_workspace_permissions_for_workspace(workspace)

    assert store.list_workspace_permissions(workspace) == []
    remaining = store.list_workspace_permissions(other_workspace)
    assert len(remaining) == 1
    assert remaining[0].workspace == other_workspace


def test_list_accessible_workspace_names(store):
    username = random_str()
    other_user = random_str()
    store.create_user(username, random_str())
    store.create_user(other_user, random_str())

    store.set_workspace_permission("workspace-use", username, USE.name)
    store.set_workspace_permission("workspace-manage", username, MANAGE.name)
    # No grant for `workspace-no-access` — absence should keep it hidden.
    store.set_workspace_permission("workspace-other", other_user, USE.name)

    accessible = store.list_accessible_workspace_names(username)
    assert accessible == {"workspace-use", "workspace-manage"}

    assert store.list_accessible_workspace_names(other_user) == {
        "workspace-other",
    }
    assert store.list_accessible_workspace_names(None) == set()


def test_list_accessible_workspace_names_includes_role_based_workspaces(store):
    # Role assignment with permissions → workspace visible, even without a legacy
    # workspace_permissions row.
    username = random_str()
    user = store.create_user(username, random_str())

    role = store.create_role(name="viewer", workspace="ws1")
    store.add_role_permission(role.id, "experiment", "*", READ.name)
    store.assign_role_to_user(user.id, role.id)

    assert store.list_accessible_workspace_names(username) == {"ws1"}


def test_list_accessible_workspace_names_includes_role_with_no_permissions(store):
    # Role membership alone implies visibility — the role has zero permission rows,
    # but the user is still assigned to it. Documents the intentional design: a
    # workspace admin can give someone "membership" without any capability and the
    # UI still surfaces the workspace in their list.
    username = random_str()
    user = store.create_user(username, random_str())

    empty_role = store.create_role(name="shell", workspace="ws1")
    store.assign_role_to_user(user.id, empty_role.id)

    assert store.list_accessible_workspace_names(username) == {"ws1"}


def test_list_accessible_workspace_names_combines_legacy_and_role_sources(store):
    # Legacy USE on ws1 + role assignment in ws2 → both surface.
    username = random_str()
    user = store.create_user(username, random_str())

    store.set_workspace_permission("ws1", username, USE.name)
    role = store.create_role(name="viewer", workspace="ws2")
    store.assign_role_to_user(user.id, role.id)

    assert store.list_accessible_workspace_names(username) == {"ws1", "ws2"}


def test_list_accessible_workspace_names_role_in_other_workspace_doesnt_leak(store):
    # A role assignment in one workspace must not surface unrelated workspaces.
    username = random_str()
    user = store.create_user(username, random_str())

    role_ws1 = store.create_role(name="viewer", workspace="ws1")
    store.assign_role_to_user(user.id, role_ws1.id)
    # ws2 and ws3 exist (via roles for other users) but the user has no assignment.
    store.create_role(name="viewer", workspace="ws2")
    store.create_role(name="viewer", workspace="ws3")

    assert store.list_accessible_workspace_names(username) == {"ws1"}


def test_list_accessible_workspace_names_combines_legacy_and_role_same_workspace(store):
    # Overlap: a user has BOTH a legacy USE grant and a role assignment in the same
    # workspace. Deduplication should collapse to a single entry.
    username = random_str()
    user = store.create_user(username, random_str())

    store.set_workspace_permission("ws1", username, USE.name)
    role = store.create_role(name="viewer", workspace="ws1")
    store.assign_role_to_user(user.id, role.id)

    accessible = store.list_accessible_workspace_names(username)
    assert accessible == {"ws1"}
    assert len(accessible) == 1


def test_rename_registered_model_permissions_scoped_by_workspace(store, monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    username = random_str()
    password = random_str()
    _user_maker(store, username, password)

    with WorkspaceContext("workspace-a"):
        _rmp_maker(store, "model", username, READ.name)
    with WorkspaceContext("workspace-b"):
        _rmp_maker(store, "model", username, READ.name)

    with WorkspaceContext("workspace-a"):
        store.rename_registered_model_permissions("model", "model-renamed")
        renamed = store.get_registered_model_permission("model-renamed", username)
        assert renamed.name == "model-renamed"
        assert renamed.workspace == "workspace-a"
        with pytest.raises(
            MlflowException,
            match=(
                "Registered model permission with workspace=workspace-a, name=model and username="
            ),
        ):
            store.get_registered_model_permission("model", username)

    with WorkspaceContext("workspace-b"):
        still_original = store.get_registered_model_permission("model", username)
        assert still_original.name == "model"
        assert still_original.workspace == "workspace-b"


def test_registered_model_permissions_are_workspace_scoped(store, monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    username = random_str()
    password = random_str()
    _user_maker(store, username, password)

    model_name = random_str()
    workspace_alt = f"workspace-{random_str()}"

    with WorkspaceContext(DEFAULT_WORKSPACE_NAME):
        store.create_registered_model_permission(model_name, username, READ.name)

    with WorkspaceContext(workspace_alt):
        perm_alt = store.create_registered_model_permission(model_name, username, EDIT.name)
        assert perm_alt.workspace == workspace_alt

    with WorkspaceContext(DEFAULT_WORKSPACE_NAME):
        perm_default = store.get_registered_model_permission(model_name, username)
        assert perm_default.permission == READ.name
        assert perm_default.workspace == DEFAULT_WORKSPACE_NAME
        perms_default = store.list_registered_model_permissions(username)
        assert [p.permission for p in perms_default] == [READ.name]

    with WorkspaceContext(workspace_alt):
        perm_alt_lookup = store.get_registered_model_permission(model_name, username)
        assert perm_alt_lookup.permission == EDIT.name
        assert perm_alt_lookup.workspace == workspace_alt
        perms_alt = store.list_registered_model_permissions(username)
        assert [p.permission for p in perms_alt] == [EDIT.name]

    # Switching back to default workspace should not affect alternate workspace permission
    with WorkspaceContext(DEFAULT_WORKSPACE_NAME):
        updated = store.update_registered_model_permission(model_name, username, MANAGE.name)
        assert updated.permission == MANAGE.name
        assert updated.workspace == DEFAULT_WORKSPACE_NAME

    with WorkspaceContext(workspace_alt):
        perm_alt_post_update = store.get_registered_model_permission(model_name, username)
        assert perm_alt_post_update.permission == EDIT.name
        assert perm_alt_post_update.workspace == workspace_alt


def test_delete_registered_model_permissions_scoped_by_workspace(store, monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    username1 = random_str()
    username2 = random_str()
    _user_maker(store, username1, random_str())
    _user_maker(store, username2, random_str())

    model_name = random_str()

    with WorkspaceContext("workspace-a"):
        _rmp_maker(store, model_name, username1, READ.name)
        _rmp_maker(store, model_name, username2, EDIT.name)

    with WorkspaceContext("workspace-b"):
        _rmp_maker(store, model_name, username1, MANAGE.name)

    with WorkspaceContext("workspace-a"):
        store.delete_registered_model_permissions(model_name)
        with pytest.raises(MlflowException, match="Registered model permission .* not found"):
            store.get_registered_model_permission(model_name, username1)
        with pytest.raises(MlflowException, match="Registered model permission .* not found"):
            store.get_registered_model_permission(model_name, username2)

    with WorkspaceContext("workspace-b"):
        remaining = store.get_registered_model_permission(model_name, username1)
        assert remaining.permission == MANAGE.name
        assert remaining.workspace == "workspace-b"
