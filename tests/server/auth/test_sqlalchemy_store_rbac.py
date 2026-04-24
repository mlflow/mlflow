import pytest

from mlflow.exceptions import MlflowException
from mlflow.server.auth.entities import Role, RolePermission, UserRoleAssignment
from mlflow.server.auth.permissions import EDIT, MANAGE, READ, USE
from mlflow.server.auth.sqlalchemy_store import SqlAlchemyStore

from tests.helper_functions import random_str

pytestmark = pytest.mark.notrackingurimock


@pytest.fixture
def store(tmp_sqlite_uri):
    store = SqlAlchemyStore()
    store.init_db(tmp_sqlite_uri)
    return store


@pytest.fixture
def user(store):
    return store.create_user(random_str(), random_str())


@pytest.fixture
def user2(store):
    return store.create_user(random_str(), random_str())


# ---- Role CRUD ----


def test_create_role(store):
    role = store.create_role(name="viewer", workspace="ws1", description="Read-only access")
    assert isinstance(role, Role)
    assert role.name == "viewer"
    assert role.workspace == "ws1"
    assert role.description == "Read-only access"
    assert role.permissions == []


def test_create_role_duplicate(store):
    store.create_role(name="viewer", workspace="ws1")
    with pytest.raises(MlflowException, match="already exists"):
        store.create_role(name="viewer", workspace="ws1")


def test_create_role_same_name_different_workspace(store):
    r1 = store.create_role(name="viewer", workspace="ws1")
    r2 = store.create_role(name="viewer", workspace="ws2")
    assert r1.id != r2.id


def test_get_role(store):
    created = store.create_role(name="editor", workspace="ws1")
    fetched = store.get_role(created.id)
    assert fetched.id == created.id
    assert fetched.name == "editor"
    assert fetched.workspace == "ws1"


def test_get_role_not_found(store):
    with pytest.raises(MlflowException, match="not found"):
        store.get_role(99999)


def test_get_role_by_name(store):
    created = store.create_role(name="editor", workspace="ws1")
    fetched = store.get_role_by_name("ws1", "editor")
    assert fetched.id == created.id


def test_get_role_by_name_not_found(store):
    with pytest.raises(MlflowException, match="not found"):
        store.get_role_by_name("ws1", "nonexistent")


def test_list_roles(store):
    store.create_role(name="viewer", workspace="ws1")
    store.create_role(name="editor", workspace="ws1")
    store.create_role(name="viewer", workspace="ws2")

    ws1_roles = store.list_roles("ws1")
    assert len(ws1_roles) == 2
    assert {r.name for r in ws1_roles} == {"viewer", "editor"}

    ws2_roles = store.list_roles("ws2")
    assert len(ws2_roles) == 1


def test_list_all_roles(store):
    store.create_role(name="viewer", workspace="ws1")
    store.create_role(name="editor", workspace="ws2")
    all_roles = store.list_all_roles()
    assert len(all_roles) == 2


def test_update_role(store):
    role = store.create_role(name="old-name", workspace="ws1", description="old desc")
    updated = store.update_role(role.id, name="new-name", description="new desc")
    assert updated.name == "new-name"
    assert updated.description == "new desc"


def test_update_role_name_conflict(store):
    store.create_role(name="existing", workspace="ws1")
    role2 = store.create_role(name="other", workspace="ws1")
    with pytest.raises(MlflowException, match="already exists"):
        store.update_role(role2.id, name="existing")


def test_delete_role(store):
    role = store.create_role(name="doomed", workspace="ws1")
    store.delete_role(role.id)
    with pytest.raises(MlflowException, match="not found"):
        store.get_role(role.id)


def test_delete_role_cascades_permissions_and_assignments(store, user):
    role = store.create_role(name="role1", workspace="ws1")
    store.add_role_permission(role.id, "experiment", "*", "READ")
    store.assign_role_to_user(user.id, role.id)

    store.delete_role(role.id)

    # Role no longer exists
    with pytest.raises(MlflowException, match="not found"):
        store.get_role(role.id)

    # User no longer has the role
    assert store.list_user_roles(user.id) == []


def test_delete_roles_for_workspace(store):
    store.create_role(name="r1", workspace="ws1")
    store.create_role(name="r2", workspace="ws1")
    store.create_role(name="r3", workspace="ws2")

    store.delete_roles_for_workspace("ws1")
    assert store.list_roles("ws1") == []
    assert len(store.list_roles("ws2")) == 1


def test_delete_roles_for_workspace_cascades(store, user):
    role = store.create_role(name="r1", workspace="ws1")
    store.add_role_permission(role.id, "experiment", "*", "READ")
    store.assign_role_to_user(user.id, role.id)

    store.delete_roles_for_workspace("ws1")

    assert store.list_roles("ws1") == []
    assert store.list_user_roles(user.id) == []


# ---- RolePermission CRUD ----


def test_add_role_permission(store):
    role = store.create_role(name="viewer", workspace="ws1")
    rp = store.add_role_permission(role.id, "experiment", "123", "READ")
    assert isinstance(rp, RolePermission)
    assert rp.role_id == role.id
    assert rp.resource_type == "experiment"
    assert rp.resource_pattern == "123"
    assert rp.permission == "READ"


def test_add_role_permission_wildcard(store):
    role = store.create_role(name="viewer", workspace="ws1")
    rp = store.add_role_permission(role.id, "experiment", "*", "READ")
    assert rp.resource_pattern == "*"


def test_add_role_permission_duplicate(store):
    role = store.create_role(name="viewer", workspace="ws1")
    store.add_role_permission(role.id, "experiment", "123", "READ")
    with pytest.raises(MlflowException, match="already exists"):
        store.add_role_permission(role.id, "experiment", "123", "EDIT")


def test_add_role_permission_invalid_permission(store):
    role = store.create_role(name="viewer", workspace="ws1")
    with pytest.raises(MlflowException, match="Invalid permission"):
        store.add_role_permission(role.id, "experiment", "123", "INVALID")


def test_add_role_permission_invalid_resource_type(store):
    role = store.create_role(name="viewer", workspace="ws1")
    with pytest.raises(MlflowException, match="Invalid resource type"):
        store.add_role_permission(role.id, "invalid_type", "123", "READ")


def test_add_role_permission_workspace_requires_wildcard(store):
    role = store.create_role(name="ws-role", workspace="ws1")
    with pytest.raises(MlflowException, match="resource_type='workspace' requires"):
        store.add_role_permission(role.id, "workspace", "42", "MANAGE")


def test_add_role_permission_nonexistent_role(store):
    with pytest.raises(MlflowException, match="not found"):
        store.add_role_permission(99999, "experiment", "123", "READ")


def test_remove_role_permission(store):
    role = store.create_role(name="viewer", workspace="ws1")
    rp = store.add_role_permission(role.id, "experiment", "123", "READ")
    store.remove_role_permission(rp.id)
    assert store.list_role_permissions(role.id) == []


def test_remove_role_permission_not_found(store):
    with pytest.raises(MlflowException, match="not found"):
        store.remove_role_permission(99999)


def test_list_role_permissions(store):
    role = store.create_role(name="viewer", workspace="ws1")
    store.add_role_permission(role.id, "experiment", "1", "READ")
    store.add_role_permission(role.id, "experiment", "2", "EDIT")
    store.add_role_permission(role.id, "registered_model", "*", "READ")

    perms = store.list_role_permissions(role.id)
    assert len(perms) == 3


def test_update_role_permission(store):
    role = store.create_role(name="viewer", workspace="ws1")
    rp = store.add_role_permission(role.id, "experiment", "123", "READ")
    updated = store.update_role_permission(rp.id, "EDIT")
    assert updated.permission == "EDIT"


def test_update_role_permission_not_found(store):
    with pytest.raises(MlflowException, match="not found"):
        store.update_role_permission(99999, "READ")


def test_update_role_permission_invalid_permission(store):
    role = store.create_role(name="viewer", workspace="ws1")
    rp = store.add_role_permission(role.id, "experiment", "123", "READ")
    with pytest.raises(MlflowException, match="Invalid permission"):
        store.update_role_permission(rp.id, "INVALID")


# ---- UserRoleAssignment CRUD ----


def test_assign_role_to_user(store, user):
    role = store.create_role(name="viewer", workspace="ws1")
    assignment = store.assign_role_to_user(user.id, role.id)
    assert isinstance(assignment, UserRoleAssignment)
    assert assignment.user_id == user.id
    assert assignment.role_id == role.id


def test_assign_role_nonexistent_user(store):
    role = store.create_role(name="viewer", workspace="ws1")
    with pytest.raises(MlflowException, match="not found"):
        store.assign_role_to_user(99999, role.id)


def test_assign_role_duplicate(store, user):
    role = store.create_role(name="viewer", workspace="ws1")
    store.assign_role_to_user(user.id, role.id)
    with pytest.raises(MlflowException, match="already exists"):
        store.assign_role_to_user(user.id, role.id)


def test_unassign_role_from_user(store, user):
    role = store.create_role(name="viewer", workspace="ws1")
    store.assign_role_to_user(user.id, role.id)
    store.unassign_role_from_user(user.id, role.id)
    assert store.list_user_roles(user.id) == []


def test_unassign_role_not_found(store, user):
    with pytest.raises(MlflowException, match="not found"):
        store.unassign_role_from_user(user.id, 99999)


def test_list_user_roles(store, user):
    r1 = store.create_role(name="viewer", workspace="ws1")
    r2 = store.create_role(name="editor", workspace="ws2")
    store.assign_role_to_user(user.id, r1.id)
    store.assign_role_to_user(user.id, r2.id)

    roles = store.list_user_roles(user.id)
    assert len(roles) == 2
    assert {r.name for r in roles} == {"viewer", "editor"}


def test_list_user_roles_for_workspace(store, user):
    r1 = store.create_role(name="viewer", workspace="ws1")
    r2 = store.create_role(name="editor", workspace="ws1")
    r3 = store.create_role(name="viewer", workspace="ws2")
    store.assign_role_to_user(user.id, r1.id)
    store.assign_role_to_user(user.id, r2.id)
    store.assign_role_to_user(user.id, r3.id)

    ws1_roles = store.list_user_roles_for_workspace(user.id, "ws1")
    assert len(ws1_roles) == 2

    ws2_roles = store.list_user_roles_for_workspace(user.id, "ws2")
    assert len(ws2_roles) == 1


def test_list_role_users(store, user, user2):
    role = store.create_role(name="viewer", workspace="ws1")
    store.assign_role_to_user(user.id, role.id)
    store.assign_role_to_user(user2.id, role.id)

    users = store.list_role_users(role.id)
    assert len(users) == 2
    assert {u.user_id for u in users} == {user.id, user2.id}


# ---- Role-based permission resolution ----


def test_get_role_permission_no_roles(store, user):
    result = store.get_role_permission_for_resource(user.id, "experiment", "1", "ws1")
    assert result is None


def test_get_role_permission_specific_match(store, user):
    role = store.create_role(name="viewer", workspace="ws1")
    store.add_role_permission(role.id, "experiment", "1", "READ")
    store.assign_role_to_user(user.id, role.id)

    result = store.get_role_permission_for_resource(user.id, "experiment", "1", "ws1")
    assert result == READ


def test_get_role_permission_no_match(store, user):
    role = store.create_role(name="viewer", workspace="ws1")
    store.add_role_permission(role.id, "experiment", "1", "READ")
    store.assign_role_to_user(user.id, role.id)

    result = store.get_role_permission_for_resource(user.id, "experiment", "999", "ws1")
    assert result is None


def test_get_role_permission_wildcard_match(store, user):
    role = store.create_role(name="viewer", workspace="ws1")
    store.add_role_permission(role.id, "experiment", "*", "EDIT")
    store.assign_role_to_user(user.id, role.id)

    result = store.get_role_permission_for_resource(user.id, "experiment", "any-id", "ws1")
    assert result == EDIT


def test_get_role_permission_union_of_multiple_roles(store, user):
    r1 = store.create_role(name="viewer", workspace="ws1")
    store.add_role_permission(r1.id, "experiment", "1", "READ")
    store.assign_role_to_user(user.id, r1.id)

    r2 = store.create_role(name="editor", workspace="ws1")
    store.add_role_permission(r2.id, "experiment", "1", "EDIT")
    store.assign_role_to_user(user.id, r2.id)

    result = store.get_role_permission_for_resource(user.id, "experiment", "1", "ws1")
    assert result == EDIT


def test_get_role_permission_wildcard_and_specific_union(store, user):
    role = store.create_role(name="mixed", workspace="ws1")
    store.add_role_permission(role.id, "experiment", "*", "READ")
    store.add_role_permission(role.id, "experiment", "1", "EDIT")
    store.assign_role_to_user(user.id, role.id)

    # Experiment 1 gets EDIT (higher of READ wildcard and EDIT specific)
    result = store.get_role_permission_for_resource(user.id, "experiment", "1", "ws1")
    assert result == EDIT

    # Other experiments get READ from wildcard
    result = store.get_role_permission_for_resource(user.id, "experiment", "999", "ws1")
    assert result == READ


def test_get_role_permission_workspace_admin(store, user):
    role = store.create_role(name="ws-admin", workspace="ws1")
    store.add_role_permission(role.id, "workspace", "*", "MANAGE")
    store.assign_role_to_user(user.id, role.id)

    # Workspace-wide MANAGE applies to any resource type.
    result = store.get_role_permission_for_resource(user.id, "experiment", "any-id", "ws1")
    assert result == MANAGE

    result = store.get_role_permission_for_resource(user.id, "registered_model", "m1", "ws1")
    assert result == MANAGE


def test_workspace_permission_applies_across_resource_types(store, user):
    role = store.create_role(name="reader", workspace="ws1")
    store.add_role_permission(role.id, "workspace", "*", "READ")
    store.assign_role_to_user(user.id, role.id)

    # READ at the workspace level grants READ on every resource type in the workspace.
    assert store.get_role_permission_for_resource(user.id, "experiment", "1", "ws1") == READ
    assert store.get_role_permission_for_resource(user.id, "registered_model", "m1", "ws1") == READ
    assert store.get_role_permission_for_resource(user.id, "gateway_endpoint", "e1", "ws1") == READ


def test_workspace_permission_respects_union_with_specific(store, user):
    role = store.create_role(name="mixed", workspace="ws1")
    store.add_role_permission(role.id, "workspace", "*", "READ")
    store.add_role_permission(role.id, "experiment", "42", "EDIT")
    store.assign_role_to_user(user.id, role.id)

    # Experiment 42: max(workspace READ, specific EDIT) = EDIT
    assert store.get_role_permission_for_resource(user.id, "experiment", "42", "ws1") == EDIT
    # Other experiments: just workspace READ
    assert store.get_role_permission_for_resource(user.id, "experiment", "99", "ws1") == READ


def test_is_workspace_admin(store, user):
    role = store.create_role(name="ws-admin", workspace="ws1")
    store.add_role_permission(role.id, "workspace", "*", "MANAGE")
    store.assign_role_to_user(user.id, role.id)

    assert store.is_workspace_admin(user.id, "ws1") is True
    assert store.is_workspace_admin(user.id, "ws2") is False


def test_is_workspace_admin_requires_manage(store, user):
    # A non-MANAGE workspace permission does not make the user a WP admin.
    role = store.create_role(name="ws-reader", workspace="ws1")
    store.add_role_permission(role.id, "workspace", "*", "READ")
    store.assign_role_to_user(user.id, role.id)

    assert store.is_workspace_admin(user.id, "ws1") is False


def test_list_role_grants_for_user_in_workspace(store, user):
    # Role with specific + wildcard experiment grants + workspace-wide grant.
    role = store.create_role(name="multi", workspace="ws1")
    store.add_role_permission(role.id, "experiment", "42", "EDIT")
    store.add_role_permission(role.id, "experiment", "*", "READ")
    store.add_role_permission(role.id, "workspace", "*", "READ")
    # Unrelated grant on another resource type.
    store.add_role_permission(role.id, "registered_model", "*", "MANAGE")
    store.assign_role_to_user(user.id, role.id)

    grants = store.list_role_grants_for_user_in_workspace(user.id, "ws1", "experiment")
    # Should include specific experiment grant, wildcard experiment grant,
    # and the workspace-wide grant. Should NOT include the registered_model grant.
    assert sorted(grants) == sorted([("42", "EDIT"), ("*", "READ"), ("*", "READ")])


def test_list_role_grants_for_user_in_workspace_cross_workspace(store, user):
    # Grants in ws2 should not surface when querying ws1.
    role = store.create_role(name="other-ws", workspace="ws2")
    store.add_role_permission(role.id, "experiment", "99", "EDIT")
    store.assign_role_to_user(user.id, role.id)

    assert store.list_role_grants_for_user_in_workspace(user.id, "ws1", "experiment") == []


def test_list_role_grants_for_user_in_workspace_no_roles(store, user):
    assert store.list_role_grants_for_user_in_workspace(user.id, "ws1", "experiment") == []


def test_list_role_grants_for_user_in_workspace_rejects_invalid_resource_type(store, user):
    with pytest.raises(MlflowException, match="Invalid resource type"):
        store.list_role_grants_for_user_in_workspace(user.id, "ws1", "not_a_type")


def test_list_workspace_admin_workspaces(store, user):
    # WP admin in ws1 + ws3, regular member in ws2.
    admin_ws1 = store.create_role(name="wa1", workspace="ws1")
    store.add_role_permission(admin_ws1.id, "workspace", "*", "MANAGE")
    store.assign_role_to_user(user.id, admin_ws1.id)
    admin_ws3 = store.create_role(name="wa3", workspace="ws3")
    store.add_role_permission(admin_ws3.id, "workspace", "*", "MANAGE")
    store.assign_role_to_user(user.id, admin_ws3.id)
    member_ws2 = store.create_role(name="mem", workspace="ws2")
    store.add_role_permission(member_ws2.id, "experiment", "*", "READ")
    store.assign_role_to_user(user.id, member_ws2.id)

    assert store.list_workspace_admin_workspaces(user.id) == {"ws1", "ws3"}


def test_list_workspace_admin_workspaces_ignores_non_manage(store, user):
    # A workspace-scope grant with a non-MANAGE permission should not count.
    role = store.create_role(name="reader", workspace="ws1")
    store.add_role_permission(role.id, "workspace", "*", "READ")
    store.assign_role_to_user(user.id, role.id)

    assert store.list_workspace_admin_workspaces(user.id) == set()


# ---- Legacy workspace_permissions as workspace-admin source ----
#
# Pre-RBAC operators relied on `workspace_permissions` MANAGE to convey workspace-wide
# admin authority. The workspace-admin helpers must still recognize that grant,
# otherwise operators mid-migration (or just not yet using roles) silently lose admin
# status behind RBAC-aware validators.


def test_is_workspace_admin_honors_legacy_workspace_permissions(store, user):
    store.set_workspace_permission("ws1", user.username, "MANAGE")

    assert store.is_workspace_admin(user.id, "ws1") is True
    assert store.is_workspace_admin(user.id, "ws2") is False


def test_is_workspace_admin_ignores_non_manage_legacy(store, user):
    store.set_workspace_permission("ws1", user.username, "READ")

    assert store.is_workspace_admin(user.id, "ws1") is False


def test_list_workspace_admin_workspaces_unions_role_and_legacy(store, user):
    # Role admin in ws1, legacy MANAGE in ws2, legacy READ in ws3 (should not count).
    role = store.create_role(name="wa1", workspace="ws1")
    store.add_role_permission(role.id, "workspace", "*", "MANAGE")
    store.assign_role_to_user(user.id, role.id)
    store.set_workspace_permission("ws2", user.username, "MANAGE")
    store.set_workspace_permission("ws3", user.username, "READ")

    assert store.list_workspace_admin_workspaces(user.id) == {"ws1", "ws2"}


def test_is_workspace_admin_of_any_of_users_workspaces_legacy_admin(store, user, user2):
    # Admin authority via legacy, target presence via role.
    store.set_workspace_permission("ws1", user.username, "MANAGE")
    target_role = store.create_role(name="member", workspace="ws1")
    store.add_role_permission(target_role.id, "experiment", "*", "READ")
    store.assign_role_to_user(user2.id, target_role.id)

    assert store.is_workspace_admin_of_any_of_users_workspaces(user.id, user2.id) is True


def test_is_workspace_admin_of_any_of_users_workspaces_legacy_target(store, user, user2):
    # Admin authority via role, target presence via legacy.
    admin_role = store.create_role(name="wa", workspace="ws1")
    store.add_role_permission(admin_role.id, "workspace", "*", "MANAGE")
    store.assign_role_to_user(user.id, admin_role.id)
    store.set_workspace_permission("ws1", user2.username, "READ")

    assert store.is_workspace_admin_of_any_of_users_workspaces(user.id, user2.id) is True


def test_is_workspace_admin_of_any_of_users_workspaces_no_overlap(store, user, user2):
    # Admin in ws1, target present only in ws2 → no intersection.
    store.set_workspace_permission("ws1", user.username, "MANAGE")
    store.set_workspace_permission("ws2", user2.username, "READ")

    assert store.is_workspace_admin_of_any_of_users_workspaces(user.id, user2.id) is False


def test_get_role_permission_does_not_cross_workspace(store, user):
    role = store.create_role(name="viewer", workspace="ws1")
    store.add_role_permission(role.id, "experiment", "*", "READ")
    store.assign_role_to_user(user.id, role.id)

    # Should not apply in ws2
    result = store.get_role_permission_for_resource(user.id, "experiment", "1", "ws2")
    assert result is None


def test_get_role_permission_different_resource_types(store, user):
    role = store.create_role(name="viewer", workspace="ws1")
    store.add_role_permission(role.id, "experiment", "*", "READ")
    store.assign_role_to_user(user.id, role.id)

    # Should not match registered_model
    result = store.get_role_permission_for_resource(user.id, "registered_model", "m1", "ws1")
    assert result is None

    # Should match experiment
    result = store.get_role_permission_for_resource(user.id, "experiment", "1", "ws1")
    assert result == READ


@pytest.mark.parametrize(
    ("perms", "expected"),
    [
        ([("experiment", "1", "READ"), ("experiment", "1", "MANAGE")], MANAGE),
        ([("experiment", "1", "USE"), ("experiment", "1", "EDIT")], EDIT),
        ([("experiment", "*", "READ"), ("experiment", "1", "USE")], USE),
    ],
)
def test_get_role_permission_picks_highest(store, user, perms, expected):
    for i, (rtype, pattern, perm) in enumerate(perms):
        role = store.create_role(name=f"role-{i}", workspace="ws1")
        store.add_role_permission(role.id, rtype, pattern, perm)
        store.assign_role_to_user(user.id, role.id)

    result = store.get_role_permission_for_resource(user.id, "experiment", "1", "ws1")
    assert result == expected
