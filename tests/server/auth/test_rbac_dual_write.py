import pytest

from mlflow.exceptions import MlflowException
from mlflow.server.auth.permissions import EDIT, MANAGE, READ, get_permission
from mlflow.server.auth.sqlalchemy_store import SqlAlchemyStore
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

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


def _assert_role_permission(store, user_id, resource_type, resource_id, expected):
    resolved = store.get_role_permission_for_resource(
        user_id, resource_type, resource_id, DEFAULT_WORKSPACE_NAME
    )
    if expected is None:
        assert resolved is None
    else:
        assert resolved == expected


# ---- Per-resource dual-write invariants ----
#
# For each of the six per-resource permission tables, creating / updating / deleting a grant
# via the legacy API must keep `get_role_permission_for_resource` in sync with the direct
# lookup so later phases can flip reads over without changing semantics.


def test_experiment_permission_dual_write(store, user):
    store.create_experiment_permission("exp1", user.username, READ.name)
    _assert_role_permission(store, user.id, "experiment", "exp1", READ)

    store.update_experiment_permission("exp1", user.username, EDIT.name)
    _assert_role_permission(store, user.id, "experiment", "exp1", EDIT)

    store.delete_experiment_permission("exp1", user.username)
    _assert_role_permission(store, user.id, "experiment", "exp1", None)


def test_registered_model_permission_dual_write(store, user):
    store.create_registered_model_permission("model_a", user.username, READ.name)
    _assert_role_permission(store, user.id, "registered_model", "model_a", READ)

    store.update_registered_model_permission("model_a", user.username, MANAGE.name)
    _assert_role_permission(store, user.id, "registered_model", "model_a", MANAGE)

    store.delete_registered_model_permission("model_a", user.username)
    _assert_role_permission(store, user.id, "registered_model", "model_a", None)


def test_registered_model_bulk_delete_unmirrors(store, user, user2):
    store.create_registered_model_permission("model_a", user.username, READ.name)
    store.create_registered_model_permission("model_a", user2.username, EDIT.name)
    _assert_role_permission(store, user.id, "registered_model", "model_a", READ)
    _assert_role_permission(store, user2.id, "registered_model", "model_a", EDIT)

    store.delete_registered_model_permissions("model_a")

    _assert_role_permission(store, user.id, "registered_model", "model_a", None)
    _assert_role_permission(store, user2.id, "registered_model", "model_a", None)


def test_registered_model_rename_updates_mirror(store, user):
    store.create_registered_model_permission("old_name", user.username, EDIT.name)
    _assert_role_permission(store, user.id, "registered_model", "old_name", EDIT)

    store.rename_registered_model_permissions("old_name", "new_name")

    _assert_role_permission(store, user.id, "registered_model", "old_name", None)
    _assert_role_permission(store, user.id, "registered_model", "new_name", EDIT)


def test_scorer_permission_dual_write(store, user):
    store.create_scorer_permission("exp1", "scorer_a", user.username, READ.name)
    _assert_role_permission(store, user.id, "scorer", "exp1/scorer_a", READ)

    store.update_scorer_permission("exp1", "scorer_a", user.username, EDIT.name)
    _assert_role_permission(store, user.id, "scorer", "exp1/scorer_a", EDIT)

    store.delete_scorer_permission("exp1", "scorer_a", user.username)
    _assert_role_permission(store, user.id, "scorer", "exp1/scorer_a", None)


def test_scorer_bulk_delete_unmirrors(store, user, user2):
    store.create_scorer_permission("exp1", "scorer_a", user.username, READ.name)
    store.create_scorer_permission("exp1", "scorer_a", user2.username, EDIT.name)
    store.delete_scorer_permissions_for_scorer("exp1", "scorer_a")
    _assert_role_permission(store, user.id, "scorer", "exp1/scorer_a", None)
    _assert_role_permission(store, user2.id, "scorer", "exp1/scorer_a", None)


def test_gateway_secret_permission_dual_write(store, user):
    store.create_gateway_secret_permission("secret1", user.username, READ.name)
    _assert_role_permission(store, user.id, "gateway_secret", "secret1", READ)

    store.update_gateway_secret_permission("secret1", user.username, MANAGE.name)
    _assert_role_permission(store, user.id, "gateway_secret", "secret1", MANAGE)

    store.delete_gateway_secret_permission("secret1", user.username)
    _assert_role_permission(store, user.id, "gateway_secret", "secret1", None)


def test_gateway_secret_bulk_delete_unmirrors(store, user, user2):
    store.create_gateway_secret_permission("secret1", user.username, READ.name)
    store.create_gateway_secret_permission("secret1", user2.username, EDIT.name)
    store.delete_gateway_secret_permissions_for_secret("secret1")
    _assert_role_permission(store, user.id, "gateway_secret", "secret1", None)
    _assert_role_permission(store, user2.id, "gateway_secret", "secret1", None)


def test_gateway_endpoint_permission_dual_write(store, user):
    store.create_gateway_endpoint_permission("endpoint1", user.username, READ.name)
    _assert_role_permission(store, user.id, "gateway_endpoint", "endpoint1", READ)

    store.update_gateway_endpoint_permission("endpoint1", user.username, EDIT.name)
    _assert_role_permission(store, user.id, "gateway_endpoint", "endpoint1", EDIT)

    store.delete_gateway_endpoint_permission("endpoint1", user.username)
    _assert_role_permission(store, user.id, "gateway_endpoint", "endpoint1", None)


def test_gateway_endpoint_bulk_delete_unmirrors(store, user, user2):
    store.create_gateway_endpoint_permission("endpoint1", user.username, READ.name)
    store.create_gateway_endpoint_permission("endpoint1", user2.username, EDIT.name)
    store.delete_gateway_endpoint_permissions_for_endpoint("endpoint1")
    _assert_role_permission(store, user.id, "gateway_endpoint", "endpoint1", None)
    _assert_role_permission(store, user2.id, "gateway_endpoint", "endpoint1", None)


def test_gateway_model_definition_permission_dual_write(store, user):
    store.create_gateway_model_definition_permission("model1", user.username, READ.name)
    _assert_role_permission(store, user.id, "gateway_model_definition", "model1", READ)

    store.update_gateway_model_definition_permission("model1", user.username, MANAGE.name)
    _assert_role_permission(store, user.id, "gateway_model_definition", "model1", MANAGE)

    store.delete_gateway_model_definition_permission("model1", user.username)
    _assert_role_permission(store, user.id, "gateway_model_definition", "model1", None)


def test_gateway_model_definition_bulk_delete_unmirrors(store, user, user2):
    store.create_gateway_model_definition_permission("model1", user.username, READ.name)
    store.create_gateway_model_definition_permission("model1", user2.username, EDIT.name)
    store.delete_gateway_model_definition_permissions_for_model_definition("model1")
    _assert_role_permission(store, user.id, "gateway_model_definition", "model1", None)
    _assert_role_permission(store, user2.id, "gateway_model_definition", "model1", None)


# ---- Synthetic role bookkeeping ----


def test_synthetic_role_is_created_and_reused(store, user):
    store.create_experiment_permission("exp1", user.username, READ.name)
    store.create_registered_model_permission("model1", user.username, EDIT.name)

    # A single synthetic role per (user, workspace) holds all mirrored grants.
    roles = store.list_roles(DEFAULT_WORKSPACE_NAME)
    synthetic = [r for r in roles if r.name == f"__user_{user.id}__"]
    assert len(synthetic) == 1
    patterns = {(p.resource_type, p.resource_pattern) for p in synthetic[0].permissions}
    assert ("experiment", "exp1") in patterns
    assert ("registered_model", "model1") in patterns


def test_synthetic_role_is_user_scoped(store, user, user2):
    store.create_experiment_permission("exp1", user.username, READ.name)
    store.create_experiment_permission("exp1", user2.username, EDIT.name)

    # Each user gets their own synthetic role — grants don't bleed across users.
    _assert_role_permission(store, user.id, "experiment", "exp1", READ)
    _assert_role_permission(store, user2.id, "experiment", "exp1", EDIT)


def test_delete_user_removes_synthetic_role(store, user):
    # Create a grant to force the synthetic role into existence, then remove the grant so
    # only the (now empty) synthetic role + assignment remain. This avoids tripping a
    # pre-existing FK issue on the legacy permission tables when a user is deleted while
    # still holding direct grants — that issue is orthogonal to the dual-write cleanup and
    # will be addressed in its own fix.
    store.create_experiment_permission("exp1", user.username, READ.name)
    store.delete_experiment_permission("exp1", user.username)
    user_id = user.id
    assert any(r.name == f"__user_{user_id}__" for r in store.list_roles(DEFAULT_WORKSPACE_NAME))

    store.delete_user(user.username)

    assert not any(
        r.name == f"__user_{user_id}__" for r in store.list_roles(DEFAULT_WORKSPACE_NAME)
    )


def test_legacy_and_role_views_agree(store, user):
    # The goal of dual-write is that whatever the legacy read returns, the role-based
    # resolution returns the same Permission object. Exercise that directly across a
    # sequence of grant mutations.
    for permission_name in (READ.name, EDIT.name, MANAGE.name):
        if store.list_experiment_permissions(user.username):
            store.update_experiment_permission("exp1", user.username, permission_name)
        else:
            store.create_experiment_permission("exp1", user.username, permission_name)
        legacy = store.get_experiment_permission("exp1", user.username).permission
        _assert_role_permission(store, user.id, "experiment", "exp1", get_permission(legacy))


# ---- Synthetic-role scoping invariants ----
#
# The synthetic `__user_<id>__` namespace is reserved. Bulk cleanup/rename helpers must
# never touch admin-created roles even when the (resource_type, resource_pattern) match.


def test_bulk_delete_does_not_touch_admin_created_roles(store, user):
    # Admin creates a regular role with a registered_model grant; user also has a
    # mirrored registered_model permission. Bulk delete must only remove the mirror.
    admin_role = store.create_role(name="team-editor", workspace=DEFAULT_WORKSPACE_NAME)
    store.add_role_permission(admin_role.id, "registered_model", "model_a", EDIT.name)
    store.create_registered_model_permission("model_a", user.username, READ.name)

    store.delete_registered_model_permissions("model_a")

    _assert_role_permission(store, user.id, "registered_model", "model_a", None)
    admin_perms = store.list_role_permissions(admin_role.id)
    assert len(admin_perms) == 1
    assert admin_perms[0].resource_pattern == "model_a"
    assert admin_perms[0].permission == EDIT.name


def test_bulk_rename_does_not_touch_admin_created_roles(store, user):
    admin_role = store.create_role(name="team-editor", workspace=DEFAULT_WORKSPACE_NAME)
    store.add_role_permission(admin_role.id, "registered_model", "old_name", EDIT.name)
    store.create_registered_model_permission("old_name", user.username, READ.name)

    store.rename_registered_model_permissions("old_name", "new_name")

    # User's mirror was renamed.
    _assert_role_permission(store, user.id, "registered_model", "new_name", READ)
    _assert_role_permission(store, user.id, "registered_model", "old_name", None)
    # Admin-created role's permission pattern unchanged.
    admin_perms = store.list_role_permissions(admin_role.id)
    assert len(admin_perms) == 1
    assert admin_perms[0].resource_pattern == "old_name"


def test_scorer_bulk_delete_does_not_touch_admin_created_roles(store, user):
    # Workspace=None bulk path (used by scorer/gateway_*) must also be synthetic-only.
    admin_role = store.create_role(name="scorer-admin", workspace=DEFAULT_WORKSPACE_NAME)
    store.add_role_permission(admin_role.id, "scorer", "exp1/judge", EDIT.name)
    store.create_scorer_permission("exp1", "judge", user.username, READ.name)

    store.delete_scorer_permissions_for_scorer("exp1", "judge")

    _assert_role_permission(store, user.id, "scorer", "exp1/judge", None)
    admin_perms = store.list_role_permissions(admin_role.id)
    assert len(admin_perms) == 1
    assert admin_perms[0].permission == EDIT.name


def test_create_role_rejects_synthetic_name_pattern(store):
    with pytest.raises(MlflowException, match="reserved synthetic pattern"):
        store.create_role(name="__user_42__", workspace=DEFAULT_WORKSPACE_NAME)
    # Non-integer suffix is allowed — only the exact pattern is reserved.
    store.create_role(name="__user_foo__", workspace=DEFAULT_WORKSPACE_NAME)


def test_update_role_rejects_synthetic_name_pattern(store):
    role = store.create_role(name="normal", workspace=DEFAULT_WORKSPACE_NAME)
    with pytest.raises(MlflowException, match="reserved synthetic pattern"):
        store.update_role(role.id, name="__user_7__")


def test_is_synthetic_role_name_rejects_edge_cases():
    assert SqlAlchemyStore._is_synthetic_role_name("__user_1__") is True
    assert SqlAlchemyStore._is_synthetic_role_name("__user_99999__") is True
    # Wrong surroundings or missing id:
    assert SqlAlchemyStore._is_synthetic_role_name("__user__") is False
    assert SqlAlchemyStore._is_synthetic_role_name("__user_abc__") is False
    assert SqlAlchemyStore._is_synthetic_role_name("user_1") is False
    assert SqlAlchemyStore._is_synthetic_role_name("_user_1__") is False
    assert SqlAlchemyStore._is_synthetic_role_name("__user_1_") is False
    assert SqlAlchemyStore._is_synthetic_role_name(None) is False
