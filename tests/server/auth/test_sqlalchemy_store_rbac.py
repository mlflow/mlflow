from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, BrokenBarrierError

import pytest
from sqlalchemy.dialects import mysql

from mlflow.exceptions import MlflowException
from mlflow.server.auth.db.models import SqlRole, SqlRolePermission, SqlUserRoleAssignment
from mlflow.server.auth.entities import Role, RolePermission, UserRoleAssignment
from mlflow.server.auth.permissions import (
    EDIT,
    MANAGE,
    READ,
    RESOURCE_TYPE_AGENT_PLUGIN,
    RESOURCE_TYPE_EXPERIMENT,
    RESOURCE_TYPE_SKILL,
    RESOURCE_TYPE_WORKSPACE,
    USE,
    VALID_RESOURCE_TYPES,
)

# Every concrete resource type the resolver accepts, excluding the special
# ``"workspace"`` (admin-only grant form) and ``"*"`` (workspace-wide grant
# form). Those two carry their own validation rules and are exercised by
# scope-specific tests rather than the shared parametrised matrix below.
_CONCRETE_RESOURCE_TYPES = sorted(VALID_RESOURCE_TYPES - {"workspace", "*"})
_SKILL_REGISTRY_RESOURCE_TYPES = {RESOURCE_TYPE_SKILL, RESOURCE_TYPE_AGENT_PLUGIN}
_COMMON_CONCRETE_RESOURCE_TYPES = sorted(
    set(_CONCRETE_RESOURCE_TYPES) - _SKILL_REGISTRY_RESOURCE_TYPES
)
_RESOURCE_GRANT_CASES = [
    (resource_type, permission.name, permission)
    for resource_type in _COMMON_CONCRETE_RESOURCE_TYPES
    for permission in (READ, USE, EDIT, MANAGE)
] + [
    (resource_type, permission.name, permission)
    for resource_type in sorted(_SKILL_REGISTRY_RESOURCE_TYPES)
    for permission in (READ, EDIT, MANAGE)
]
from mlflow.server.auth.sqlalchemy_store import SqlAlchemyStore
from mlflow.store.db.db_types import MYSQL
from mlflow.store.tracking.dbmodels.models import SqlSkill
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore as TrackingSqlAlchemyStore
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


# Sample names matching the reserved ``__user_`` prefix. Includes both the
# strict synthetic pattern (``__user_<digits>__``) and looser variants
# (admin-author-able lookalikes) so the broader prefix guard is exercised.
_RESERVED_USER_PREFIX_NAMES = [
    "__user_1__",
    "__user_42__",
    "__user_999999__",
    "__user_admin",
    "__user_alice",
    "__user_foo_bar",
    "__user_",
]


@pytest.mark.parametrize("name", _RESERVED_USER_PREFIX_NAMES)
def test_create_role_rejects_reserved_user_prefix(store, name):
    # The ``__user_`` prefix is reserved for synthetic per-user roles created
    # internally by ``grant_user_permission``. Allowing an operator to author
    # a role with that prefix lets them collide with (or shadow) a real
    # user's synthetic role and silently attach grants to that user — a
    # privilege-escalation footgun. The guard covers the whole prefix, not
    # just the strict ``__user_<digits>__`` pattern, so future changes to the
    # synthetic naming scheme can't be hijacked by an existing collision.
    with pytest.raises(MlflowException, match="reserved '__user_' prefix"):
        store.create_role(name=name, workspace="ws1")


@pytest.mark.parametrize("name", _RESERVED_USER_PREFIX_NAMES)
def test_update_role_rejects_reserved_user_prefix(store, name):
    # The same reservation must apply when *renaming* an existing role —
    # otherwise the create-time guard could be bypassed by creating with a
    # normal name and then renaming into the reserved prefix.
    role = store.create_role(name="viewer", workspace="ws1")
    with pytest.raises(MlflowException, match="reserved '__user_' prefix"):
        store.update_role(role.id, name=name)


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
    store.create_role(name="other", workspace="ws3")

    # Single workspace.
    ws1_roles = store.list_roles(["ws1"])
    assert {r.name for r in ws1_roles} == {"viewer", "editor"}

    # Subset across two workspaces.
    roles = store.list_roles(["ws1", "ws2"])
    assert {(r.workspace, r.name) for r in roles} == {
        ("ws1", "viewer"),
        ("ws1", "editor"),
        ("ws2", "viewer"),
    }

    # Empty iterable is interpreted literally — no roles.
    assert store.list_roles([]) == []

    # ``None`` (default) lists across the whole system — the admin path.
    all_roles = store.list_roles()
    assert {(r.workspace, r.name) for r in all_roles} == {
        ("ws1", "viewer"),
        ("ws1", "editor"),
        ("ws2", "viewer"),
        ("ws3", "other"),
    }


def test_list_users_with_roles_eager_loads_in_one_batch(store, user, user2):
    # Two roles for the first user, one for the second; one extra unassigned
    # role just to confirm we only return roles tied to a user.
    r1 = store.create_role(name="viewer", workspace="ws1")
    r2 = store.create_role(name="editor", workspace="ws2")
    r3 = store.create_role(name="other", workspace="ws3")
    store.create_role(name="unassigned", workspace="ws1")

    store.assign_role_to_user(user.id, r1.id)
    store.assign_role_to_user(user.id, r2.id)
    store.assign_role_to_user(user2.id, r3.id)

    users_with_roles = store.list_users_with_roles()
    by_username = {u.username: roles for u, roles in users_with_roles}

    assert {(r.workspace, r.name) for r in by_username[user.username]} == {
        ("ws1", "viewer"),
        ("ws2", "editor"),
    }
    assert {(r.workspace, r.name) for r in by_username[user2.username]} == {("ws3", "other")}


def test_list_user_present_workspaces(store, user):
    # No assignments → empty set.
    assert store.list_user_present_workspaces(user.id) == set()

    r1 = store.create_role(name="viewer", workspace="ws1")
    r2 = store.create_role(name="editor", workspace="ws2")
    r3 = store.create_role(name="other", workspace="ws2")

    store.assign_role_to_user(user.id, r1.id)
    store.assign_role_to_user(user.id, r2.id)
    store.assign_role_to_user(user.id, r3.id)

    assert store.list_user_present_workspaces(user.id) == {"ws1", "ws2"}


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
    assert store.list_roles(["ws1"]) == []
    assert len(store.list_roles(["ws2"])) == 1


def test_delete_roles_for_workspace_cascades(store, user):
    role = store.create_role(name="r1", workspace="ws1")
    store.add_role_permission(role.id, "experiment", "*", "READ")
    store.assign_role_to_user(user.id, role.id)

    store.delete_roles_for_workspace("ws1")

    assert store.list_roles(["ws1"]) == []
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


# ---- Session-aware per-user grant helpers ----


def test_grant_user_permissions_in_session_commits_with_outer_transaction(store, user):
    with store.ManagedSessionMaker(read_only=False) as session:
        store.grant_user_permissions_in_session(
            session,
            user.username,
            [
                (RESOURCE_TYPE_SKILL, "demo-skill", MANAGE.name),
                (RESOURCE_TYPE_AGENT_PLUGIN, "@acme/demo-plugin", MANAGE.name),
            ],
        )

    assert (
        store.get_role_permission_for_resource(
            user.id, RESOURCE_TYPE_SKILL, "demo-skill", DEFAULT_WORKSPACE_NAME
        )
        == MANAGE
    )
    assert (
        store.get_role_permission_for_resource(
            user.id,
            RESOURCE_TYPE_AGENT_PLUGIN,
            "@acme/demo-plugin",
            DEFAULT_WORKSPACE_NAME,
        )
        == MANAGE
    )


def test_grant_user_permissions_in_session_rolls_back_with_outer_transaction(store, user):
    def grant_then_abort():
        with store.ManagedSessionMaker(read_only=False) as session:
            store.grant_user_permissions_in_session(
                session,
                user.username,
                [(RESOURCE_TYPE_SKILL, "demo-skill", MANAGE.name)],
            )
            raise MlflowException("abort grant transaction")

    with pytest.raises(MlflowException, match="abort grant transaction"):
        grant_then_abort()

    assert (
        store.get_role_permission_for_resource(
            user.id, RESOURCE_TYPE_SKILL, "demo-skill", DEFAULT_WORKSPACE_NAME
        )
        is None
    )


def test_aborted_grant_rolls_back_created_role_and_assignment(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_WORKSPACES", "false")
    store = SqlAlchemyStore()
    store.init_db(f"sqlite:///{tmp_path / 'auth.db'}")
    user = store.create_user("alice", "strong-password")

    def grant_then_abort():
        with store.ManagedSessionMaker(read_only=False) as session:
            store.grant_user_permissions_in_session(
                session,
                user.username,
                [(RESOURCE_TYPE_SKILL, "demo-skill", MANAGE.name)],
            )
            raise MlflowException("abort")

    try:
        with pytest.raises(MlflowException, match="abort"):
            grant_then_abort()

        with store.ManagedSessionMaker() as session:
            assert session.query(SqlRole).count() == 0
            assert session.query(SqlUserRoleAssignment).count() == 0
            assert session.query(SqlRolePermission).count() == 0
    finally:
        store.engine.dispose()


@pytest.mark.parametrize(
    ("case_name", "resource_type", "setup_permission", "grant_permission", "grant_method"),
    [
        (
            "insert",
            RESOURCE_TYPE_SKILL,
            None,
            MANAGE.name,
            "grant_user_resource_permission",
        ),
        (
            "update",
            RESOURCE_TYPE_SKILL,
            READ.name,
            MANAGE.name,
            "grant_user_permission",
        ),
        (
            "creator_grant",
            RESOURCE_TYPE_EXPERIMENT,
            None,
            MANAGE.name,
            "grant_user_permission",
        ),
    ],
)
def test_concurrent_sqlite_user_grants_do_not_lose_one_write(
    tmp_path,
    monkeypatch,
    case_name,
    resource_type,
    setup_permission,
    grant_permission,
    grant_method,
):
    monkeypatch.setenv("MLFLOW_ENABLE_WORKSPACES", "false")
    store = SqlAlchemyStore()
    store.init_db(f"sqlite:///{tmp_path / f'{case_name}.db'}")
    user = store.create_user("alice", "strong-password")
    resource_patterns = [f"{case_name}-first", f"{case_name}-second"]

    try:
        store.grant_user_permission(user.username, RESOURCE_TYPE_SKILL, "seed-role", READ.name)
        if setup_permission is not None:
            for resource_pattern in resource_patterns:
                store.grant_user_permission(
                    user.username,
                    resource_type,
                    resource_pattern,
                    setup_permission,
                )

        start_barrier = Barrier(2)
        lookup_barrier = Barrier(2)
        original_lookup = store._get_role_permission_in_session

        def lookup(session, role_id, resource_type, resource_pattern, *, for_update=False):
            result = original_lookup(
                session,
                role_id,
                resource_type,
                resource_pattern,
                for_update=for_update,
            )
            try:
                lookup_barrier.wait(timeout=1)
            except BrokenBarrierError:
                pass
            return result

        monkeypatch.setattr(store, "_get_role_permission_in_session", lookup)

        def grant(resource_pattern):
            start_barrier.wait(timeout=10)
            try:
                getattr(store, grant_method)(
                    user.username,
                    resource_type,
                    resource_pattern,
                    grant_permission,
                )
                return "ok"
            except MlflowException as error:
                return error.error_code

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(grant, resource_patterns))

        assert results == ["ok", "ok"]
        for resource_pattern in resource_patterns:
            assert (
                store.get_role_permission_for_resource(
                    user.id,
                    resource_type,
                    resource_pattern,
                    DEFAULT_WORKSPACE_NAME,
                )
                == MANAGE
            )
    finally:
        store.engine.dispose()


def test_mysql_upsert_recovery_role_permission_lookup_uses_locking_read(store):
    store.db_type = MYSQL
    with store.ManagedSessionMaker(read_only=False) as session:
        query = store._role_permission_query(
            session,
            role_id=1,
            resource_type=RESOURCE_TYPE_SKILL,
            resource_pattern="snapshot-race",
            for_update=True,
        )

    assert "FOR UPDATE" in str(query.statement.compile(dialect=mysql.dialect()))


def test_session_grants_reject_tracking_session_from_separate_database(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_WORKSPACES", "false")
    auth_store = SqlAlchemyStore()
    auth_store.init_db(f"sqlite:///{tmp_path / 'auth.db'}")
    tracking_store = TrackingSqlAlchemyStore(
        f"sqlite:///{tmp_path / 'tracking.db'}",
        str(tmp_path / "artifacts"),
    )
    auth_store.create_user("alice", "strong-password")

    try:
        with pytest.raises(MlflowException, match="same database"):
            with tracking_store.ManagedSessionMaker(read_only=False) as session:
                auth_store.grant_user_permissions_in_session(
                    session,
                    "alice",
                    [(RESOURCE_TYPE_SKILL, "separate-db-skill", MANAGE.name)],
                )
    finally:
        auth_store.engine.dispose()
        tracking_store.engine.dispose()


def test_session_grants_share_tracking_transaction_when_databases_are_colocated(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("MLFLOW_ENABLE_WORKSPACES", "false")
    db_uri = f"sqlite:///{tmp_path / 'shared.db'}"
    auth_store = SqlAlchemyStore()
    auth_store.init_db(db_uri)
    tracking_store = TrackingSqlAlchemyStore(db_uri, str(tmp_path / "artifacts"))
    user = auth_store.create_user("alice", "strong-password")

    def create_skill_and_grant_then_abort():
        with tracking_store.ManagedSessionMaker(read_only=False) as session:
            session.add(
                SqlSkill(
                    workspace=DEFAULT_WORKSPACE_NAME,
                    organization="",
                    name="shared-db-skill",
                )
            )
            auth_store.grant_user_permissions_in_session(
                session,
                user.username,
                [(RESOURCE_TYPE_SKILL, "shared-db-skill", MANAGE.name)],
            )
            raise MlflowException("abort shared transaction")

    try:
        with pytest.raises(MlflowException, match="abort shared transaction"):
            create_skill_and_grant_then_abort()

        with tracking_store.ManagedSessionMaker() as session:
            assert (
                session.query(SqlSkill).filter(SqlSkill.name == "shared-db-skill").first() is None
            )
        assert (
            auth_store.get_role_permission_for_resource(
                user.id,
                RESOURCE_TYPE_SKILL,
                "shared-db-skill",
                DEFAULT_WORKSPACE_NAME,
            )
            is None
        )
    finally:
        auth_store.engine.dispose()
        tracking_store.engine.dispose()


def test_grant_user_permissions_in_session_rolls_back_partial_batch_on_duplicate(store, user):
    store.grant_user_permission(user.username, RESOURCE_TYPE_SKILL, "existing-skill", READ.name)

    with pytest.raises(MlflowException, match="already exists"):
        with store.ManagedSessionMaker(read_only=False) as session:
            store.grant_user_permissions_in_session(
                session,
                user.username,
                [
                    (RESOURCE_TYPE_SKILL, "new-before-duplicate", MANAGE.name),
                    (RESOURCE_TYPE_SKILL, "existing-skill", MANAGE.name),
                    (RESOURCE_TYPE_AGENT_PLUGIN, "new-after-duplicate", MANAGE.name),
                ],
            )

    assert (
        store.get_role_permission_for_resource(
            user.id, RESOURCE_TYPE_SKILL, "existing-skill", DEFAULT_WORKSPACE_NAME
        )
        == READ
    )
    assert (
        store.get_role_permission_for_resource(
            user.id, RESOURCE_TYPE_SKILL, "new-before-duplicate", DEFAULT_WORKSPACE_NAME
        )
        is None
    )
    assert (
        store.get_role_permission_for_resource(
            user.id, RESOURCE_TYPE_AGENT_PLUGIN, "new-after-duplicate", DEFAULT_WORKSPACE_NAME
        )
        is None
    )


def test_grant_user_resource_permission_in_session_savepoint_keeps_session_usable(
    store, user
):
    store.grant_user_permission(user.username, RESOURCE_TYPE_SKILL, "existing-skill", READ.name)

    class _HiddenRolePermissionQuery:
        def filter(self, *args, **kwargs):
            return self

        def first(self):
            return None

    with store.ManagedSessionMaker(read_only=False) as session:
        original_query = session.query

        def query(*entities, **kwargs):
            if len(entities) == 1 and entities[0] is SqlRolePermission:
                return _HiddenRolePermissionQuery()
            return original_query(*entities, **kwargs)

        session.query = query
        with pytest.raises(MlflowException, match="already exists"):
            store.grant_user_resource_permission_in_session(
                session,
                user.username,
                RESOURCE_TYPE_SKILL,
                "existing-skill",
                MANAGE.name,
            )

        session.query = original_query
        store.grant_user_resource_permission_in_session(
            session,
            user.username,
            RESOURCE_TYPE_AGENT_PLUGIN,
            "new-after-integrity-error",
            MANAGE.name,
        )

    assert (
        store.get_role_permission_for_resource(
            user.id, RESOURCE_TYPE_SKILL, "existing-skill", DEFAULT_WORKSPACE_NAME
        )
        == READ
    )
    assert (
        store.get_role_permission_for_resource(
            user.id,
            RESOURCE_TYPE_AGENT_PLUGIN,
            "new-after-integrity-error",
            DEFAULT_WORKSPACE_NAME,
        )
        == MANAGE
    )


def test_grant_user_permission_in_session_recovers_from_upsert_insert_race(store, user):
    store.grant_user_permission(user.username, RESOURCE_TYPE_SKILL, "existing-skill", READ.name)

    class _HiddenRolePermissionQuery:
        def filter(self, *args, **kwargs):
            return self

        def first(self):
            return None

    hidden_queries = 0
    with store.ManagedSessionMaker(read_only=False) as session:
        original_query = session.query

        def query(*entities, **kwargs):
            nonlocal hidden_queries
            if (
                len(entities) == 1
                and entities[0] is SqlRolePermission
                and hidden_queries == 0
            ):
                hidden_queries += 1
                return _HiddenRolePermissionQuery()
            return original_query(*entities, **kwargs)

        session.query = query
        store.grant_user_permission_in_session(
            session,
            user.username,
            RESOURCE_TYPE_SKILL,
            "existing-skill",
            MANAGE.name,
        )

    assert hidden_queries == 1
    assert (
        store.get_role_permission_for_resource(
            user.id, RESOURCE_TYPE_SKILL, "existing-skill", DEFAULT_WORKSPACE_NAME
        )
        == MANAGE
    )


def test_grant_user_permission_in_session_rejects_workspace_resource_type(store, user):
    with pytest.raises(MlflowException, match="resource_type 'workspace' is not supported"):
        with store.ManagedSessionMaker(read_only=False) as session:
            store.grant_user_permissions_in_session(
                session,
                user.username,
                [(RESOURCE_TYPE_WORKSPACE, "*", MANAGE.name)],
                upsert=True,
            )


def test_grant_user_permission_rejects_workspace_resource_type(store, user):
    with pytest.raises(MlflowException, match="resource_type 'workspace' is not supported"):
        store.grant_user_permission(user.username, RESOURCE_TYPE_WORKSPACE, "*", MANAGE.name)


def test_grant_user_resource_permission_in_session_does_not_overwrite_existing_grant(
    store, user
):
    store.grant_user_permission(user.username, RESOURCE_TYPE_SKILL, "demo-skill", READ.name)

    with pytest.raises(MlflowException, match="already exists"):
        with store.ManagedSessionMaker(read_only=False) as session:
            store.grant_user_resource_permission_in_session(
                session,
                user.username,
                RESOURCE_TYPE_SKILL,
                "demo-skill",
                EDIT.name,
            )

    assert (
        store.get_role_permission_for_resource(
            user.id, RESOURCE_TYPE_SKILL, "demo-skill", DEFAULT_WORKSPACE_NAME
        )
        == READ
    )


def test_grant_user_permission_in_session_preserves_upsert_behavior(store, user):
    with store.ManagedSessionMaker(read_only=False) as session:
        store.grant_user_permission_in_session(
            session,
            user.username,
            RESOURCE_TYPE_SKILL,
            "demo-skill",
            READ.name,
        )

    with store.ManagedSessionMaker(read_only=False) as session:
        store.grant_user_permission_in_session(
            session,
            user.username,
            RESOURCE_TYPE_SKILL,
            "demo-skill",
            EDIT.name,
        )

    assert (
        store.get_role_permission_for_resource(
            user.id, RESOURCE_TYPE_SKILL, "demo-skill", DEFAULT_WORKSPACE_NAME
        )
        == EDIT
    )


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


def test_workspace_use_does_not_fold_into_resource_lookups(store, user):
    # USE is the "member" tier: confers create + workspace-tier access only.
    role = store.create_role(name="user", workspace="ws1")
    store.add_role_permission(role.id, "workspace", "*", "USE")
    store.assign_role_to_user(user.id, role.id)

    assert store.get_role_permission_for_resource(user.id, "experiment", "1", "ws1") is None
    assert store.get_role_permission_for_resource(user.id, "registered_model", "m1", "ws1") is None
    assert store.get_role_permission_for_resource(user.id, "gateway_endpoint", "e1", "ws1") is None
    # Workspace-tier query still finds it (used by _user_can_create_in_workspace).
    assert store.get_role_permission_for_resource(user.id, "workspace", "*", "ws1") == USE


def test_workspace_manage_folds_across_resource_types(store, user):
    # MANAGE still folds — workspace admins see and manage every resource.
    role = store.create_role(name="ws-admin", workspace="ws1")
    store.add_role_permission(role.id, "workspace", "*", "MANAGE")
    store.assign_role_to_user(user.id, role.id)

    assert store.get_role_permission_for_resource(user.id, "experiment", "1", "ws1") == MANAGE
    assert (
        store.get_role_permission_for_resource(user.id, "registered_model", "m1", "ws1") == MANAGE
    )
    assert (
        store.get_role_permission_for_resource(user.id, "gateway_endpoint", "e1", "ws1") == MANAGE
    )


def test_workspace_use_plus_specific_grant_returns_specific(store, user):
    # USE doesn't fold; the specific EDIT grant stands alone.
    role = store.create_role(name="mixed", workspace="ws1")
    store.add_role_permission(role.id, "workspace", "*", "USE")
    store.add_role_permission(role.id, "experiment", "42", "EDIT")
    store.assign_role_to_user(user.id, role.id)

    assert store.get_role_permission_for_resource(user.id, "experiment", "42", "ws1") == EDIT
    assert store.get_role_permission_for_resource(user.id, "experiment", "99", "ws1") is None


def test_is_workspace_admin(store, user):
    role = store.create_role(name="ws-admin", workspace="ws1")
    store.add_role_permission(role.id, "workspace", "*", "MANAGE")
    store.assign_role_to_user(user.id, role.id)

    assert store.is_workspace_admin(user.id, "ws1") is True
    assert store.is_workspace_admin(user.id, "ws2") is False


def test_is_workspace_admin_requires_manage(store, user):
    # A non-MANAGE workspace-wide grant does not make the user a WP admin.
    role = store.create_role(name="ws-user", workspace="ws1")
    store.add_role_permission(role.id, "workspace", "*", "USE")
    store.assign_role_to_user(user.id, role.id)

    assert store.is_workspace_admin(user.id, "ws1") is False


def test_list_role_grants_for_user_in_workspace(store, user):
    # Role with specific + wildcard experiment grants + workspace-wide grant.
    role = store.create_role(name="multi", workspace="ws1")
    store.add_role_permission(role.id, "experiment", "42", "EDIT")
    store.add_role_permission(role.id, "experiment", "*", "READ")
    store.add_role_permission(role.id, "workspace", "*", "USE")
    # Unrelated grant on another resource type.
    store.add_role_permission(role.id, "registered_model", "*", "MANAGE")
    store.assign_role_to_user(user.id, role.id)

    grants = store.list_role_grants_for_user_in_workspace(user.id, "ws1", "experiment")
    # Should include specific experiment grant, wildcard experiment grant,
    # and the workspace-wide grant. Should NOT include the registered_model grant.
    assert sorted(grants) == sorted([("42", "EDIT"), ("*", "READ"), ("*", "USE")])


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
    # A workspace-wide grant with a non-MANAGE permission should not count.
    role = store.create_role(name="user", workspace="ws1")
    store.add_role_permission(role.id, "workspace", "*", "USE")
    store.assign_role_to_user(user.id, role.id)

    assert store.list_workspace_admin_workspaces(user.id) == set()


# ---- Resolver coverage: cross-workspace isolation, NO_PERMISSIONS, resource types ----


def test_resolver_cross_workspace_isolation(store, user):
    """A role scoped to ws1 must not grant anything when resolving in ws2,
    even if the user has the role assigned and the permission pattern matches.
    """
    role = store.create_role(name="editor", workspace="ws1")
    store.add_role_permission(role.id, "experiment", "*", "EDIT")
    store.assign_role_to_user(user.id, role.id)

    # ws1: resolver finds the role and returns EDIT.
    assert store.get_role_permission_for_resource(user.id, "experiment", "1", "ws1") == EDIT
    # ws2: no role tied to ws2 for this user — resolver returns None.
    assert store.get_role_permission_for_resource(user.id, "experiment", "1", "ws2") is None


def test_resolver_role_assignment_in_other_workspace_doesnt_leak(store, user):
    r_ws1 = store.create_role(name="ws1-editor", workspace="ws1")
    store.add_role_permission(r_ws1.id, "experiment", "*", "EDIT")
    store.assign_role_to_user(user.id, r_ws1.id)

    r_ws2 = store.create_role(name="ws2-reader", workspace="ws2")
    store.add_role_permission(r_ws2.id, "experiment", "*", "READ")
    store.assign_role_to_user(user.id, r_ws2.id)

    assert store.get_role_permission_for_resource(user.id, "experiment", "1", "ws1") == EDIT
    assert store.get_role_permission_for_resource(user.id, "experiment", "1", "ws2") == READ
    assert store.get_role_permission_for_resource(user.id, "experiment", "1", "ws3") is None


def _insert_raw_role_permission(store, role_id, resource_type, resource_pattern, permission):
    """Bypass ``add_role_permission``'s scope-aware validator to seed a row that
    pre-dates the simplified model. ``NO_PERMISSIONS`` rows can no longer be
    created through the public API, but legacy databases may still carry them
    from when the early RBAC API accepted any value — the resolver continues to
    honour them as explicit denies for backwards compatibility.
    """
    from sqlalchemy import text

    with store.ManagedSessionMaker() as session:
        session.execute(
            text(
                "INSERT INTO role_permissions"
                " (role_id, resource_type, resource_pattern, permission)"
                " VALUES (:role_id, :resource_type, :resource_pattern, :permission)"
            ),
            {
                "role_id": role_id,
                "resource_type": resource_type,
                "resource_pattern": resource_pattern,
                "permission": permission,
            },
        )


def test_resolver_returns_no_permissions_when_role_only_has_no_permissions(store, user):
    """Backwards-compat: a pre-existing ``NO_PERMISSIONS`` row from the early RBAC
    API is still honoured by the resolver as an explicit deny — the function
    returns the ``NO_PERMISSIONS`` Permission object rather than ``None``.
    """
    role = store.create_role(name="locked", workspace="ws1")
    _insert_raw_role_permission(store, role.id, "experiment", "*", "NO_PERMISSIONS")
    store.assign_role_to_user(user.id, role.id)

    result = store.get_role_permission_for_resource(user.id, "experiment", "1", "ws1")
    assert result is not None
    assert result.name == "NO_PERMISSIONS"


def test_resolver_no_permissions_loses_to_any_positive_grant(store, user):
    """Backwards-compat: when a user has both a legacy ``NO_PERMISSIONS`` row and
    a positive grant (from different roles), the positive grant wins. Mirrors
    the ``max_permission`` policy where explicit grants outrank explicit denies.
    """
    r_deny = store.create_role(name="deny", workspace="ws1")
    _insert_raw_role_permission(store, r_deny.id, "experiment", "*", "NO_PERMISSIONS")
    store.assign_role_to_user(user.id, r_deny.id)

    r_read = store.create_role(name="reader", workspace="ws1")
    store.add_role_permission(r_read.id, "experiment", "*", "READ")
    store.assign_role_to_user(user.id, r_read.id)

    assert store.get_role_permission_for_resource(user.id, "experiment", "1", "ws1") == READ


def test_resolver_unassigned_role_doesnt_grant(store, user):
    """A role in the workspace with the right permissions doesn't help if the
    user isn't assigned to it.
    """
    role = store.create_role(name="editor", workspace="ws1")
    store.add_role_permission(role.id, "experiment", "*", "EDIT")
    # Intentionally skip assign_role_to_user.

    assert store.get_role_permission_for_resource(user.id, "experiment", "1", "ws1") is None


def test_resolver_resource_type_filter(store, user):
    """A grant on resource_type=registered_model does not satisfy an
    experiment lookup (and vice versa). Only the ``workspace`` resource type
    promotes across all types.
    """
    role = store.create_role(name="models-only", workspace="ws1")
    store.add_role_permission(role.id, "registered_model", "*", "EDIT")
    store.assign_role_to_user(user.id, role.id)

    assert store.get_role_permission_for_resource(user.id, "experiment", "1", "ws1") is None
    assert store.get_role_permission_for_resource(user.id, "registered_model", "m1", "ws1") == EDIT


# ---- Resolver coverage: permission hierarchy matrix ----


@pytest.mark.parametrize(
    ("resource_type", "granted", "expected"),
    _RESOURCE_GRANT_CASES,
)
def test_resolver_returns_granted_permission_for_each_resource_type(
    store, user, resource_type, granted, expected
):
    """For each (resource_type, granted_permission) pair, resolving the user's
    permission on a specific resource of that type returns exactly the granted
    permission. This ensures the resolver applies uniformly across every
    resource type the system knows about.
    """
    role = store.create_role(name=f"{resource_type}-{granted}", workspace="ws1")
    store.add_role_permission(role.id, resource_type, "*", granted)
    store.assign_role_to_user(user.id, role.id)

    assert store.get_role_permission_for_resource(user.id, resource_type, "id", "ws1") == expected


@pytest.mark.parametrize("resource_type", _CONCRETE_RESOURCE_TYPES)
def test_resolver_workspace_grant_promotes_to_every_resource_type(store, user, resource_type):
    """``(workspace, *, MANAGE)`` should grant MANAGE on every known resource
    type in the role's workspace. This is the workspace admin short-circuit —
    if it regresses, workspace admins silently lose authority over specific
    resource types.
    """
    role = store.create_role(name="ws-admin", workspace="ws1")
    store.add_role_permission(role.id, "workspace", "*", "MANAGE")
    store.assign_role_to_user(user.id, role.id)

    assert store.get_role_permission_for_resource(user.id, resource_type, "any-id", "ws1") == MANAGE


@pytest.mark.parametrize(
    ("granted", "expected"),
    [
        ("USE", None),
        ("MANAGE", MANAGE),
    ],
)
def test_resolver_workspace_grant_folds_for_manage_only(store, user, granted, expected):
    role = store.create_role(name=f"ws-{granted}", workspace="ws1")
    store.add_role_permission(role.id, "workspace", "*", granted)
    store.assign_role_to_user(user.id, role.id)

    for resource_type in _CONCRETE_RESOURCE_TYPES:
        assert (
            store.get_role_permission_for_resource(user.id, resource_type, "id", "ws1") == expected
        )


def test_resolver_workspace_grant_scoped_to_role_workspace(store, user):
    """A workspace-wide grant in ws1 has no effect when resolving in ws2 —
    the role's workspace scopes the grant.
    """
    role = store.create_role(name="ws1-admin", workspace="ws1")
    store.add_role_permission(role.id, "workspace", "*", "MANAGE")
    store.assign_role_to_user(user.id, role.id)

    assert store.get_role_permission_for_resource(user.id, "experiment", "1", "ws1") == MANAGE
    assert store.get_role_permission_for_resource(user.id, "experiment", "1", "ws2") is None


# ---- Resolver coverage: pattern matching completeness ----


def test_resolver_specific_pattern_does_not_apply_to_different_id(store, user):
    role = store.create_role(name="e42-editor", workspace="ws1")
    store.add_role_permission(role.id, "experiment", "42", "EDIT")
    store.assign_role_to_user(user.id, role.id)

    assert store.get_role_permission_for_resource(user.id, "experiment", "42", "ws1") == EDIT
    assert store.get_role_permission_for_resource(user.id, "experiment", "99", "ws1") is None


def test_resolver_wildcard_applies_to_any_id(store, user):
    role = store.create_role(name="any-experiment", workspace="ws1")
    store.add_role_permission(role.id, "experiment", "*", "READ")
    store.assign_role_to_user(user.id, role.id)

    for eid in ["1", "42", "long-uuid-6a4c"]:
        assert store.get_role_permission_for_resource(user.id, "experiment", eid, "ws1") == READ


def test_resolver_specific_outranks_wildcard_when_higher(store, user):
    # Specific grant > wildcard grant → specific wins.
    role = store.create_role(name="mixed", workspace="ws1")
    store.add_role_permission(role.id, "experiment", "*", "READ")
    store.add_role_permission(role.id, "experiment", "42", "EDIT")
    store.assign_role_to_user(user.id, role.id)

    assert store.get_role_permission_for_resource(user.id, "experiment", "42", "ws1") == EDIT
    assert store.get_role_permission_for_resource(user.id, "experiment", "99", "ws1") == READ


def test_resolver_wildcard_outranks_specific_when_higher(store, user):
    """Wildcard grant > specific grant → wildcard wins (best grant policy,
    not "most specific wins"). This prevents an operator from accidentally
    *downgrading* a user's access by adding a narrower grant with a lower
    permission level.
    """
    role = store.create_role(name="mixed", workspace="ws1")
    store.add_role_permission(role.id, "experiment", "*", "EDIT")
    store.add_role_permission(role.id, "experiment", "42", "READ")
    store.assign_role_to_user(user.id, role.id)

    assert store.get_role_permission_for_resource(user.id, "experiment", "42", "ws1") == EDIT
    assert store.get_role_permission_for_resource(user.id, "experiment", "99", "ws1") == EDIT


# ---- Resolver coverage: multi-role union ----


def test_resolver_union_picks_max_across_roles(store, user):
    # Permissions union across all roles assigned to the user — max wins.
    r1 = store.create_role(name="r1", workspace="ws1")
    store.add_role_permission(r1.id, "experiment", "*", "READ")
    store.assign_role_to_user(user.id, r1.id)

    r2 = store.create_role(name="r2", workspace="ws1")
    store.add_role_permission(r2.id, "experiment", "*", "USE")
    store.assign_role_to_user(user.id, r2.id)

    r3 = store.create_role(name="r3", workspace="ws1")
    store.add_role_permission(r3.id, "experiment", "*", "MANAGE")
    store.assign_role_to_user(user.id, r3.id)

    assert store.get_role_permission_for_resource(user.id, "experiment", "1", "ws1") == MANAGE


def test_resolver_union_mixes_workspace_use_and_resource_grants(store, user):
    # USE doesn't fold; specific READ wins.
    r_ws = store.create_role(name="ws-user", workspace="ws1")
    store.add_role_permission(r_ws.id, "workspace", "*", "USE")
    store.assign_role_to_user(user.id, r_ws.id)

    r_specific = store.create_role(name="one-reader", workspace="ws1")
    store.add_role_permission(r_specific.id, "experiment", "42", "READ")
    store.assign_role_to_user(user.id, r_specific.id)

    assert store.get_role_permission_for_resource(user.id, "experiment", "42", "ws1") == READ


def test_resolver_union_mixes_workspace_manage_and_resource_grants(store, user):
    # MANAGE folds and wins against any lesser specific grant.
    r_ws = store.create_role(name="ws-admin", workspace="ws1")
    store.add_role_permission(r_ws.id, "workspace", "*", "MANAGE")
    store.assign_role_to_user(user.id, r_ws.id)

    r_specific = store.create_role(name="one-reader", workspace="ws1")
    store.add_role_permission(r_specific.id, "experiment", "42", "READ")
    store.assign_role_to_user(user.id, r_specific.id)

    assert store.get_role_permission_for_resource(user.id, "experiment", "42", "ws1") == MANAGE


# ---- Legacy workspace_permissions as workspace admin source ----
#
# Pre-RBAC operators relied on `workspace_permissions` MANAGE to convey workspace-wide
# admin authority. The workspace admin helpers must still recognize that grant,
# otherwise operators mid-migration (or just not yet using roles) silently lose admin
# status behind RBAC-aware validators.


def test_is_workspace_admin_honors_legacy_workspace_permissions(store, user):
    store.set_workspace_permission("ws1", user.username, "MANAGE")

    assert store.is_workspace_admin(user.id, "ws1") is True
    assert store.is_workspace_admin(user.id, "ws2") is False


def test_is_workspace_admin_ignores_non_manage_legacy(store, user):
    store.set_workspace_permission("ws1", user.username, "USE")

    assert store.is_workspace_admin(user.id, "ws1") is False


def test_list_workspace_admin_workspaces_unions_role_and_legacy(store, user):
    # Role admin in ws1, legacy MANAGE in ws2, legacy USE in ws3 (should not count).
    role = store.create_role(name="wa1", workspace="ws1")
    store.add_role_permission(role.id, "workspace", "*", "MANAGE")
    store.assign_role_to_user(user.id, role.id)
    store.set_workspace_permission("ws2", user.username, "MANAGE")
    store.set_workspace_permission("ws3", user.username, "USE")

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
    store.set_workspace_permission("ws1", user2.username, "USE")

    assert store.is_workspace_admin_of_any_of_users_workspaces(user.id, user2.id) is True


def test_is_workspace_admin_of_any_of_users_workspaces_no_overlap(store, user, user2):
    # Admin in ws1, target present only in ws2 → no intersection.
    store.set_workspace_permission("ws1", user.username, "MANAGE")
    store.set_workspace_permission("ws2", user2.username, "USE")

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


def test_prompt_and_registered_model_grants_do_not_cross_resolve(store, user):
    """A grant on `(registered_model, foo)` must NOT satisfy a request for
    `(prompt, foo)` and vice versa — the resource_type discriminant has to
    isolate the two namespaces post-promotion.
    """
    role = store.create_role(name="grants", workspace="ws1")
    store.add_role_permission(role.id, "registered_model", "foo", "READ")
    store.add_role_permission(role.id, "prompt", "bar", "MANAGE")
    store.assign_role_to_user(user.id, role.id)

    assert store.get_role_permission_for_resource(user.id, "prompt", "foo", "ws1") is None
    assert store.get_role_permission_for_resource(user.id, "registered_model", "bar", "ws1") is None
    assert store.get_role_permission_for_resource(user.id, "registered_model", "foo", "ws1") == READ
    assert store.get_role_permission_for_resource(user.id, "prompt", "bar", "ws1") == MANAGE


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
