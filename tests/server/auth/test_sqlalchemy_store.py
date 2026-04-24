import pytest

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)
from mlflow.server.auth.entities import User
from mlflow.server.auth.permissions import EDIT, MANAGE, READ, USE
from mlflow.server.auth.sqlalchemy_store import SqlAlchemyStore
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

from tests.helper_functions import random_str

pytestmark = pytest.mark.notrackingurimock


@pytest.fixture
def store(tmp_sqlite_uri):
    store = SqlAlchemyStore()
    store.init_db(tmp_sqlite_uri)
    return store


def _user_maker(store, username, password, is_admin=False):
    return store.create_user(username, password, is_admin)


def test_create_user(store):
    username1 = random_str()
    password1 = random_str()
    user1 = _user_maker(store, username1, password1)
    assert user1.username == username1
    assert user1.password_hash != password1
    assert user1.is_admin is False

    # error on duplicate
    with pytest.raises(
        MlflowException, match=rf"User \(username={username1}\) already exists"
    ) as exception_context:
        _user_maker(store, username1, password1)
    assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_ALREADY_EXISTS)

    # slightly different name is ok
    username2 = username1 + "_2"
    password2 = password1 + "_2"
    user2 = _user_maker(store, username2, password2, is_admin=True)
    assert user2.username == username2
    assert user2.password_hash != password2
    assert user2.is_admin is True

    # invalid username will fail
    with pytest.raises(MlflowException, match=r"Username cannot be empty") as exception_context:
        _user_maker(store, None, None)
    assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)
    with pytest.raises(MlflowException, match=r"Username cannot be empty") as exception_context:
        _user_maker(store, "", "")
    assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_has_user(store):
    username1 = random_str()
    password1 = random_str()
    _user_maker(store, username1, password1)
    assert store.has_user(username=username1) is True

    # error on non-existent user
    username2 = random_str()
    assert store.has_user(username=username2) is False


def test_get_user(store):
    username1 = random_str()
    password1 = random_str()
    _user_maker(store, username1, password1)
    user1 = store.get_user(username=username1)
    assert isinstance(user1, User)
    assert user1.username == username1

    # error on non-existent user
    username2 = random_str()
    with pytest.raises(
        MlflowException, match=rf"User with username={username2} not found"
    ) as exception_context:
        store.get_user(username=username2)
    assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_list_user(store):
    username1 = "1" + random_str()
    password1 = "1" + random_str()
    _user_maker(store, username1, password1)

    username2 = "2" + random_str()
    password2 = "2" + random_str()
    _user_maker(store, username2, password2)

    username3 = "3" + random_str()
    password3 = "3" + random_str()
    _user_maker(store, username3, password3)

    users = store.list_users()
    users.sort(key=lambda u: u.username)

    assert len(users) == 3
    assert isinstance(users[0], User)
    assert users[0].username == username1
    assert users[1].username == username2
    assert users[2].username == username3


def test_authenticate_user(store):
    username1 = random_str()
    password1 = random_str()
    _user_maker(store, username1, password1)
    assert store.authenticate_user(username1, password1)
    assert not store.authenticate_user(username1, random_str())
    # non existent user
    assert not store.authenticate_user(random_str(), random_str())


def test_update_user(store):
    username1 = random_str()
    password1 = random_str()
    _user_maker(store, username1, password1)
    password2 = random_str()
    store.update_user(username1, password=password2)
    assert not store.authenticate_user(username1, password1)
    assert store.authenticate_user(username1, password2)

    store.update_user(username1, is_admin=True)
    assert store.get_user(username1).is_admin
    store.update_user(username1, is_admin=False)
    assert not store.get_user(username1).is_admin


def test_delete_user(store):
    username1 = random_str()
    password1 = random_str()
    _user_maker(store, username1, password1)
    store.delete_user(username1)

    with pytest.raises(
        MlflowException,
        match=rf"User with username={username1} not found",
    ) as exception_context:
        store.get_user(username1)
    assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


# ---------------------------------------------------------------------------
# Per-user (synthetic role) logic
#
# The store represents per-user grants as ``role_permissions`` rows on a
# synthetic ``__user_<id>__`` role scoped to a workspace. The tests above
# exercise the create/get/update/delete API surface through this representation
# implicitly, but the per-user invariants below are what protect the design
# from regressions. A change that broke any of these would silently leak grants
# across users / workspaces or fragment a single user's grants across multiple
# synthetic roles.
# ---------------------------------------------------------------------------


def _count_synthetic_roles_for(store, user_id: int) -> int:
    from mlflow.server.auth.db.models import SqlRole

    name = store._synthetic_user_role_name(user_id)
    with store.ManagedSessionMaker() as session:
        return session.query(SqlRole).filter(SqlRole.name == name).count()


def test_per_user_grant_does_not_leak_to_other_users(store):
    user_a = _user_maker(store, random_str(), random_str())
    user_b = _user_maker(store, random_str(), random_str())
    experiment_id = random_str()

    store.create_experiment_permission(experiment_id, user_a.username, MANAGE.name)

    granted = store.get_experiment_permission(experiment_id, user_a.username)
    assert granted.user_id == user_a.id
    assert granted.permission == MANAGE.name

    with pytest.raises(MlflowException, match=r"not found") as exc:
        store.get_experiment_permission(experiment_id, user_b.username)
    assert exc.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_per_user_grants_share_one_synthetic_role(store):
    user = _user_maker(store, random_str(), random_str())

    # Three grants on different resource types for the same user in the default
    # workspace. All must land on the same synthetic role row, not three.
    store.create_experiment_permission(random_str(), user.username, READ.name)
    store.create_registered_model_permission(random_str(), user.username, EDIT.name)
    store.create_scorer_permission(random_str(), random_str(), user.username, USE.name)

    assert _count_synthetic_roles_for(store, user.id) == 1


def test_per_user_grants_use_one_synthetic_role_per_workspace(store, monkeypatch):
    from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
    from mlflow.server.auth.db.models import SqlRole
    from mlflow.utils.workspace_context import WorkspaceContext

    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    user = _user_maker(store, random_str(), random_str())

    workspaces = [f"ws-{random_str()}" for _ in range(2)]
    for ws in workspaces:
        with WorkspaceContext(ws):
            store.create_registered_model_permission(f"model-{ws}", user.username, USE.name)

    # Distinct synthetic roles: one per workspace.
    name = store._synthetic_user_role_name(user.id)
    with store.ManagedSessionMaker() as session:
        observed_workspaces = sorted(
            workspace
            for (workspace,) in session.query(SqlRole.workspace).filter(SqlRole.name == name).all()
        )
    assert observed_workspaces == sorted(workspaces)


def test_per_user_grants_resolved_via_role_resolver(store):
    user_a = _user_maker(store, random_str(), random_str())
    user_b = _user_maker(store, random_str(), random_str())
    experiment_id = random_str()

    store.create_experiment_permission(experiment_id, user_a.username, EDIT.name)

    # User A's grant resolves through the unified resolver — the same mechanism
    # the runtime authz uses, not a per-resource fallback.
    perm = store.get_role_permission_for_resource(
        user_a.id, "experiment", experiment_id, DEFAULT_WORKSPACE_NAME
    )
    assert perm is not None
    assert perm.name == EDIT.name

    # User B has no grant on this experiment — resolver returns None.
    assert (
        store.get_role_permission_for_resource(
            user_b.id, "experiment", experiment_id, DEFAULT_WORKSPACE_NAME
        )
        is None
    )


def test_per_user_grants_isolated_across_workspaces(store, monkeypatch):
    from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
    from mlflow.utils.workspace_context import WorkspaceContext

    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    user = _user_maker(store, random_str(), random_str())
    ws_a = f"ws-{random_str()}"
    ws_b = f"ws-{random_str()}"

    with WorkspaceContext(ws_a):
        store.create_registered_model_permission("model-1", user.username, MANAGE.name)

    # In ws_a the grant is visible.
    assert (
        store.get_role_permission_for_resource(user.id, "registered_model", "model-1", ws_a).name
        == MANAGE.name
    )
    # In ws_b the grant must not leak.
    assert (
        store.get_role_permission_for_resource(user.id, "registered_model", "model-1", ws_b) is None
    )


def test_delete_user_clears_retained_legacy_permission_rows(store):
    """``e5f6a7b8c9d0`` retains the legacy permission tables on disk (they hold
    pre-migration data for rollback). Their FKs to ``users.id`` are non-cascading
    in earlier migrations, so a user with surviving legacy rows can't be deleted
    on strict FK backends (e.g. PostgreSQL) unless ``delete_user`` first clears
    them. Simulate that state by inserting raw rows into every legacy table and
    confirm ``delete_user`` cleans them up rather than orphaning or erroring.
    """
    from sqlalchemy import text

    username = random_str()
    user = _user_maker(store, username, random_str())

    legacy_inserts = {
        "experiment_permissions": (
            "INSERT INTO experiment_permissions (experiment_id, user_id, permission)"
            " VALUES (:eid, :uid, 'READ')"
        ),
        "registered_model_permissions": (
            "INSERT INTO registered_model_permissions (workspace, name, user_id, permission)"
            " VALUES ('ws-default', :rid, :uid, 'READ')"
        ),
        "scorer_permissions": (
            "INSERT INTO scorer_permissions"
            " (experiment_id, scorer_name, user_id, permission)"
            " VALUES (:eid, :sname, :uid, 'READ')"
        ),
        "gateway_secret_permissions": (
            "INSERT INTO gateway_secret_permissions (secret_id, user_id, permission)"
            " VALUES (:gid, :uid, 'READ')"
        ),
        "gateway_endpoint_permissions": (
            "INSERT INTO gateway_endpoint_permissions (endpoint_id, user_id, permission)"
            " VALUES (:eid, :uid, 'READ')"
        ),
        "gateway_model_definition_permissions": (
            "INSERT INTO gateway_model_definition_permissions"
            " (model_definition_id, user_id, permission)"
            " VALUES (:mid, :uid, 'READ')"
        ),
        "workspace_permissions": (
            "INSERT INTO workspace_permissions (workspace, user_id, permission)"
            " VALUES ('ws-default', :uid, 'USE')"
        ),
    }
    with store.ManagedSessionMaker() as session:
        for stmt in legacy_inserts.values():
            session.execute(
                text(stmt),
                {
                    "uid": user.id,
                    "eid": random_str(),
                    "rid": random_str(),
                    "sname": random_str(),
                    "gid": random_str(),
                    "mid": random_str(),
                },
            )

    store.delete_user(username)

    with store.ManagedSessionMaker() as session:
        for table in legacy_inserts:
            count = session.execute(
                text(f"SELECT COUNT(*) FROM {table} WHERE user_id = :uid"),
                {"uid": user.id},
            ).scalar()
            assert count == 0, f"legacy rows in {table} not cleaned up by delete_user"
