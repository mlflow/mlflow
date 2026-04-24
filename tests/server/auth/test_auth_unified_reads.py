import pytest
from alembic.command import downgrade as alembic_downgrade
from sqlalchemy import create_engine

from mlflow.environment_variables import MLFLOW_RBAC_UNIFIED_READS
from mlflow.exceptions import MlflowException
from mlflow.server import auth as auth_module
from mlflow.server.auth.db.utils import _get_alembic_config, migrate
from mlflow.server.auth.permissions import EDIT, MANAGE, READ
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


# ---- Resolver under the unified-reads flag ----


def test_role_only_grant_resolves_via_unified_reads(store, user, monkeypatch):
    # Mirror an explicit role grant directly (no dual-write detour).
    role = store.create_role(name="viewer", workspace=DEFAULT_WORKSPACE_NAME)
    store.add_role_permission(role.id, "experiment", "exp1", READ.name)
    store.assign_role_to_user(user.id, role.id)

    # Without the flag, legacy mode also finds it via role_permission_func.
    resolved = store.get_role_permission_for_resource(
        user.id, "experiment", "exp1", DEFAULT_WORKSPACE_NAME
    )
    assert resolved == READ


def test_dual_written_grant_resolves_via_unified_reads(store, user):
    # Legacy grant → dual-write mirrors it → unified reads surface the same permission.
    store.create_experiment_permission("exp1", user.username, EDIT.name)
    assert (
        store.get_role_permission_for_resource(
            user.id, "experiment", "exp1", DEFAULT_WORKSPACE_NAME
        )
        == EDIT
    )


def test_union_across_direct_and_role_grants_takes_max(store, user):
    # Direct READ + role EDIT → EDIT. Dual-write mirrors the direct grant too, so the role
    # resolution itself sees both.
    role = store.create_role(name="editor", workspace=DEFAULT_WORKSPACE_NAME)
    store.add_role_permission(role.id, "experiment", "*", EDIT.name)
    store.assign_role_to_user(user.id, role.id)

    store.create_experiment_permission("exp1", user.username, READ.name)

    # The role resolver takes the max across all grants touching this resource.
    assert (
        store.get_role_permission_for_resource(
            user.id, "experiment", "exp1", DEFAULT_WORKSPACE_NAME
        )
        == EDIT
    )


def test_unified_resolver_honors_explicit_no_permissions(store, user, monkeypatch):
    # An explicit NO_PERMISSIONS grant must surface under the flag (it's a denial, not
    # "no grant"). Critical for parity with the legacy direct-grant path.
    monkeypatch.setenv(MLFLOW_RBAC_UNIFIED_READS.name, "true")
    store.create_experiment_permission("exp1", user.username, "NO_PERMISSIONS")

    role_perm = store.get_role_permission_for_resource(
        user.id, "experiment", "exp1", DEFAULT_WORKSPACE_NAME
    )
    assert role_perm.name == "NO_PERMISSIONS"


# ---- Workspace enumeration helpers ----


def test_list_accessible_workspace_names_via_roles(store, user, monkeypatch):
    role_a = store.create_role(name="reader", workspace="ws-a")
    store.add_role_permission(role_a.id, "experiment", "*", READ.name)
    store.assign_role_to_user(user.id, role_a.id)

    role_b = store.create_role(name="manager", workspace="ws-b")
    store.add_role_permission(role_b.id, "workspace", "*", MANAGE.name)
    store.assign_role_to_user(user.id, role_b.id)

    # A role in ws-c that grants no readable permission should not show up.
    role_c = store.create_role(name="denied", workspace="ws-c")
    store.add_role_permission(role_c.id, "experiment", "exp-x", "NO_PERMISSIONS")
    store.assign_role_to_user(user.id, role_c.id)

    assert store.list_accessible_workspace_names_via_roles(user.username) == {"ws-a", "ws-b"}


def test_get_workspace_permission_via_roles(store, user):
    # type-wildcard (`*`) grant should be visible to the workspace-permission helper.
    role = store.create_role(name="ws", workspace="ws-a")
    store.add_role_permission(role.id, "*", "*", READ.name)
    store.assign_role_to_user(user.id, role.id)

    assert store.get_workspace_permission_via_roles(user.username, "ws-a") == READ
    # No grant in ws-b → None (triggers the "deny by default" workspace fallback in
    # `_workspace_permission` when workspaces are enabled).
    assert store.get_workspace_permission_via_roles(user.username, "ws-b") is None


# ---- Filter-path helpers ----


def test_role_can_read_map_returns_wildcard_flag(store, user, monkeypatch):
    # ``_role_can_read_map`` powers the filter-function fast path for wildcard grants.
    role = store.create_role(name="reader", workspace=DEFAULT_WORKSPACE_NAME)
    store.add_role_permission(role.id, "experiment", "*", READ.name)
    store.assign_role_to_user(user.id, role.id)

    monkeypatch.setattr(auth_module, "store", store)
    can_read, wildcard = auth_module._role_can_read_map(user.id, "experiment")
    assert wildcard is True
    assert can_read == {}


def test_role_can_read_map_specific_grant(store, user, monkeypatch):
    role = store.create_role(name="reader", workspace=DEFAULT_WORKSPACE_NAME)
    store.add_role_permission(role.id, "experiment", "exp1", READ.name)
    store.add_role_permission(role.id, "experiment", "exp2", "NO_PERMISSIONS")
    store.assign_role_to_user(user.id, role.id)

    monkeypatch.setattr(auth_module, "store", store)
    can_read, wildcard = auth_module._role_can_read_map(user.id, "experiment")
    assert wildcard is False
    assert can_read == {"exp1": True, "exp2": False}


# ---- Startup precondition ----


def _make_store_at_revision(db_path, revision: str) -> SqlAlchemyStore:
    """Migrate an auth DB to ``revision`` (possibly pre-head), then return a store bound
    to the same engine WITHOUT re-running the normal ``init_db`` auto-upgrade path.
    """
    engine = create_engine(f"sqlite:///{db_path}")
    migrate(engine, "d4e5f6a7b8c9")
    if revision != "d4e5f6a7b8c9":
        cfg = _get_alembic_config(engine.url.render_as_string(hide_password=False))
        with engine.begin() as conn:
            cfg.attributes["connection"] = conn
            alembic_downgrade(cfg, revision)
    fresh = SqlAlchemyStore()
    fresh.engine = engine
    return fresh


def test_startup_assertion_rejects_pre_backfill_revision(tmp_path, monkeypatch):
    store = _make_store_at_revision(tmp_path / "auth.db", "c3d4e5f6a7b8")
    monkeypatch.setattr(auth_module, "store", store, raising=False)
    monkeypatch.setenv(MLFLOW_RBAC_UNIFIED_READS.name, "true")

    with pytest.raises(MlflowException, match="role_permissions backfill"):
        auth_module._assert_unified_reads_preconditions()

    store.engine.dispose()


def test_startup_assertion_accepts_post_backfill_revision(tmp_path, monkeypatch):
    store = _make_store_at_revision(tmp_path / "auth.db", "d4e5f6a7b8c9")
    monkeypatch.setattr(auth_module, "store", store, raising=False)
    monkeypatch.setenv(MLFLOW_RBAC_UNIFIED_READS.name, "true")

    auth_module._assert_unified_reads_preconditions()
    store.engine.dispose()


def test_startup_assertion_noop_when_flag_off(tmp_path, monkeypatch):
    store = _make_store_at_revision(tmp_path / "auth.db", "c3d4e5f6a7b8")
    monkeypatch.setattr(auth_module, "store", store, raising=False)
    # Explicitly force the flag off (default is True in 3.13+).
    monkeypatch.setenv(MLFLOW_RBAC_UNIFIED_READS.name, "false")

    auth_module._assert_unified_reads_preconditions()
    store.engine.dispose()
