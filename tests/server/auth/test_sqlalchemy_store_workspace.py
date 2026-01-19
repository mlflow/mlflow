import pytest

from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.exceptions import MlflowException
from mlflow.server.auth.entities import WorkspacePermission
from mlflow.server.auth.permissions import EDIT, MANAGE, NO_PERMISSIONS, READ
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

    perm = store.set_workspace_permission(workspace, username, READ.name)
    assert isinstance(perm, WorkspacePermission)
    assert perm.workspace == workspace
    assert perm.user_id == user.id
    assert perm.permission == READ.name

    updated = store.set_workspace_permission(workspace, username, MANAGE.name)
    assert updated.permission == MANAGE.name


def test_get_workspace_permission_precedence(store):
    workspace = "team-beta"
    username = random_str()
    store.create_user(username, random_str())

    assert store.get_workspace_permission(workspace, username) is None

    store.set_workspace_permission(workspace, username, READ.name)
    perm = store.get_workspace_permission(workspace, username)
    assert perm == READ


def test_list_workspace_permissions(store):
    workspace = "team-gamma"
    other_workspace = "team-delta"
    username = random_str()
    other_username = random_str()
    user = store.create_user(username, random_str())
    other_user = store.create_user(other_username, random_str())

    p1 = store.set_workspace_permission(workspace, username, READ.name)
    p2 = store.set_workspace_permission(workspace, other_username, EDIT.name)
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

    store.set_workspace_permission(workspace, username, READ.name)

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

    store.set_workspace_permission(workspace, username, READ.name)
    store.set_workspace_permission(other_workspace, username, EDIT.name)

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

    store.set_workspace_permission("workspace-read", username, READ.name)
    store.set_workspace_permission("workspace-edit", username, EDIT.name)
    store.set_workspace_permission("workspace-no-access", username, NO_PERMISSIONS.name)
    store.set_workspace_permission("workspace-other", other_user, READ.name)

    accessible = store.list_accessible_workspace_names(username)
    assert accessible == {"workspace-read", "workspace-edit"}

    assert store.list_accessible_workspace_names(other_user) == {
        "workspace-other",
    }
    assert store.list_accessible_workspace_names(None) == set()


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
