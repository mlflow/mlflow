import pytest

from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)
from mlflow.server.auth.entities import (
    ExperimentPermission,
    RegisteredModelPermission,
    ScorerPermission,
    User,
    WorkspacePermission,
)
from mlflow.server.auth.permissions import (
    ALL_PERMISSIONS,
    EDIT,
    MANAGE,
    NO_PERMISSIONS,
    READ,
)
from mlflow.server.auth.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracking._workspace.context import WorkspaceContext
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


def _ep_maker(store, experiment_id, username, permission):
    return store.create_experiment_permission(experiment_id, username, permission)


def _rmp_maker(store, name, username, permission):
    return store.create_registered_model_permission(name, username, permission)


def _sp_maker(store, experiment_id, scorer_name, username, permission):
    return store.create_scorer_permission(experiment_id, scorer_name, username, permission)


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


def test_create_experiment_permission(store):
    username1 = random_str()
    password1 = random_str()
    user1 = _user_maker(store, username1, password1)

    experiment_id1 = random_str()
    user_id1 = user1.id
    permission1 = READ.name
    ep1 = _ep_maker(store, experiment_id1, username1, permission1)
    assert ep1.experiment_id == experiment_id1
    assert ep1.user_id == user_id1
    assert ep1.permission == permission1

    # error on duplicate
    with pytest.raises(
        MlflowException,
        match=rf"Experiment permission \(experiment_id={experiment_id1}, "
        rf"username={username1}\) already exists",
    ) as exception_context:
        _ep_maker(store, experiment_id1, username1, permission1)
    assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_ALREADY_EXISTS)

    # slightly different name is ok
    experiment_id2 = random_str()
    ep2 = _ep_maker(store, experiment_id2, username1, permission1)
    assert ep2.experiment_id == experiment_id2
    assert ep2.user_id == user_id1
    assert ep2.permission == permission1

    # all permissions are ok
    for perm in ALL_PERMISSIONS:
        experiment_id3 = random_str()
        ep3 = _ep_maker(store, experiment_id3, username1, perm)
        assert ep3.experiment_id == experiment_id3
        assert ep3.user_id == user_id1
        assert ep3.permission == perm

    # invalid permission will fail
    experiment_id4 = random_str()
    with pytest.raises(MlflowException, match=r"Invalid permission") as exception_context:
        _ep_maker(store, experiment_id4, username1, "some_invalid_permission_string")
    assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_get_experiment_permission(store):
    username1 = random_str()
    password1 = random_str()
    user1 = _user_maker(store, username1, password1)

    experiment_id1 = random_str()
    user_id1 = user1.id
    permission1 = READ.name
    _ep_maker(store, experiment_id1, username1, permission1)
    ep1 = store.get_experiment_permission(experiment_id1, username1)
    assert isinstance(ep1, ExperimentPermission)
    assert ep1.experiment_id == experiment_id1
    assert ep1.user_id == user_id1
    assert ep1.permission == permission1

    # error on non-existent row
    experiment_id2 = random_str()
    with pytest.raises(
        MlflowException,
        match=rf"Experiment permission with experiment_id={experiment_id2} "
        rf"and username={username1} not found",
    ) as exception_context:
        store.get_experiment_permission(experiment_id2, username1)
    assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_list_experiment_permission(store):
    username1 = random_str()
    password1 = random_str()
    _user_maker(store, username1, password1)

    experiment_id1 = "1" + random_str()
    _ep_maker(store, experiment_id1, username1, READ.name)

    experiment_id2 = "2" + random_str()
    _ep_maker(store, experiment_id2, username1, READ.name)

    experiment_id3 = "3" + random_str()
    _ep_maker(store, experiment_id3, username1, READ.name)

    eps = store.list_experiment_permissions(username1)
    eps.sort(key=lambda ep: ep.experiment_id)

    assert len(eps) == 3
    assert isinstance(eps[0], ExperimentPermission)
    assert eps[0].experiment_id == experiment_id1
    assert eps[1].experiment_id == experiment_id2
    assert eps[2].experiment_id == experiment_id3


def test_update_experiment_permission(store):
    username1 = random_str()
    password1 = random_str()
    _user_maker(store, username1, password1)

    experiment_id1 = random_str()
    permission1 = READ.name
    _ep_maker(store, experiment_id1, username1, permission1)

    permission2 = EDIT.name
    store.update_experiment_permission(experiment_id1, username1, permission2)
    ep1 = store.get_experiment_permission(experiment_id1, username1)
    assert ep1.permission == permission2

    # invalid permission will fail
    with pytest.raises(MlflowException, match=r"Invalid permission") as exception_context:
        store.update_experiment_permission(
            experiment_id1, username1, "some_invalid_permission_string"
        )
    assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_delete_experiment_permission(store):
    username1 = random_str()
    password1 = random_str()
    _user_maker(store, username1, password1)

    experiment_id1 = random_str()
    permission1 = READ.name
    _ep_maker(store, experiment_id1, username1, permission1)

    store.delete_experiment_permission(experiment_id1, username1)
    with pytest.raises(
        MlflowException,
        match=rf"Experiment permission with experiment_id={experiment_id1} "
        rf"and username={username1} not found",
    ) as exception_context:
        store.get_experiment_permission(experiment_id1, username1)
    assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_create_registered_model_permission(store):
    username1 = random_str()
    password1 = random_str()
    user1 = _user_maker(store, username1, password1)

    name1 = random_str()
    user_id1 = user1.id
    permission1 = READ.name
    rmp1 = _rmp_maker(store, name1, username1, permission1)
    assert rmp1.name == name1
    assert rmp1.user_id == user_id1
    assert rmp1.permission == permission1
    assert rmp1.workspace == DEFAULT_WORKSPACE_NAME

    # error on duplicate
    duplicate_permission_pattern = (
        rf"(?s)Registered model permission "
        rf"with workspace={DEFAULT_WORKSPACE_NAME}, name={name1} "
        rf"and username={username1} already exists"
    )
    with pytest.raises(
        MlflowException,
        match=duplicate_permission_pattern,
    ) as exception_context:
        _rmp_maker(store, name1, username1, permission1)
    assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_ALREADY_EXISTS)

    # slightly different name is ok
    name2 = random_str()
    rmp2 = _rmp_maker(store, name2, username1, permission1)
    assert rmp2.name == name2
    assert rmp2.user_id == user_id1
    assert rmp2.permission == permission1
    assert rmp2.workspace == DEFAULT_WORKSPACE_NAME

    # all permissions are ok
    for perm in ALL_PERMISSIONS:
        name3 = random_str()
        rmp3 = _rmp_maker(store, name3, username1, perm)
        assert rmp3.name == name3
        assert rmp3.user_id == user_id1
        assert rmp3.permission == perm
        assert rmp3.workspace == DEFAULT_WORKSPACE_NAME

    # invalid permission will fail
    name4 = random_str()
    with pytest.raises(MlflowException, match=r"Invalid permission") as exception_context:
        _rmp_maker(store, name4, username1, "some_invalid_permission_string")
    assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_get_registered_model_permission(store):
    username1 = random_str()
    password1 = random_str()
    user1 = _user_maker(store, username1, password1)

    name1 = random_str()
    user_id1 = user1.id
    permission1 = READ.name
    _rmp_maker(store, name1, username1, permission1)
    rmp1 = store.get_registered_model_permission(name1, username1)
    assert isinstance(rmp1, RegisteredModelPermission)
    assert rmp1.name == name1
    assert rmp1.user_id == user_id1
    assert rmp1.permission == permission1
    assert rmp1.workspace == DEFAULT_WORKSPACE_NAME

    # error on non-existent row
    name2 = random_str()
    missing_permission_message = (
        "Registered model permission with "
        f"workspace={DEFAULT_WORKSPACE_NAME}, name={name2} "
        f"and username={username1} not found"
    )
    with pytest.raises(
        MlflowException,
        match=missing_permission_message,
    ) as exception_context:
        store.get_registered_model_permission(name2, username1)
    assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_list_registered_model_permission(store):
    username1 = random_str()
    password1 = random_str()
    _user_maker(store, username1, password1)

    name1 = "1" + random_str()
    _rmp_maker(store, name1, username1, READ.name)

    name2 = "2" + random_str()
    _rmp_maker(store, name2, username1, READ.name)

    name3 = "3" + random_str()
    _rmp_maker(store, name3, username1, READ.name)

    rmps = store.list_registered_model_permissions(username1)
    rmps.sort(key=lambda rmp: rmp.name)

    assert len(rmps) == 3
    assert isinstance(rmps[0], RegisteredModelPermission)
    assert rmps[0].name == name1
    assert rmps[0].workspace == DEFAULT_WORKSPACE_NAME
    assert rmps[1].name == name2
    assert rmps[1].workspace == DEFAULT_WORKSPACE_NAME
    assert rmps[2].name == name3
    assert rmps[2].workspace == DEFAULT_WORKSPACE_NAME


def test_update_registered_model_permission(store):
    username1 = random_str()
    password1 = random_str()
    _user_maker(store, username1, password1)

    name1 = random_str()
    permission1 = READ.name
    _rmp_maker(store, name1, username1, permission1)

    permission2 = EDIT.name
    store.update_registered_model_permission(name1, username1, permission2)
    rmp1 = store.get_registered_model_permission(name1, username1)
    assert rmp1.permission == permission2
    assert rmp1.workspace == DEFAULT_WORKSPACE_NAME

    # invalid permission will fail
    with pytest.raises(MlflowException, match=r"Invalid permission") as exception_context:
        store.update_registered_model_permission(name1, username1, "some_invalid_permission_string")
    assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_delete_registered_model_permission(store):
    username1 = random_str()
    password1 = random_str()
    _user_maker(store, username1, password1)

    name1 = random_str()
    permission1 = READ.name
    _rmp_maker(store, name1, username1, permission1)

    store.delete_registered_model_permission(name1, username1)
    missing_permission_message = (
        "Registered model permission with "
        f"workspace={DEFAULT_WORKSPACE_NAME}, name={name1} "
        f"and username={username1} not found"
    )
    with pytest.raises(
        MlflowException,
        match=missing_permission_message,
    ) as exception_context:
        store.get_registered_model_permission(name1, username1)
    assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_set_workspace_permission_creates_and_updates(store):
    workspace = "team-alpha"
    username = random_str()

    perm = store.set_workspace_permission(workspace, username, "experiments", READ.name)
    assert isinstance(perm, WorkspacePermission)
    assert perm.workspace == workspace
    assert perm.username == username
    assert perm.resource_type == "experiments"
    assert perm.permission == READ.name

    updated = store.set_workspace_permission(workspace, username, "experiments", MANAGE.name)
    assert updated.permission == MANAGE.name

    with pytest.raises(MlflowException, match="Invalid resource type"):
        store.set_workspace_permission(workspace, username, "invalid-resource", READ.name)


def test_get_workspace_permission_precedence(store):
    workspace = "team-beta"
    username = random_str()
    other_user = random_str()

    # wildcard defaults
    store.set_workspace_permission(workspace, "*", "*", READ.name)
    # wildcard resource, specific user
    store.set_workspace_permission(workspace, username, "*", EDIT.name)
    # resource specific wildcard user
    store.set_workspace_permission(workspace, "*", "registered_models", MANAGE.name)
    # specific user and resource
    store.set_workspace_permission(workspace, username, "registered_models", READ.name)

    perm = store.get_workspace_permission(workspace, username, "registered_models")
    assert perm == READ

    # For experiments no specific entry -> fall back to username wildcard "*"
    perm = store.get_workspace_permission(workspace, username, "experiments")
    assert perm == EDIT

    # Different user should fall back to wildcard resource entry
    perm = store.get_workspace_permission(workspace, other_user, "registered_models")
    assert perm == MANAGE

    # No entries -> returns None
    assert store.get_workspace_permission("missing", username, "experiments") is None

    with pytest.raises(MlflowException, match="Invalid resource type"):
        store.get_workspace_permission(workspace, username, "dashboards")


def test_list_workspace_permissions(store):
    workspace = "team-gamma"
    other_workspace = "team-delta"
    username = random_str()

    p1 = store.set_workspace_permission(workspace, username, "experiments", READ.name)
    p2 = store.set_workspace_permission(workspace, username, "registered_models", EDIT.name)
    p3 = store.set_workspace_permission(other_workspace, username, "*", MANAGE.name)

    perms = store.list_workspace_permissions(workspace)
    actual = {
        (perm.workspace, perm.username, perm.resource_type, perm.permission) for perm in perms
    }
    expected = {
        (p1.workspace, p1.username, p1.resource_type, p1.permission),
        (p2.workspace, p2.username, p2.resource_type, p2.permission),
    }
    assert actual == expected

    perms_other = store.list_workspace_permissions(other_workspace)
    assert {
        (perm.workspace, perm.username, perm.resource_type, perm.permission) for perm in perms_other
    } == {(p3.workspace, p3.username, p3.resource_type, p3.permission)}


def test_list_user_workspace_permissions_includes_wildcards(store):
    username = random_str()
    workspace1 = "workspace-1"
    workspace2 = "workspace-2"

    p1 = store.set_workspace_permission(workspace1, username, "experiments", READ.name)
    p2 = store.set_workspace_permission(workspace2, "*", "experiments", EDIT.name)

    perms = store.list_user_workspace_permissions(username)
    actual = {
        (perm.workspace, perm.username, perm.resource_type, perm.permission) for perm in perms
    }
    expected = {
        (p1.workspace, p1.username, p1.resource_type, p1.permission),
        (p2.workspace, p2.username, p2.resource_type, p2.permission),
    }
    assert actual == expected


def test_delete_workspace_permission(store):
    workspace = "workspace-delete"
    username = random_str()

    store.set_workspace_permission(workspace, username, "experiments", READ.name)

    store.delete_workspace_permission(workspace, username, "experiments")
    assert store.get_workspace_permission(workspace, username, "experiments") is None

    with pytest.raises(
        MlflowException,
        match=(
            "Workspace permission does not exist for "
            f"workspace='{workspace}', username='{username}', resource_type='experiments'"
        ),
    ):
        store.delete_workspace_permission(workspace, username, "experiments")


def test_delete_workspace_permissions_for_workspace(store):
    workspace = "workspace-delete-all"
    other_workspace = "workspace-keep"
    username = random_str()

    store.set_workspace_permission(workspace, username, "experiments", READ.name)
    store.set_workspace_permission(workspace, username, "registered_models", READ.name)
    store.set_workspace_permission(other_workspace, username, "*", EDIT.name)

    store.delete_workspace_permissions_for_workspace(workspace)

    assert store.list_workspace_permissions(workspace) == []
    remaining = store.list_workspace_permissions(other_workspace)
    assert len(remaining) == 1
    assert remaining[0].workspace == other_workspace


def test_list_accessible_workspace_names(store):
    username = random_str()
    other_user = random_str()

    store.set_workspace_permission("workspace-read", username, "*", READ.name)
    store.set_workspace_permission("workspace-edit", username, "experiments", EDIT.name)
    store.set_workspace_permission("workspace-no-access", username, "*", NO_PERMISSIONS.name)
    store.set_workspace_permission("workspace-wildcard", "*", "*", READ.name)
    store.set_workspace_permission("workspace-other", other_user, "*", READ.name)

    accessible = store.list_accessible_workspace_names(username)
    assert accessible == {"workspace-read", "workspace-edit", "workspace-wildcard"}

    assert store.list_accessible_workspace_names(other_user) == {
        "workspace-other",
        "workspace-wildcard",
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


def test_rename_registered_model_permission(store):
    # create 2 users and create 2 permission for the model registry with the same name
    model_name = random_str()
    username1 = random_str()
    password1 = random_str()
    _user_maker(store, username1, password1)
    _rmp_maker(store, model_name, username1, MANAGE.name)

    username2 = random_str()
    password2 = random_str()
    _user_maker(store, username2, password2)
    _rmp_maker(store, model_name, username2, READ.name)

    new_name = random_str()

    store.rename_registered_model_permissions(model_name, new_name)

    # get permission by model registry new name and all user must have the same new name
    perm_user_1 = store.get_registered_model_permission(new_name, username1)
    perm_user_2 = store.get_registered_model_permission(new_name, username2)
    assert isinstance(perm_user_1, RegisteredModelPermission)
    assert isinstance(perm_user_2, RegisteredModelPermission)
    assert perm_user_1.name == new_name
    assert perm_user_2.name == new_name

    assert perm_user_1.permission == MANAGE.name
    assert perm_user_1.workspace == DEFAULT_WORKSPACE_NAME
    assert perm_user_2.permission == READ.name
    assert perm_user_2.workspace == DEFAULT_WORKSPACE_NAME


def test_create_scorer_permission(store):
    username1 = random_str()
    password1 = random_str()
    user1 = _user_maker(store, username1, password1)

    experiment_id1 = random_str()
    scorer_name1 = random_str()
    user_id1 = user1.id
    permission1 = READ.name
    sp1 = _sp_maker(store, experiment_id1, scorer_name1, username1, permission1)
    assert sp1.experiment_id == experiment_id1
    assert sp1.scorer_name == scorer_name1
    assert sp1.user_id == user_id1
    assert sp1.permission == permission1

    with pytest.raises(
        MlflowException,
        match=rf"Scorer permission \(experiment_id={experiment_id1}, scorer_name={scorer_name1}, "
        rf"username={username1}\) already exists",
    ) as exception_context:
        _sp_maker(store, experiment_id1, scorer_name1, username1, permission1)
    assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_ALREADY_EXISTS)

    experiment_id2 = random_str()
    sp2 = _sp_maker(store, experiment_id2, scorer_name1, username1, permission1)
    assert sp2.experiment_id == experiment_id2
    assert sp2.scorer_name == scorer_name1
    assert sp2.user_id == user_id1
    assert sp2.permission == permission1

    for perm in ALL_PERMISSIONS:
        experiment_id3 = random_str()
        scorer_name3 = random_str()
        sp3 = _sp_maker(store, experiment_id3, scorer_name3, username1, perm)
        assert sp3.experiment_id == experiment_id3
        assert sp3.scorer_name == scorer_name3
        assert sp3.user_id == user_id1
        assert sp3.permission == perm

    experiment_id4 = random_str()
    scorer_name4 = random_str()
    with pytest.raises(MlflowException, match=r"Invalid permission") as exception_context:
        _sp_maker(store, experiment_id4, scorer_name4, username1, "some_invalid_permission_string")
    assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_get_scorer_permission(store):
    username1 = random_str()
    password1 = random_str()
    user1 = _user_maker(store, username1, password1)

    experiment_id1 = random_str()
    scorer_name1 = random_str()
    user_id1 = user1.id
    permission1 = READ.name
    _sp_maker(store, experiment_id1, scorer_name1, username1, permission1)
    sp1 = store.get_scorer_permission(experiment_id1, scorer_name1, username1)
    assert sp1.experiment_id == experiment_id1
    assert sp1.scorer_name == scorer_name1
    assert sp1.user_id == user_id1
    assert sp1.permission == permission1

    experiment_id2 = random_str()
    with pytest.raises(
        MlflowException,
        match=rf"Scorer permission with experiment_id={experiment_id2}, "
        rf"scorer_name={scorer_name1}, and username={username1} not found",
    ) as exception_context:
        store.get_scorer_permission(experiment_id2, scorer_name1, username1)
    assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_list_scorer_permission(store):
    username1 = random_str()
    password1 = random_str()
    _user_maker(store, username1, password1)

    experiment_id1 = random_str()
    scorer_name1 = random_str()
    permission1 = READ.name
    _sp_maker(store, experiment_id1, scorer_name1, username1, permission1)

    experiment_id2 = random_str()
    scorer_name2 = random_str()
    permission2 = EDIT.name
    _sp_maker(store, experiment_id2, scorer_name2, username1, permission2)

    sps = store.list_scorer_permissions(username1)
    assert len(sps) == 2
    assert isinstance(sps[0], ScorerPermission)
    assert isinstance(sps[1], ScorerPermission)


def test_update_scorer_permission(store):
    username1 = random_str()
    password1 = random_str()
    user1 = _user_maker(store, username1, password1)

    experiment_id1 = random_str()
    scorer_name1 = random_str()
    user_id1 = user1.id
    permission1 = READ.name
    _sp_maker(store, experiment_id1, scorer_name1, username1, permission1)

    permission2 = MANAGE.name
    sp2 = store.update_scorer_permission(experiment_id1, scorer_name1, username1, permission2)
    assert sp2.experiment_id == experiment_id1
    assert sp2.scorer_name == scorer_name1
    assert sp2.user_id == user_id1
    assert sp2.permission == permission2


def test_delete_scorer_permission(store):
    username1 = random_str()
    password1 = random_str()
    _user_maker(store, username1, password1)

    experiment_id1 = random_str()
    scorer_name1 = random_str()
    permission1 = READ.name
    _sp_maker(store, experiment_id1, scorer_name1, username1, permission1)

    store.delete_scorer_permission(experiment_id1, scorer_name1, username1)

    with pytest.raises(
        MlflowException,
        match=rf"Scorer permission with experiment_id={experiment_id1}, "
        rf"scorer_name={scorer_name1}, and username={username1} not found",
    ) as exception_context:
        store.get_scorer_permission(experiment_id1, scorer_name1, username1)
    assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_delete_scorer_permissions_for_scorer(store):
    username1 = random_str()
    password1 = random_str()
    _user_maker(store, username1, password1)

    username2 = random_str()
    password2 = random_str()
    _user_maker(store, username2, password2)

    experiment_id1 = random_str()
    scorer_name1 = random_str()
    _sp_maker(store, experiment_id1, scorer_name1, username1, MANAGE.name)
    _sp_maker(store, experiment_id1, scorer_name1, username2, READ.name)

    store.delete_scorer_permissions_for_scorer(experiment_id1, scorer_name1)

    with pytest.raises(MlflowException, match=r"not found") as exception_context:
        store.get_scorer_permission(experiment_id1, scorer_name1, username1)
    assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)

    with pytest.raises(MlflowException, match=r"not found") as exception_context:
        store.get_scorer_permission(experiment_id1, scorer_name1, username2)
    assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


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
