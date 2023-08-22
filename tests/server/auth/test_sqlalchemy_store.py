import pytest

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)
from mlflow.server.auth.entities import ExperimentPermission, RegisteredModelPermission, User
from mlflow.server.auth.permissions import (
    ALL_PERMISSIONS,
    EDIT,
    READ,
)
from mlflow.server.auth.sqlalchemy_store import SqlAlchemyStore

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

    # error on duplicate
    with pytest.raises(
        MlflowException,
        match=rf"Registered model permission \(name={name1}, username={username1}\) already exists",
    ) as exception_context:
        _rmp_maker(store, name1, username1, permission1)
    assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_ALREADY_EXISTS)

    # slightly different name is ok
    name2 = random_str()
    rmp2 = _rmp_maker(store, name2, username1, permission1)
    assert rmp2.name == name2
    assert rmp2.user_id == user_id1
    assert rmp2.permission == permission1

    # all permissions are ok
    for perm in ALL_PERMISSIONS:
        name3 = random_str()
        rmp3 = _rmp_maker(store, name3, username1, perm)
        assert rmp3.name == name3
        assert rmp3.user_id == user_id1
        assert rmp3.permission == perm

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

    # error on non-existent row
    name2 = random_str()
    with pytest.raises(
        MlflowException,
        match=rf"Registered model permission with name={name2} and username={username1} not found",
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
    assert rmps[1].name == name2
    assert rmps[2].name == name3


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
    with pytest.raises(
        MlflowException,
        match=rf"Registered model permission with name={name1} and username={username1} not found",
    ) as exception_context:
        store.get_registered_model_permission(name1, username1)
    assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
