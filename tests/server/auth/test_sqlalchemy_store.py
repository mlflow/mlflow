import os
import pytest

from mlflow.exceptions import MlflowException
from mlflow.server.auth import (
    _AUTH_CONFIG_PATH_ENV_VAR,
)
from mlflow.server.auth.config import read_auth_config
from mlflow.server.auth.entities import User, ExperimentPermission
from mlflow.server.auth.sqlalchemy_store import (
    SqlUser,
    SqlExperimentPermission,
    SqlRegisteredModelPermission,
    SqlAlchemyStore,
)
from mlflow.server.auth.permissions import (
    READ,
    EDIT,
    ALL_PERMISSIONS,
)
from mlflow.protos.databricks_pb2 import (
    ErrorCode,
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
)
from tests.helper_functions import random_str

pytestmark = pytest.mark.notrackingurimock


def _get_db_uri_from_env_var():
    auth_config_path = os.environ.get(_AUTH_CONFIG_PATH_ENV_VAR)
    if auth_config_path:
        auth_config = read_auth_config(auth_config_path)
        return auth_config.database_uri


@pytest.fixture
def store(tmp_sqlite_uri):
    db_uri_from_env_var = _get_db_uri_from_env_var()
    store = SqlAlchemyStore()
    store.init_db(db_uri_from_env_var if db_uri_from_env_var else tmp_sqlite_uri)
    yield store

    if db_uri_from_env_var is not None:
        with store.ManagedSessionMaker() as session:
            for model in (
                SqlUser,
                SqlRegisteredModelPermission,
                SqlExperimentPermission,
            ):
                session.query(model).delete()


def _user_maker(store, username, password, is_admin=False):
    return store.create_user(username, password, is_admin)


def _ep_maker(store, experiment_id, user_id, permission):
    return store.create_experiment_permission(experiment_id, user_id, permission)


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


def test_create_experiment_permission(store):
    username1 = random_str()
    password1 = random_str()
    user1 = _user_maker(store, username1, password1)

    experiment_id1 = random_str()
    user_id1 = user1.id
    permission1 = READ.name
    ep1 = _ep_maker(store, experiment_id1, user_id1, permission1)
    assert ep1.experiment_id == experiment_id1
    assert ep1.user_id == user_id1
    assert ep1.permission == permission1

    # error on duplicate
    with pytest.raises(
        MlflowException, match=r"Experiment permission creation error"
    ) as exception_context:
        _ep_maker(store, experiment_id1, user_id1, permission1)
    assert exception_context.value.error_code == ErrorCode.Name(INTERNAL_ERROR)

    # slightly different name is ok
    experiment_id2 = random_str()
    ep2 = _ep_maker(store, experiment_id2, user_id1, permission1)
    assert ep2.experiment_id == experiment_id2
    assert ep2.user_id == user_id1
    assert ep2.permission == permission1

    # all permissions are ok
    for perm in ALL_PERMISSIONS:
        experiment_id3 = random_str()
        ep3 = _ep_maker(store, experiment_id3, user_id1, perm)
        assert ep3.experiment_id == experiment_id3
        assert ep3.user_id == user_id1
        assert ep3.permission == perm

    # invalid permission will fail
    experiment_id4 = random_str()
    with pytest.raises(MlflowException, match=r"Invalid permission") as exception_context:
        _ep_maker(store, experiment_id4, user_id1, "some_invalid_permission_string")
    assert exception_context.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_get_experiment_permission(store):
    username1 = random_str()
    password1 = random_str()
    user1 = _user_maker(store, username1, password1)

    experiment_id1 = random_str()
    user_id1 = user1.id
    permission1 = READ.name
    _ep_maker(store, experiment_id1, user_id1, permission1)
    ep1 = store.get_experiment_permission(experiment_id1, user_id1)
    assert isinstance(ep1, ExperimentPermission)
    assert ep1.experiment_id == experiment_id1
    assert ep1.user_id == user_id1
    assert ep1.permission == permission1

    # error on non-existent row
    user_id2 = random_str()
    with pytest.raises(
        MlflowException,
        match=rf"Experiment permission with experiment_id={experiment_id1} and user_id={user_id2} not found",
    ) as exception_context:
        store.get_experiment_permission(experiment_id1, user_id2)
    assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_list_experiment_permission(store):
    username1 = random_str()
    password1 = random_str()
    user1 = _user_maker(store, username1, password1)

    experiment_id1 = "1" + random_str()
    _ep_maker(store, experiment_id1, user1.id, READ.name)

    experiment_id2 = "2" + random_str()
    _ep_maker(store, experiment_id2, user1.id, READ.name)

    experiment_id3 = "3" + random_str()
    _ep_maker(store, experiment_id3, user1.id, READ.name)

    eps = store.list_experiment_permissions(user1.id)
    eps.sort(key=lambda ep: ep.experiment_id)

    assert len(eps) == 3
    assert isinstance(eps[0], ExperimentPermission)
    assert eps[0].experiment_id == experiment_id1
    assert eps[1].experiment_id == experiment_id2
    assert eps[2].experiment_id == experiment_id3


def test_update_experiment_permission(store):
    username1 = random_str()
    password1 = random_str()
    user1 = _user_maker(store, username1, password1)

    experiment_id1 = random_str()
    user_id1 = user1.id
    permission1 = READ.name
    _ep_maker(store, experiment_id1, user_id1, permission1)

    permission2 = EDIT.name
    store.update_experiment_permission(experiment_id1, user_id1, permission2)
    ep1 = store.get_experiment_permission(experiment_id1, user_id1)
    assert ep1.permission == permission2


def test_delete_experiment_permission(store):
    username1 = random_str()
    password1 = random_str()
    user1 = _user_maker(store, username1, password1)

    experiment_id1 = random_str()
    user_id1 = user1.id
    permission1 = READ.name
    _ep_maker(store, experiment_id1, user_id1, permission1)

    store.delete_experiment_permission(experiment_id1, user_id1)
    with pytest.raises(
        MlflowException,
        match=rf"Experiment permission with experiment_id={experiment_id1} and user_id={user_id1} not found",
    ) as exception_context:
        store.get_experiment_permission(experiment_id1, user_id1)
    assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
