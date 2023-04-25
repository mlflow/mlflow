import os
import pytest

from mlflow.exceptions import MlflowException
from mlflow.server.auth import (
    _AUTH_CONFIG_PATH_ENV_VAR,
)
from mlflow.server.auth.config import read_auth_config
from mlflow.server.auth.entities import User
from mlflow.server.auth.sqlalchemy_store import (
    SqlUser,
    SqlExperimentPermission,
    SqlRegisteredModelPermission,
    SqlAlchemyStore,
)
from mlflow.protos.databricks_pb2 import (
    ErrorCode,
    RESOURCE_DOES_NOT_EXIST,
    INVALID_PARAMETER_VALUE,
    RESOURCE_ALREADY_EXISTS,
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
