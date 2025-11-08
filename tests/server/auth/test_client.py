import os
from contextlib import contextmanager

import pytest

import mlflow
from mlflow import MlflowException
from mlflow.environment_variables import (
    MLFLOW_FLASK_SERVER_SECRET_KEY,
    MLFLOW_TRACKING_PASSWORD,
    MLFLOW_TRACKING_USERNAME,
)
from mlflow.protos.databricks_pb2 import (
    PERMISSION_DENIED,
    RESOURCE_DOES_NOT_EXIST,
    UNAUTHENTICATED,
    ErrorCode,
)
from mlflow.server.auth import auth_config
from mlflow.server.auth.client import AuthServiceClient
from mlflow.utils.os import is_windows

from tests.helper_functions import random_str
from tests.server.auth.auth_test_utils import (
    ADMIN_PASSWORD,
    ADMIN_USERNAME,
    NEW_PERMISSION,
    PERMISSION,
    User,
    create_user,
)
from tests.tracking.integration_test_utils import _init_server


@pytest.fixture(autouse=True)
def clear_credentials(monkeypatch):
    monkeypatch.delenv(MLFLOW_TRACKING_USERNAME.name, raising=False)
    monkeypatch.delenv(MLFLOW_TRACKING_PASSWORD.name, raising=False)


@pytest.fixture
def client(tmp_path):
    # clean up users & permissions created from previous tests
    db_file = os.path.abspath(os.path.basename(auth_config.database_uri))
    if os.path.exists(db_file):
        os.remove(db_file)

    path = tmp_path.joinpath("sqlalchemy.db").as_uri()
    backend_uri = ("sqlite://" if is_windows() else "sqlite:////") + path[len("file://") :]

    with _init_server(
        backend_uri=backend_uri,
        root_artifact_uri=tmp_path.joinpath("artifacts").as_uri(),
        app="mlflow.server.auth:create_app",
        extra_env={MLFLOW_FLASK_SERVER_SECRET_KEY.name: "my-secret-key"},
        server_type="flask",
    ) as url:
        yield AuthServiceClient(url)


@contextmanager
def assert_unauthenticated():
    with pytest.raises(MlflowException, match=r"You are not authenticated.") as exception_context:
        yield
    assert exception_context.value.error_code == ErrorCode.Name(UNAUTHENTICATED)


@contextmanager
def assert_unauthorized():
    with pytest.raises(MlflowException, match=r"Permission denied.") as exception_context:
        yield
    assert exception_context.value.error_code == ErrorCode.Name(PERMISSION_DENIED)


def test_get_client():
    client = mlflow.server.get_app_client("basic-auth", "uri:/fake")
    assert isinstance(client, AuthServiceClient)


def test_create_user(client, monkeypatch):
    username = random_str()
    password = random_str()

    with assert_unauthenticated():
        client.create_user(username, password)

    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        user = client.create_user(username, password)
    assert user.username == username
    assert user.is_admin is False

    username2 = random_str()
    password2 = random_str()
    with User(username, password, monkeypatch), assert_unauthorized():
        client.create_user(username2, password2)


def test_get_user(client, monkeypatch):
    username = random_str()
    password = random_str()
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.create_user(username, password)

    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        user = client.get_user(username)
    assert user.username == username

    with assert_unauthenticated():
        client.get_user(username)

    username2 = random_str()
    password2 = random_str()
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.create_user(username2, password2)
    with User(username2, password2, monkeypatch), assert_unauthorized():
        client.get_user(username)


def test_update_user_password(client, monkeypatch):
    username = random_str()
    password = random_str()
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.create_user(username, password)

    new_password = random_str()
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.update_user_password(username, new_password)

    with User(username, password, monkeypatch), assert_unauthenticated():
        client.get_user(username)

    with User(username, new_password, monkeypatch):
        client.get_user(username)

    with assert_unauthenticated():
        client.update_user_password(username, new_password)

    username2 = random_str()
    password2 = random_str()
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.create_user(username2, password2)
    with User(username2, password2, monkeypatch), assert_unauthorized():
        client.update_user_password(username, new_password)


def test_update_user_admin(client, monkeypatch):
    username = random_str()
    password = random_str()
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.create_user(username, password)

    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.update_user_admin(username, True)
        user = client.get_user(username)
        assert user.is_admin is True

    with assert_unauthenticated():
        client.update_user_admin(username, True)

    username2 = random_str()
    password2 = random_str()
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.create_user(username2, password2)
    with User(username2, password2, monkeypatch), assert_unauthorized():
        client.update_user_admin(username, True)


def test_delete_user(client, monkeypatch):
    username = random_str()
    password = random_str()
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.create_user(username, password)

    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.update_user_admin(username, True)
        client.delete_user(username)
        with pytest.raises(
            MlflowException,
            match=rf"User with username={username} not found",
        ) as exception_context:
            client.get_user(username)
        assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)

    with assert_unauthenticated():
        client.delete_user(username)

    username2 = random_str()
    password2 = random_str()
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.create_user(username2, password2)
    with User(username2, password2, monkeypatch), assert_unauthorized():
        client.delete_user(username)


def test_client_create_experiment_permission(client, monkeypatch):
    experiment_id = random_str()
    username, password = create_user(client.tracking_uri)

    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        ep = client.create_experiment_permission(experiment_id, username, PERMISSION)
    assert ep.experiment_id == experiment_id
    assert ep.permission == PERMISSION

    with assert_unauthenticated():
        client.create_experiment_permission(experiment_id, username, PERMISSION)

    with User(username, password, monkeypatch), assert_unauthorized():
        client.create_experiment_permission(experiment_id, username, PERMISSION)


def test_client_get_experiment_permission(client, monkeypatch):
    experiment_id = random_str()
    username, password = create_user(client.tracking_uri)

    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.create_experiment_permission(experiment_id, username, PERMISSION)
        ep = client.get_experiment_permission(experiment_id, username)
    assert ep.experiment_id == experiment_id
    assert ep.permission == PERMISSION

    with assert_unauthenticated():
        client.get_experiment_permission(experiment_id, username)

    with User(username, password, monkeypatch), assert_unauthorized():
        client.get_experiment_permission(experiment_id, username)


def test_client_update_experiment_permission(client, monkeypatch):
    experiment_id = random_str()
    username, password = create_user(client.tracking_uri)

    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.create_experiment_permission(experiment_id, username, PERMISSION)
        client.update_experiment_permission(experiment_id, username, NEW_PERMISSION)
        ep = client.get_experiment_permission(experiment_id, username)
    assert ep.experiment_id == experiment_id
    assert ep.permission == NEW_PERMISSION

    with assert_unauthenticated():
        client.update_experiment_permission(experiment_id, username, PERMISSION)

    with User(username, password, monkeypatch), assert_unauthorized():
        client.update_experiment_permission(experiment_id, username, PERMISSION)


def test_client_delete_experiment_permission(client, monkeypatch):
    experiment_id = random_str()
    username, password = create_user(client.tracking_uri)

    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.create_experiment_permission(experiment_id, username, PERMISSION)
        client.delete_experiment_permission(experiment_id, username)
        with pytest.raises(
            MlflowException,
            match=rf"Experiment permission with experiment_id={experiment_id} "
            rf"and username={username} not found",
        ) as exception_context:
            client.get_experiment_permission(experiment_id, username)
        assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)

    with assert_unauthenticated():
        client.delete_experiment_permission(experiment_id, username)

    with User(username, password, monkeypatch), assert_unauthorized():
        client.delete_experiment_permission(experiment_id, username)


def test_client_create_registered_model_permission(client, monkeypatch):
    name = random_str()
    username, password = create_user(client.tracking_uri)

    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        rmp = client.create_registered_model_permission(name, username, PERMISSION)
    assert rmp.name == name
    assert rmp.permission == PERMISSION

    with assert_unauthenticated():
        client.create_registered_model_permission(name, username, PERMISSION)

    with User(username, password, monkeypatch), assert_unauthorized():
        client.create_registered_model_permission(name, username, PERMISSION)


def test_client_get_registered_model_permission(client, monkeypatch):
    name = random_str()
    username, password = create_user(client.tracking_uri)

    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.create_registered_model_permission(name, username, PERMISSION)
        rmp = client.get_registered_model_permission(name, username)
    assert rmp.name == name
    assert rmp.permission == PERMISSION

    with assert_unauthenticated():
        client.get_registered_model_permission(name, username)

    with User(username, password, monkeypatch), assert_unauthorized():
        client.get_registered_model_permission(name, username)


def test_client_update_registered_model_permission(client, monkeypatch):
    name = random_str()
    username, password = create_user(client.tracking_uri)

    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.create_registered_model_permission(name, username, PERMISSION)
        client.update_registered_model_permission(name, username, NEW_PERMISSION)
        rmp = client.get_registered_model_permission(name, username)
    assert rmp.name == name
    assert rmp.permission == NEW_PERMISSION

    with assert_unauthenticated():
        client.update_registered_model_permission(name, username, PERMISSION)

    with User(username, password, monkeypatch), assert_unauthorized():
        client.update_registered_model_permission(name, username, PERMISSION)


def test_client_delete_registered_model_permission(client, monkeypatch):
    name = random_str()
    username, password = create_user(client.tracking_uri)

    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.create_registered_model_permission(name, username, PERMISSION)
        client.delete_registered_model_permission(name, username)
        with pytest.raises(
            MlflowException,
            match=rf"Registered model permission with name={name} "
            rf"and username={username} not found",
        ) as exception_context:
            client.get_registered_model_permission(name, username)
        assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)

    with assert_unauthenticated():
        client.delete_registered_model_permission(name, username)

    with User(username, password, monkeypatch), assert_unauthorized():
        client.delete_registered_model_permission(name, username)
