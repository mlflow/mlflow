import os

import pytest

import mlflow
from mlflow import MlflowException
from mlflow.protos.databricks_pb2 import ErrorCode, RESOURCE_DOES_NOT_EXIST
from mlflow.server.auth import auth_config
from mlflow.server.auth.client import AuthServiceClient
from mlflow.utils.os import is_windows
from tests.helper_functions import random_str
from tests.server.auth.auth_test_utils import create_user, User
from tests.tracking.integration_test_utils import _init_server, _terminate_server

PERMISSION = "READ"
NEW_PERMISSION = "EDIT"
ADMIN_USERNAME = auth_config.admin_username
ADMIN_PASSWORD = auth_config.admin_password


@pytest.fixture
def client(tmp_path):
    path = tmp_path.joinpath("sqlalchemy.db").as_uri()
    backend_uri = ("sqlite://" if is_windows() else "sqlite:////") + path[len("file://") :]

    url, process = _init_server(
        backend_uri=backend_uri,
        root_artifact_uri=tmp_path.joinpath("artifacts").as_uri(),
        app_module="mlflow.server.auth",
    )
    yield AuthServiceClient(url)
    _terminate_server(process)
    # clean up users & permissions created in the tests
    db_file = os.path.abspath(os.path.basename(auth_config.database_uri))
    os.remove(db_file)


def test_get_client():
    client = mlflow.server.get_app_client("basic-auth", "uri:/fake")
    assert isinstance(client, AuthServiceClient)


def test_client_create_experiment_permission(client, monkeypatch):
    experiment_id = random_str()
    username, _ = create_user(client.tracking_uri)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        ep = client.create_experiment_permission(experiment_id, username, PERMISSION)

    assert ep.experiment_id == experiment_id
    assert ep.permission == PERMISSION


def test_client_get_experiment_permission(client, monkeypatch):
    experiment_id = random_str()
    username, _ = create_user(client.tracking_uri)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.create_experiment_permission(experiment_id, username, PERMISSION)
        ep = client.get_experiment_permission(experiment_id, username)

    assert ep.experiment_id == experiment_id
    assert ep.permission == PERMISSION


def test_client_update_experiment_permission(client, monkeypatch):
    experiment_id = random_str()
    username, _ = create_user(client.tracking_uri)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.create_experiment_permission(experiment_id, username, PERMISSION)
        client.update_experiment_permission(experiment_id, username, NEW_PERMISSION)
        ep = client.get_experiment_permission(experiment_id, username)

    assert ep.experiment_id == experiment_id
    assert ep.permission == NEW_PERMISSION


def test_client_delete_experiment_permission(client, monkeypatch):
    experiment_id = random_str()
    username, _ = create_user(client.tracking_uri)
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


def test_client_create_registered_model_permission(client, monkeypatch):
    name = random_str()
    username, _ = create_user(client.tracking_uri)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        rmp = client.create_registered_model_permission(name, username, PERMISSION)

    assert rmp.name == name
    assert rmp.permission == PERMISSION


def test_client_get_registered_model_permission(client, monkeypatch):
    name = random_str()
    username, _ = create_user(client.tracking_uri)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.create_registered_model_permission(name, username, PERMISSION)
        rmp = client.get_registered_model_permission(name, username)

    assert rmp.name == name
    assert rmp.permission == PERMISSION


def test_client_update_registered_model_permission(client, monkeypatch):
    name = random_str()
    username, _ = create_user(client.tracking_uri)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.create_registered_model_permission(name, username, PERMISSION)
        client.update_registered_model_permission(name, username, NEW_PERMISSION)
        rmp = client.get_registered_model_permission(name, username)

    assert rmp.name == name
    assert rmp.permission == NEW_PERMISSION


def test_client_delete_registered_model_permission(client, monkeypatch):
    name = random_str()
    username, _ = create_user(client.tracking_uri)
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
