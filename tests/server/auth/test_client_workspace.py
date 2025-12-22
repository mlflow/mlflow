from contextlib import contextmanager

import pytest
import requests

from mlflow import MlflowException
from mlflow.environment_variables import (
    MLFLOW_ENABLE_WORKSPACES,
    MLFLOW_FLASK_SERVER_SECRET_KEY,
    MLFLOW_TRACKING_PASSWORD,
    MLFLOW_TRACKING_USERNAME,
    MLFLOW_WORKSPACE_STORE_URI,
)
from mlflow.protos.databricks_pb2 import PERMISSION_DENIED, UNAUTHENTICATED, ErrorCode
from mlflow.server.auth.client import AuthServiceClient
from mlflow.utils.os import is_windows

from tests.helper_functions import random_str
from tests.server.auth.auth_test_utils import (
    ADMIN_PASSWORD,
    ADMIN_USERNAME,
    User,
    create_user,
)
from tests.tracking.integration_test_utils import _init_server


@pytest.fixture(autouse=True)
def clear_credentials(monkeypatch):
    monkeypatch.delenv(MLFLOW_TRACKING_USERNAME.name, raising=False)
    monkeypatch.delenv(MLFLOW_TRACKING_PASSWORD.name, raising=False)


@pytest.fixture
def workspace_client(tmp_path):
    path = tmp_path.joinpath("sqlalchemy.db").as_uri()
    backend_uri = ("sqlite://" if is_windows() else "sqlite:////") + path[len("file://") :]

    with _init_server(
        backend_uri=backend_uri,
        root_artifact_uri=tmp_path.joinpath("artifacts").as_uri(),
        app="mlflow.server.auth:create_app",
        extra_env={
            MLFLOW_FLASK_SERVER_SECRET_KEY.name: "my-secret-key",
            MLFLOW_ENABLE_WORKSPACES.name: "true",
            MLFLOW_WORKSPACE_STORE_URI.name: backend_uri,
        },
        server_type="flask",
    ) as url:
        yield AuthServiceClient(url), url


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


def _create_workspace(tracking_uri: str, workspace_name: str):
    response = requests.post(
        f"{tracking_uri}/api/2.0/mlflow/workspaces",
        json={"name": workspace_name},
        auth=(ADMIN_USERNAME, ADMIN_PASSWORD),
    )
    response.raise_for_status()


@pytest.fixture
def workspace_setup(workspace_client):
    client, tracking_uri = workspace_client
    workspace_name = f"team-{random_str()}"
    _create_workspace(tracking_uri, workspace_name)
    username, password = create_user(tracking_uri)
    return client, tracking_uri, workspace_name, username, password


def test_workspace_permission_set_and_list(workspace_setup, monkeypatch):
    client, _tracking_uri, workspace_name, username, _password = workspace_setup

    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        perm = client.set_workspace_permission(workspace_name, username, "experiments", "MANAGE")
    assert perm.workspace == workspace_name
    assert perm.username == username
    assert perm.resource_type == "experiments"
    assert perm.permission == "MANAGE"

    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        perms = client.list_workspace_permissions(workspace_name)
        assert any(p.username == username and p.resource_type == "experiments" for p in perms)

        user_perms = client.list_user_workspace_permissions(username)
        assert any(
            p.workspace == workspace_name and p.resource_type == "experiments" for p in user_perms
        )

        client.delete_workspace_permission(workspace_name, username, "experiments")
        assert client.list_workspace_permissions(workspace_name) == []


def test_workspace_permission_list_requires_authentication(workspace_setup):
    client, _tracking_uri, workspace_name, _username, _password = workspace_setup

    with assert_unauthenticated():
        client.list_workspace_permissions(workspace_name)


def test_workspace_permission_list_requires_admin(workspace_setup, monkeypatch):
    client, _tracking_uri, workspace_name, username, password = workspace_setup

    with User(username, password, monkeypatch), assert_unauthorized():
        client.list_workspace_permissions(workspace_name)


def test_workspace_permission_set_requires_admin(workspace_client, monkeypatch):
    client, tracking_uri = workspace_client
    workspace_name = "team-b"
    _create_workspace(tracking_uri, workspace_name)
    username, password = create_user(tracking_uri)

    with User(username, password, monkeypatch), assert_unauthorized():
        client.set_workspace_permission(workspace_name, username, "experiments", "READ")

    with User(username, password, monkeypatch), assert_unauthorized():
        client.delete_workspace_permission(workspace_name, username, "experiments")
