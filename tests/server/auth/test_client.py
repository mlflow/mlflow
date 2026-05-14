from contextlib import contextmanager

import pytest
import requests

import mlflow
from mlflow import MlflowException
from mlflow.environment_variables import (
    MLFLOW_AUTH_CONFIG_PATH,
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
from mlflow.server.auth.client import AuthServiceClient
from mlflow.utils.os import is_windows

from tests.helper_functions import random_str
from tests.server.auth.auth_test_utils import (
    ADMIN_PASSWORD,
    ADMIN_USERNAME,
    User,
    write_isolated_auth_config,
)
from tests.tracking.integration_test_utils import _init_server


@pytest.fixture(autouse=True)
def clear_credentials(monkeypatch):
    monkeypatch.delenv(MLFLOW_TRACKING_USERNAME.name, raising=False)
    monkeypatch.delenv(MLFLOW_TRACKING_PASSWORD.name, raising=False)


@pytest.fixture
def client(tmp_path):
    auth_config_path = write_isolated_auth_config(tmp_path)
    path = tmp_path.joinpath("sqlalchemy.db").as_uri()
    backend_uri = ("sqlite://" if is_windows() else "sqlite:////") + path[len("file://") :]

    with _init_server(
        backend_uri=backend_uri,
        root_artifact_uri=tmp_path.joinpath("artifacts").as_uri(),
        app="mlflow.server.auth:create_app",
        extra_env={
            MLFLOW_FLASK_SERVER_SECRET_KEY.name: "my-secret-key",
            MLFLOW_AUTH_CONFIG_PATH.name: str(auth_config_path),
        },
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


def test_get_current_user(client, monkeypatch):
    # /users/current returns minimal identity for whoever the request is
    # authenticated as. The admin UI relies on the response shape (id,
    # username, is_admin) to gate admin-only links.
    username = random_str()
    password = random_str()
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        created = client.create_user(username, password)

    url = f"{client.tracking_uri}/api/2.0/mlflow/users/current"

    resp = requests.get(url, auth=(username, password))
    assert resp.status_code == 200
    body = resp.json()
    assert body["user"] == {
        "id": created.id,
        "username": username,
        "is_admin": False,
    }
    # ``is_basic_auth`` lets the frontend gate Basic-Auth-only flows
    # (logout XHR, change-password) when a custom authorization_function
    # is configured.
    assert body["is_basic_auth"] is True

    resp = requests.get(url, auth=(ADMIN_USERNAME, ADMIN_PASSWORD))
    assert resp.status_code == 200
    admin_payload = resp.json()["user"]
    assert admin_payload["username"] == ADMIN_USERNAME
    assert admin_payload["is_admin"] is True

    resp = requests.get(url)
    assert resp.status_code == 401


@pytest.mark.parametrize(
    ("path", "method"),
    [
        ("/api/2.0/mlflow/experiments/permissions/get", "GET"),
        ("/api/2.0/mlflow/experiments/permissions/create", "POST"),
        ("/api/2.0/mlflow/experiments/permissions/update", "PATCH"),
        ("/api/2.0/mlflow/experiments/permissions/delete", "DELETE"),
        ("/api/2.0/mlflow/registered-models/permissions/get", "GET"),
        ("/api/2.0/mlflow/registered-models/permissions/create", "POST"),
        ("/api/2.0/mlflow/registered-models/permissions/update", "PATCH"),
        ("/api/2.0/mlflow/registered-models/permissions/delete", "DELETE"),
        ("/api/3.0/mlflow/scorers/permissions/get", "GET"),
        ("/api/3.0/mlflow/scorers/permissions/create", "POST"),
        ("/api/3.0/mlflow/scorers/permissions/update", "PATCH"),
        ("/api/3.0/mlflow/scorers/permissions/delete", "DELETE"),
        ("/api/3.0/mlflow/gateway/secrets/permissions/get", "GET"),
        ("/api/3.0/mlflow/gateway/secrets/permissions/create", "POST"),
        ("/api/3.0/mlflow/gateway/secrets/permissions/update", "PATCH"),
        ("/api/3.0/mlflow/gateway/secrets/permissions/delete", "DELETE"),
        ("/api/3.0/mlflow/gateway/endpoints/permissions/get", "GET"),
        ("/api/3.0/mlflow/gateway/endpoints/permissions/create", "POST"),
        ("/api/3.0/mlflow/gateway/endpoints/permissions/update", "PATCH"),
        ("/api/3.0/mlflow/gateway/endpoints/permissions/delete", "DELETE"),
        ("/api/3.0/mlflow/gateway/model-definitions/permissions/get", "GET"),
        ("/api/3.0/mlflow/gateway/model-definitions/permissions/create", "POST"),
        ("/api/3.0/mlflow/gateway/model-definitions/permissions/update", "PATCH"),
        ("/api/3.0/mlflow/gateway/model-definitions/permissions/delete", "DELETE"),
    ],
)
def test_legacy_permission_endpoints_remain_registered(client, path, method):
    resp = requests.request(
        method, client.tracking_uri + path, auth=(ADMIN_USERNAME, ADMIN_PASSWORD)
    )
    assert resp.status_code != 404, (
        f"{method} {path} unexpectedly returned 404 — legacy permission endpoints "
        "must remain registered for backward compatibility"
    )


def test_legacy_client_methods_emit_deprecation_warning(client, monkeypatch):
    username = random_str()
    password = random_str()
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.create_user(username, password)
        with pytest.warns(FutureWarning, match="create_experiment_permission"):
            client.create_experiment_permission("exp-deprecation", username, "READ")


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


def test_self_service_password_change_requires_current_password(client, monkeypatch):
    # Defense-in-depth: a user changing their own password must re-assert the
    # current password. Admins changing someone else's password don't (and
    # can't) supply it — that path is exercised in `test_update_user_password`.
    username = random_str()
    password = random_str()
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.create_user(username, password)

    new_password = random_str()

    with User(username, password, monkeypatch):
        # Missing current_password: rejected.
        with pytest.raises(MlflowException, match="Current password is required"):
            client.update_user_password(username, new_password)

        # Wrong current_password: rejected.
        with pytest.raises(MlflowException, match="Current password does not match"):
            client.update_user_password(
                username, new_password, current_password="not-the-current-password"
            )

        # Correct current_password: accepted.
        client.update_user_password(username, new_password, current_password=password)

    # Old password no longer authenticates; new one does.
    with User(username, password, monkeypatch), assert_unauthenticated():
        client.get_user(username)
    with User(username, new_password, monkeypatch):
        client.get_user(username)


def test_create_user_with_null_or_missing_json_body_returns_400(client, monkeypatch):
    # Defensive: a JSON-typed POST whose body is literal ``null`` (or empty)
    # used to crash ``_get_request_param`` with ``None | dict`` and surface as
    # a 500. Make sure it's a clean 400 instead.
    url = f"{client.tracking_uri}/api/2.0/mlflow/users/create"
    headers = {"Content-Type": "application/json"}
    auth = (ADMIN_USERNAME, ADMIN_PASSWORD)

    # Literal null body.
    resp = requests.post(url, data="null", headers=headers, auth=auth)
    assert resp.status_code == 400

    # Empty body.
    resp = requests.post(url, data="", headers=headers, auth=auth)
    assert resp.status_code == 400


def test_self_service_password_change_with_null_body_returns_400(client, monkeypatch):
    # Defensive: self-service password changes used to crash with
    # ``request.json.get(...)`` when the body was literal ``null`` (raises
    # AttributeError -> 500). Make sure it's the standard 400 instead.
    username = random_str()
    password = random_str()
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.create_user(username, password)

    url = f"{client.tracking_uri}/api/2.0/mlflow/users/update-password"
    resp = requests.patch(
        url,
        data="null",
        headers={"Content-Type": "application/json"},
        auth=(username, password),
    )
    assert resp.status_code == 400


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
