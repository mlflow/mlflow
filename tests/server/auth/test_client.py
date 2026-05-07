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
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

from tests.helper_functions import random_str
from tests.server.auth.auth_test_utils import (
    ADMIN_PASSWORD,
    ADMIN_USERNAME,
    NEW_PERMISSION,
    PERMISSION,
    User,
    create_user,
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


def test_list_current_user_permissions(client, monkeypatch):
    # /users/current/permissions returns the calling user's direct per-resource
    # grants. Sender == target, so any authenticated user can read their own
    # without an admin gate.
    username = random_str()
    password = random_str()
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.create_user(username, password)
        client.create_experiment_permission("exp-1", username, "EDIT")
        client.create_registered_model_permission("model-x", username, "READ")
        client.create_gateway_secret_permission("secret-1", username, "MANAGE")
        client.create_gateway_endpoint_permission("endpoint-1", username, "READ")
        client.create_gateway_model_definition_permission("md-1", username, "EDIT")

    url = f"{client.tracking_uri}/api/3.0/mlflow/users/current/permissions"

    resp = requests.get(url, auth=(username, password))
    assert resp.status_code == 200
    grants = resp.json()["permissions"]
    # Each entry uses the unified ``resource_pattern`` shape - ready for a
    # future migration that folds these tables into the role/permission model.
    assert {(g["resource_type"], g["resource_pattern"], g["permission"]) for g in grants} == {
        ("experiment", "exp-1", "EDIT"),
        ("registered_model", "model-x", "READ"),
        ("gateway_secret", "secret-1", "MANAGE"),
        ("gateway_endpoint", "endpoint-1", "READ"),
        ("gateway_model_definition", "md-1", "EDIT"),
    }
    # Every grant carries a ``workspace`` field. ``registered_model`` rows
    # have it on the permission row natively; the others are resolved via
    # the resource→workspace lookup. The lookup-based resources don't
    # exist in the tracking store in this test, so their workspace falls
    # back to ``None`` (which is the documented "deleted-or-unknown"
    # path).
    by_type = {g["resource_type"]: g for g in grants}
    assert "workspace" in by_type["registered_model"]
    assert by_type["registered_model"]["workspace"] == DEFAULT_WORKSPACE_NAME
    assert by_type["experiment"]["workspace"] is None
    assert by_type["gateway_secret"]["workspace"] is None
    assert by_type["gateway_endpoint"]["workspace"] is None
    assert by_type["gateway_model_definition"]["workspace"] is None

    # An unrelated user sees their own (empty) list, not the target user's.
    other_username = random_str()
    other_password = random_str()
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.create_user(other_username, other_password)
    resp = requests.get(url, auth=(other_username, other_password))
    assert resp.status_code == 200
    assert resp.json()["permissions"] == []

    # Unauthenticated requests are rejected.
    resp = requests.get(url)
    assert resp.status_code == 401


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
    assert rmp.workspace == DEFAULT_WORKSPACE_NAME

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
    assert rmp.workspace == DEFAULT_WORKSPACE_NAME

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
    assert rmp.workspace == DEFAULT_WORKSPACE_NAME

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
        expected_message = (
            "Registered model permission with "
            f"workspace={DEFAULT_WORKSPACE_NAME}, name={name} "
            f"and username={username} not found"
        )
        with pytest.raises(
            MlflowException,
            match=expected_message,
        ) as exception_context:
            client.get_registered_model_permission(name, username)
        assert exception_context.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)

    with assert_unauthenticated():
        client.delete_registered_model_permission(name, username)

    with User(username, password, monkeypatch), assert_unauthorized():
        client.delete_registered_model_permission(name, username)
