from contextlib import contextmanager

import pytest
import requests

import mlflow
from mlflow import MlflowClient, MlflowException
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


# ---- Unified per-user permission convenience APIs ----


def _create_experiment(tracking_uri: str, monkeypatch, name: str) -> str:
    """Create an experiment as admin so ``check_user_permission`` workspace
    lookup succeeds; without a real row the resolver falls through to default.
    """
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        return MlflowClient(tracking_uri).create_experiment(name)


def _new_user(client, monkeypatch):
    """Create a fresh user as admin and return (username, password)."""
    username = random_str()
    password = random_str()
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.create_user(username, password)
    return username, password


def test_grant_user_permission_roundtrip(client, monkeypatch):
    # Admin grants READ on an experiment to a new user, then the user's
    # synthetic role surfaces the row via list_user_roles.
    username, _ = _new_user(client, monkeypatch)
    exp_id = _create_experiment(client.tracking_uri, monkeypatch, f"grant-{random_str()}")
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.grant_user_permission(username, "experiment", exp_id, "READ")
        roles = client.list_user_roles(username)

    synthetic = [r for r in roles if r.name.startswith("__user_")]
    assert len(synthetic) == 1
    grants = [(p.resource_type, p.resource_pattern, p.permission) for p in synthetic[0].permissions]
    assert ("experiment", exp_id, "READ") in grants


def test_grant_user_permission_duplicate_raises(client, monkeypatch):
    username, _ = _new_user(client, monkeypatch)
    exp_id = _create_experiment(client.tracking_uri, monkeypatch, f"dup-{random_str()}")
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.grant_user_permission(username, "experiment", exp_id, "READ")
        with pytest.raises(MlflowException, match="already exists") as exc:
            client.grant_user_permission(username, "experiment", exp_id, "EDIT")
    from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS

    assert exc.value.error_code == ErrorCode.Name(RESOURCE_ALREADY_EXISTS)


def test_grant_user_permission_unknown_user_raises(client, monkeypatch):
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        with pytest.raises(MlflowException, match="not found") as exc:
            client.grant_user_permission(
                "no-such-user-" + random_str(), "experiment", "exp-1", "READ"
            )
    assert exc.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_grant_user_permission_invalid_resource_type(client, monkeypatch):
    username, _ = _new_user(client, monkeypatch)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        with pytest.raises(MlflowException, match="Invalid resource type"):
            client.grant_user_permission(username, "bogus", "x", "READ")


@pytest.mark.parametrize(
    ("api_method", "args"),
    [
        ("grant_user_permission", ("workspace", "*", "USE")),
        ("revoke_user_permission", ("workspace", "*")),
    ],
    ids=["grant", "revoke"],
)
def test_admin_cannot_target_workspace_resource_type(client, monkeypatch, api_method, args):
    # Super admins skip ``validate_can_manage_resource`` via ``sender_is_admin``.
    # Handler/store-level rejection is the only defense — must fire for the admin path.
    username, _ = _new_user(client, monkeypatch)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        with pytest.raises(MlflowException, match="not supported by the per-user"):
            getattr(client, api_method)(username, *args)


def test_grant_user_permission_invalid_permission_for_resource_type(client, monkeypatch):
    # ``NO_PERMISSIONS`` is intentionally disallowed at the resource scope —
    # absence of a grant + ``default_permission`` already expresses "no access".
    username, _ = _new_user(client, monkeypatch)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        with pytest.raises(MlflowException, match="Invalid permission"):
            client.grant_user_permission(username, "experiment", "exp-1", "NO_PERMISSIONS")


def test_revoke_user_permission_roundtrip(client, monkeypatch):
    username, _ = _new_user(client, monkeypatch)
    exp_id = _create_experiment(client.tracking_uri, monkeypatch, f"revoke-{random_str()}")
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.grant_user_permission(username, "experiment", exp_id, "READ")
        roles_before = client.list_user_roles(username)
        synthetic_before = next(r for r in roles_before if r.name.startswith("__user_"))
        assert any(p.resource_pattern == exp_id for p in synthetic_before.permissions)

        client.revoke_user_permission(username, "experiment", exp_id)
        roles_after = client.list_user_roles(username)
        synthetic_after = next((r for r in roles_after if r.name.startswith("__user_")), None)
        remaining = synthetic_after.permissions if synthetic_after else []
        assert not any(p.resource_pattern == exp_id for p in remaining)


def test_revoke_user_permission_missing_row_raises(client, monkeypatch):
    username, _ = _new_user(client, monkeypatch)
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        with pytest.raises(MlflowException, match="not found") as exc:
            client.revoke_user_permission(username, "experiment", "exp-not-granted")
    assert exc.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_check_user_permission_self_check(client, monkeypatch):
    # A non-admin user can check their own permissions on any resource.
    username, password = _new_user(client, monkeypatch)
    exp_id = _create_experiment(client.tracking_uri, monkeypatch, f"self-{random_str()}")
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.grant_user_permission(username, "experiment", exp_id, "READ")

    with User(username, password, monkeypatch):
        result = client.check_user_permission(username, "experiment", exp_id)

    # READ.can_use is False — ``allowed`` mirrors ``can_use``.
    assert result.permission == "READ"
    assert result.allowed is False


def test_check_user_permission_returns_max_grant(client, monkeypatch):
    # An explicit MANAGE grant yields allowed=True (MANAGE.can_use=True).
    username, password = _new_user(client, monkeypatch)
    exp_id = _create_experiment(client.tracking_uri, monkeypatch, f"mgr-{random_str()}")
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.grant_user_permission(username, "experiment", exp_id, "MANAGE")

    with User(username, password, monkeypatch):
        result = client.check_user_permission(username, "experiment", exp_id)

    assert result.permission == "MANAGE"
    assert result.allowed is True


def test_check_user_permission_cross_user_requires_admin(client, monkeypatch):
    # A plain user cannot check another user's permissions.
    target, _ = _new_user(client, monkeypatch)
    requester, requester_pw = _new_user(client, monkeypatch)

    with User(requester, requester_pw, monkeypatch), assert_unauthorized():
        client.check_user_permission(target, "experiment", "exp-x")


def test_check_user_permission_admin_can_check_any_user(client, monkeypatch):
    username, _ = _new_user(client, monkeypatch)
    exp_id = _create_experiment(client.tracking_uri, monkeypatch, f"admin-{random_str()}")
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        client.grant_user_permission(username, "experiment", exp_id, "EDIT")
        result = client.check_user_permission(username, "experiment", exp_id)

    assert result.permission == "EDIT"
    assert result.allowed is True


def test_grant_user_permission_requires_authentication(client):
    with assert_unauthenticated():
        client.grant_user_permission("alice", "experiment", "exp-1", "READ")


def test_grant_user_permission_non_admin_without_manage_rejected(client, monkeypatch):
    # A plain user without per-resource MANAGE cannot grant permissions.
    requester, requester_pw = _new_user(client, monkeypatch)
    target, _ = _new_user(client, monkeypatch)

    with User(requester, requester_pw, monkeypatch), assert_unauthorized():
        client.grant_user_permission(target, "experiment", "exp-1", "READ")


@pytest.mark.parametrize("api_prefix", ["api", "ajax-api"])
@pytest.mark.parametrize(
    "endpoint",
    [
        "/3.0/mlflow/users/permissions/grant",
        "/3.0/mlflow/users/permissions/revoke",
        "/3.0/mlflow/auth/check",
    ],
)
def test_unified_permission_endpoints_reachable_at_both_path_prefixes(client, api_prefix, endpoint):
    # The MLflow frontend hits /ajax-api/ paths; the Python client hits /api/ paths.
    # Every unified permission route must be reachable at both — a 404 here would
    # silently break the admin UI without surfacing as a permission-system failure.
    resp = requests.post(
        f"{client.tracking_uri}/{api_prefix}{endpoint}",
        json={},
        auth=(ADMIN_USERNAME, ADMIN_PASSWORD),
    )
    assert resp.status_code != 404, f"POST /{api_prefix}{endpoint} unexpectedly returned 404"
