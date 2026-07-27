import asyncio
import base64
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import jwt
import pytest
import requests
from cachetools import TTLCache
from cryptography.fernet import Fernet

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Dataset, DatasetInput, InputTag, LoggedModelOutput
from mlflow.entities.logged_model_status import LoggedModelStatus
from mlflow.environment_variables import (
    _MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN,
    MLFLOW_ENABLE_WORKSPACES,
    MLFLOW_FLASK_SERVER_SECRET_KEY,
    MLFLOW_TRACKING_PASSWORD,
    MLFLOW_TRACKING_USERNAME,
    MLFLOW_WORKSPACE_STORE_URI,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    RESOURCE_DOES_NOT_EXIST,
    UNAUTHENTICATED,
    ErrorCode,
)
from mlflow.server import auth as auth_module
from mlflow.server.asgi_utils import get_routed_asgi_path
from mlflow.server.auth import (
    _authenticate_fastapi_request,
    _find_fastapi_response_filter,
    _find_fastapi_validator,
    _re_compile_path,
)
from mlflow.server.auth.permissions import NO_PERMISSIONS, READ, USE
from mlflow.server.auth.routes import (
    AJAX_LIST_USERS,
    LIST_USERS,
)
from mlflow.server.auth.sqlalchemy_store import SqlAlchemyStore
from mlflow.server.handlers import STATIC_PREFIX_ENV_VAR, _get_ajax_path
from mlflow.utils import workspace_context
from mlflow.utils.os import is_windows
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

from tests.helper_functions import kill_process_tree, random_str
from tests.server.auth.auth_test_utils import (
    ADMIN_PASSWORD,
    ADMIN_USERNAME,
    User,
    create_user,
    grant_role_permission,
    write_isolated_auth_config,
)
from tests.tracking.integration_test_utils import (
    _init_server,
    _send_rest_tracking_post_request,
    get_safe_port,
)

_PACKAGED_BASIC_AUTH_INI = Path(auth_module.__file__).parent / "basic_auth.ini"
_TEST_DIR = Path(__file__).parent


def _isolate_auth_config(extra_env: dict[str, str], tmp_path: Path) -> dict[str, str]:
    """Redirect the auth store's SQLite DB to a tmp_path-scoped file.

    Both the packaged default config (``mlflow/server/auth/basic_auth.ini``)
    and the static fixture .ini files under ``fixtures/``
    set ``database_uri = sqlite:///basic_auth.db`` — a *relative* path that
    the spawned server resolves against its CWD (typically the repo root).
    Without redirection, every test that boots the auth server shares one
    ``basic_auth.db`` next to the dev server, leaking users / roles across
    runs (bug-bash report: "so many users with hash usernames I didn't
    create manually"). Rewrite the source .ini into ``tmp_path`` with the
    DB URI swapped for an absolute path, and inject the rewritten copy via
    ``MLFLOW_AUTH_CONFIG_PATH``.

    Relative ``MLFLOW_AUTH_CONFIG_PATH`` values are anchored to this test
    file's directory so the helper works regardless of pytest's CWD.
    """
    if raw := extra_env.get("MLFLOW_AUTH_CONFIG_PATH"):
        src_path = Path(raw)
        if not src_path.is_absolute():
            src_path = _TEST_DIR / src_path
    else:
        src_path = _PACKAGED_BASIC_AUTH_INI
    isolated_db = tmp_path / "basic_auth.db"
    isolated_text = re.sub(
        r"^database_uri\s*=.*$",
        f"database_uri = sqlite:///{isolated_db}",
        src_path.read_text(),
        flags=re.MULTILINE,
    )
    dst_path = tmp_path / src_path.name
    dst_path.write_text(isolated_text)
    return {**extra_env, "MLFLOW_AUTH_CONFIG_PATH": str(dst_path)}


@pytest.fixture
def client(request, tmp_path):
    path = tmp_path.joinpath("sqlalchemy.db").as_uri()
    backend_uri = ("sqlite://" if is_windows() else "sqlite:////") + path[len("file://") :]
    extra_env = _isolate_auth_config(getattr(request, "param", {}), tmp_path)
    extra_env[MLFLOW_FLASK_SERVER_SECRET_KEY.name] = "my-secret-key"

    with _init_server(
        backend_uri=backend_uri,
        root_artifact_uri=tmp_path.joinpath("artifacts").as_uri(),
        extra_env=extra_env,
        app="mlflow.server.auth:create_app",
        server_type="flask",
    ) as url:
        yield MlflowClient(url)


@pytest.fixture
def fastapi_client(request, tmp_path):
    """FastAPI client fixture for testing FastAPI-specific middleware (e.g., gateway routes)."""
    path = tmp_path.joinpath("sqlalchemy.db").as_uri()
    backend_uri = ("sqlite://" if is_windows() else "sqlite:////") + path[len("file://") :]
    extra_env = _isolate_auth_config(getattr(request, "param", {}), tmp_path)
    extra_env[MLFLOW_FLASK_SERVER_SECRET_KEY.name] = "my-secret-key"
    # Set _MLFLOW_SGI_NAME to "uvicorn" so auth module returns FastAPI app
    extra_env["_MLFLOW_SGI_NAME"] = "uvicorn"

    with _init_server(
        backend_uri=backend_uri,
        root_artifact_uri=tmp_path.joinpath("artifacts").as_uri(),
        extra_env=extra_env,
        app="mlflow.server.auth:create_app",
        server_type="fastapi",
    ) as url:
        yield MlflowClient(url)


@pytest.fixture
def fastapi_workspace_client(tmp_path):
    """FastAPI client fixture with workspaces enabled, for workspace-scoped gateway auth."""
    auth_config_path = write_isolated_auth_config(tmp_path)
    path = tmp_path.joinpath("sqlalchemy.db").as_uri()
    backend_uri = ("sqlite://" if is_windows() else "sqlite:////") + path[len("file://") :]

    with _init_server(
        backend_uri=backend_uri,
        root_artifact_uri=tmp_path.joinpath("artifacts").as_uri(),
        extra_env={
            MLFLOW_FLASK_SERVER_SECRET_KEY.name: "my-secret-key",
            "MLFLOW_AUTH_CONFIG_PATH": str(auth_config_path),
            MLFLOW_ENABLE_WORKSPACES.name: "true",
            MLFLOW_WORKSPACE_STORE_URI.name: backend_uri,
            "_MLFLOW_SGI_NAME": "uvicorn",
        },
        app="mlflow.server.auth:create_app",
        server_type="fastapi",
    ) as url:
        yield MlflowClient(url)


def test_experiment_permission_honored_when_tracking_store_lacks_experiment(tmp_path, monkeypatch):
    # Regression test for https://github.com/mlflow/mlflow/issues/24566:
    # On an --artifacts-only server the tracking store has no experiment data, so the
    # resource->workspace lookup fails. With workspaces disabled, permission resolution must
    # still honor an explicit experiment grant in the auth DB instead of falling through to
    # default_permission (NO_PERMISSIONS => 403).
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")
    monkeypatch.setattr(
        auth_module,
        "auth_config",
        auth_module.auth_config._replace(default_permission=NO_PERMISSIONS.name),
    )

    auth_store = SqlAlchemyStore()
    auth_store.init_db(f"sqlite:///{tmp_path / 'auth-store.db'}")
    monkeypatch.setattr(auth_module, "store", auth_store, raising=False)

    username = "restricted"
    experiment_id = "123"
    auth_store.create_user(username, "supersecurepassword", is_admin=False)
    auth_store.create_experiment_permission(experiment_id, username, READ.name)

    def _raise_not_found(_experiment_id):
        raise MlflowException("no experiment data", RESOURCE_DOES_NOT_EXIST)

    monkeypatch.setattr(
        auth_module,
        "_get_tracking_store",
        lambda: SimpleNamespace(get_experiment=_raise_not_found),
    )

    try:
        # default_permission is NO_PERMISSIONS, so a READ result proves the grant (not the
        # default) is what's honored.
        perm = auth_module._get_experiment_permission(experiment_id, username)
        assert perm.name == READ.name
        assert perm.can_read

        # A user without a grant falls through to default_permission. Use a default distinct
        # from NO_PERMISSIONS so this asserts the no-grant fall-through path (resolver returns
        # None) rather than the NO_PERMISSIONS workspace-deny sentinel — the two are otherwise
        # indistinguishable when default_permission == NO_PERMISSIONS.
        monkeypatch.setattr(
            auth_module,
            "auth_config",
            auth_module.auth_config._replace(default_permission=READ.name),
        )
        auth_store.create_user("stranger", "supersecurepassword", is_admin=False)
        stranger_perm = auth_module._get_experiment_permission(experiment_id, "stranger")
        assert stranger_perm.name == READ.name
    finally:
        auth_store.engine.dispose()


def test_known_workspace_resolver_honors_grant_when_workspace_unresolved(tmp_path, monkeypatch):
    # Sibling of test_experiment_permission_honored_when_tracking_store_lacks_experiment for the
    # _role_permission_for_known_workspace path (registered models / prompts): when the workspace
    # can't be resolved (e.g. the registry lookup returned no workspace) and workspaces are
    # disabled, resolution must still honor an explicit grant instead of falling through to
    # default_permission.
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")
    monkeypatch.setattr(
        auth_module,
        "auth_config",
        auth_module.auth_config._replace(default_permission=NO_PERMISSIONS.name),
    )

    auth_store = SqlAlchemyStore()
    auth_store.init_db(f"sqlite:///{tmp_path / 'auth-store.db'}")
    monkeypatch.setattr(auth_module, "store", auth_store, raising=False)

    username = "restricted"
    model_name = "m1"
    auth_store.create_user(username, "supersecurepassword", is_admin=False)
    auth_store.create_registered_model_permission(model_name, username, READ.name)

    try:
        # workspace_name=None mimics an unresolved workspace (e.g. RESOURCE_DOES_NOT_EXIST).
        # default_permission is NO_PERMISSIONS, so a READ result proves the grant is honored.
        resolver = auth_module._role_permission_for_known_workspace(
            username, "registered_model", model_name, None
        )
        perm = auth_module._get_role_permission_or_default(resolver)
        assert perm.name == READ.name
        assert perm.can_read

        # A user without a grant falls through to default_permission. Use a default distinct
        # from NO_PERMISSIONS so this asserts the no-grant fall-through path (resolver returns
        # None) rather than the NO_PERMISSIONS workspace-deny sentinel.
        monkeypatch.setattr(
            auth_module,
            "auth_config",
            auth_module.auth_config._replace(default_permission=READ.name),
        )
        auth_store.create_user("stranger", "supersecurepassword", is_admin=False)
        stranger_resolver = auth_module._role_permission_for_known_workspace(
            "stranger", "registered_model", model_name, None
        )
        stranger_perm = auth_module._get_role_permission_or_default(stranger_resolver)
        assert stranger_perm.name == READ.name
    finally:
        auth_store.engine.dispose()


def test_authenticate(client, monkeypatch):
    # unauthenticated
    monkeypatch.delenv(MLFLOW_TRACKING_USERNAME.name, raising=False)
    monkeypatch.delenv(MLFLOW_TRACKING_PASSWORD.name, raising=False)
    with pytest.raises(MlflowException, match=r"You are not authenticated.") as exception_context:
        client.search_experiments()
    assert exception_context.value.error_code == ErrorCode.Name(UNAUTHENTICATED)

    # authenticated
    username, password = create_user(client.tracking_uri)
    with User(username, password, monkeypatch):
        client.search_experiments()


@pytest.mark.parametrize(
    ("username", "password"),
    [
        ("", "password"),
        ("username", ""),
        ("", ""),
    ],
)
def test_validate_username_and_password(client, username, password):
    with pytest.raises(requests.exceptions.HTTPError, match=r"BAD REQUEST"):
        create_user(client.tracking_uri, username=username, password=password)


def test_proxy_artifact_path_detection():
    assert auth_module._is_proxy_artifact_path("/api/2.0/mlflow-artifacts/artifacts/foo")
    assert auth_module._is_proxy_artifact_path("/ajax-api/2.0/mlflow-artifacts/artifacts/foo")


def test_proxy_artifact_path_detection_with_static_prefix(monkeypatch):
    monkeypatch.setenv(STATIC_PREFIX_ENV_VAR, "/mlflow")

    assert auth_module._is_proxy_artifact_path("/mlflow/api/2.0/mlflow-artifacts/artifacts/foo")
    assert auth_module._is_proxy_artifact_path(
        "/mlflow/ajax-api/2.0/mlflow-artifacts/presigned/1/run-id/artifacts/model.pkl"
    )
    assert not auth_module._is_proxy_artifact_path("/api/2.0/mlflow/experiments/get")


def test_is_unprotected_route_handles_static_prefix(monkeypatch):
    # When ``_MLFLOW_STATIC_PREFIX`` is set, the health/static/favicon routes
    # are served from e.g. ``/mlflow/health``. Health checks must not require
    # auth on prefixed deployments.
    monkeypatch.delenv(STATIC_PREFIX_ENV_VAR, raising=False)
    assert auth_module.is_unprotected_route("/health")
    assert auth_module.is_unprotected_route("/favicon.ico")
    assert auth_module.is_unprotected_route("/static/foo.js")
    assert not auth_module.is_unprotected_route("/api/2.0/mlflow/users/list")

    monkeypatch.setenv(STATIC_PREFIX_ENV_VAR, "/mlflow")
    assert auth_module.is_unprotected_route("/mlflow/health")
    assert auth_module.is_unprotected_route("/mlflow/favicon.ico")
    assert auth_module.is_unprotected_route("/mlflow/static/foo.js")
    # Unprefixed forms still pass through (local dev / non-prefixed deployments).
    assert auth_module.is_unprotected_route("/health")
    # Protected routes stay protected even with the prefix.
    assert not auth_module.is_unprotected_route("/mlflow/api/2.0/mlflow/users/list")


def test_proxy_artifact_mpu_path_detection():
    # MPU create/complete/abort paths should be recognized as proxy artifact paths
    for action in ("create", "complete", "abort"):
        assert auth_module._is_proxy_artifact_path(
            f"/api/2.0/mlflow-artifacts/mpu/{action}/1/run-id/artifacts/model"
        )
        assert auth_module._is_proxy_artifact_path(
            f"/ajax-api/2.0/mlflow-artifacts/mpu/{action}/1/run-id/artifacts/model"
        )

    # Non-artifact paths should not match
    assert not auth_module._is_proxy_artifact_path("/api/2.0/mlflow/experiments/get")


def test_proxy_artifact_mpu_validator_returns_update_for_post():
    validator = auth_module._get_proxy_artifact_validator(
        "POST", {"artifact_path": "1/run-id/artifacts/model"}
    )
    assert validator is auth_module.validate_can_update_experiment_artifact_proxy


def test_proxy_artifact_presigned_path_detection():
    # GetPresignedDownloadUrl paths must be recognized so basic-auth applies the same
    # experiment artifact READ check it applies to /mlflow-artifacts/artifacts downloads.
    assert auth_module._is_proxy_artifact_path(
        "/api/2.0/mlflow-artifacts/presigned/1/run-id/artifacts/model.pkl"
    )
    assert auth_module._is_proxy_artifact_path(
        "/ajax-api/2.0/mlflow-artifacts/presigned/1/run-id/artifacts/model.pkl"
    )


def test_proxy_artifact_presigned_validator_returns_read_for_get():
    validator = auth_module._get_proxy_artifact_validator(
        "GET", {"artifact_path": "1/run-id/artifacts/model.pkl"}
    )
    assert validator is auth_module.validate_can_read_experiment_artifact_proxy


@pytest.mark.parametrize(
    ("path", "method"),
    [
        ("/ajax-api/3.0/mlflow/issues/invoke", "POST"),
        ("/ajax-api/3.0/mlflow/genai/evaluate/invoke", "POST"),
        ("/ajax-api/3.0/mlflow/demo/generate", "POST"),
        ("/ajax-api/3.0/mlflow/demo/delete", "POST"),
        ("/ajax-api/3.0/mlflow/jobs/<job_id>", "GET"),
        ("/ajax-api/3.0/mlflow/jobs/cancel/<job_id>", "PATCH"),
        ("/graphql", "GET"),
        ("/api/3.0/mlflow/server-info", "GET"),
    ],
)
def test_before_request_validators_excludes_view_function_endpoints(path, method):
    # ``get_endpoints`` hardcodes the view function for explicitly defined endpoints,
    # so without filtering these leak into BEFORE_REQUEST_VALIDATORS and get called as
    # validators — re-running the endpoint's side effects. Guard against that.
    assert (path, method) not in auth_module.BEFORE_REQUEST_VALIDATORS


def test_before_request_validators_only_contains_real_validators():
    proto_validators = set(auth_module.BEFORE_REQUEST_HANDLERS.values())
    leaked = {
        (path, method): handler
        for (path, method), handler in auth_module.BEFORE_REQUEST_VALIDATORS.items()
        if getattr(handler, "__module__", "") == "mlflow.server.handlers"
        and handler not in proto_validators
    }
    assert leaked == {}


def test_proxy_artifact_authorization_required(client, monkeypatch):
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)

    with User(username1, password1, monkeypatch):
        experiment_id = client.create_experiment("proxy-artifact-authz-test")

    response = requests.put(
        url=(
            client.tracking_uri
            + f"/ajax-api/2.0/mlflow-artifacts/artifacts/{experiment_id}/test.txt"
        ),
        data=b"forbidden",
        auth=(username2, password2),
    )
    assert response.status_code == 403


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_proxy_artifact_presigned_authorization_required(client, monkeypatch):
    # Regression test for https://github.com/mlflow/mlflow/issues/24567:
    # GetPresignedDownloadUrl must enforce the same experiment artifact READ permission
    # as the proxied download route. Without authorization, a user with no grant would
    # reach the handler (returning a working presigned URL on cloud backends), leaking
    # artifacts. A denied user must receive 403 before the handler runs.
    # Runs against ``default_permission=NO_PERMISSIONS`` so a GET (READ) without an
    # explicit grant is denied — a READ-permission default would otherwise allow it.
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)

    with User(username1, password1, monkeypatch):
        experiment_id = client.create_experiment("proxy-artifact-presigned-authz-test")

    presigned_url = (
        client.tracking_uri + f"/api/2.0/mlflow-artifacts/presigned/{experiment_id}/test.txt"
    )
    response = requests.get(url=presigned_url, auth=(username2, password2))
    assert response.status_code == 403


@pytest.mark.parametrize(
    "client",
    [
        {
            "MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini",
            STATIC_PREFIX_ENV_VAR: "/mlflow",
        }
    ],
    indirect=True,
)
def test_presigned_download_authorization_required_with_static_prefix(client, monkeypatch):
    prefixed_tracking_uri = f"{client.tracking_uri}/mlflow"
    prefixed_client = MlflowClient(prefixed_tracking_uri)
    username1, password1 = create_user(prefixed_tracking_uri)
    username2, password2 = create_user(prefixed_tracking_uri)

    with User(username1, password1, monkeypatch):
        experiment_id = prefixed_client.create_experiment("prefixed-presigned-download-authz-test")

    response = requests.get(
        url=(
            client.tracking_uri
            + (
                f"/mlflow/api/2.0/mlflow-artifacts/presigned/"
                f"{experiment_id}/run-id/artifacts/model.pkl"
            )
        ),
        auth=(username2, password2),
    )
    assert response.status_code == 403


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_proxy_artifact_list_query_param_uses_experiment_permission(client, monkeypatch):
    # Regression test for https://github.com/mlflow/mlflow/issues/21201:
    # When default_permission is NO_PERMISSIONS, a user with explicit experiment permission
    # should be able to list artifacts via query parameter path (GET ?path=<experiment_id>/...).
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)

    with User(username1, password1, monkeypatch):
        experiment_id = client.create_experiment("proxy-artifact-list-query-param-test")

    # user1 has MANAGE on experiment — list via query param path should be allowed (HTTP 200)
    response = requests.get(
        url=client.tracking_uri + "/api/2.0/mlflow-artifacts/artifacts",
        params={"path": f"{experiment_id}/models/m-abc123/artifacts"},
        auth=(username1, password1),
    )
    assert response.status_code != 403

    # user2 has no permission on the experiment — expect 403
    response = requests.get(
        url=client.tracking_uri + "/api/2.0/mlflow-artifacts/artifacts",
        params={"path": f"{experiment_id}/models/m-abc123/artifacts"},
        auth=(username2, password2),
    )
    assert response.status_code == 403


@pytest.mark.parametrize("mpu_action", ["create", "complete", "abort"])
def test_mpu_authorization_required(client, monkeypatch, mpu_action):
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)

    with User(username1, password1, monkeypatch):
        experiment_id = client.create_experiment(f"mpu-authz-test-{mpu_action}")

    # user2 has no permission on user1's experiment — expect 403
    response = requests.post(
        url=(
            client.tracking_uri
            + f"/api/2.0/mlflow-artifacts/mpu/{mpu_action}/{experiment_id}/artifacts/model"
        ),
        json={"path": "python_model.pkl", "num_parts": 1},
        auth=(username2, password2),
    )
    assert response.status_code == 403


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_presigned_download_url_authorization_required(client, monkeypatch):
    # Minting a presigned download URL grants direct read access to a run's artifacts,
    # so it must enforce the same per-run READ permission as the proxied download paths.
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)

    with User(username1, password1, monkeypatch):
        experiment_id = client.create_experiment("presigned-download-authz-test")
        run = client.create_run(experiment_id)
        run_id = run.info.run_id

    # user2 has no permission on user1's experiment — the auth layer must reject with
    # 403 before the handler runs (without the validator this reaches the handler and
    # returns a handler-level status such as 501 for the local artifact repository).
    response = requests.post(
        url=client.tracking_uri + "/api/2.0/mlflow/artifacts/presigned-download-url",
        json={"run_id": run_id, "path": "model.pkl"},
        auth=(username2, password2),
    )
    assert response.status_code == 403

    # user1 (creator, MANAGE on the experiment) passes the auth layer; the request
    # reaches the handler, which rejects the local (file://) artifact repo with 501.
    response = requests.post(
        url=client.tracking_uri + "/api/2.0/mlflow/artifacts/presigned-download-url",
        json={"run_id": run_id, "path": "model.pkl"},
        auth=(username1, password1),
    )
    assert response.status_code == 501


def _mlflow_search_experiments_rest(base_uri, headers):
    response = requests.post(
        f"{base_uri}/api/2.0/mlflow/experiments/search",
        headers=headers,
        json={
            "max_results": 100,
        },
    )
    response.raise_for_status()
    return response


def _mlflow_create_user_rest(base_uri, headers):
    username = random_str()
    password = random_str()
    response = requests.post(
        f"{base_uri}/api/2.0/mlflow/users/create",
        headers=headers,
        json={
            "username": username,
            "password": password,
        },
    )
    response.raise_for_status()
    return username, password


@pytest.mark.parametrize(
    "client",
    [
        {
            "MLFLOW_AUTH_CONFIG_PATH": "fixtures/jwt_auth.ini",
            "PYTHONPATH": str(Path.cwd() / "examples" / "jwt_auth"),
        }
    ],
    indirect=True,
)
def test_authenticate_jwt(client):
    # unauthenticated
    with pytest.raises(requests.HTTPError, match=r"401 Client Error: UNAUTHORIZED") as e:
        _mlflow_search_experiments_rest(client.tracking_uri, {})
    assert e.value.response.status_code == 401  # Unauthorized

    # authenticated
    # we need to use jwt to authenticate as admin so that we can create a new user
    bearer_token = jwt.encode({"username": ADMIN_USERNAME}, "secret", algorithm="HS256")
    headers = {"Authorization": f"Bearer {bearer_token}"}
    username, password = _mlflow_create_user_rest(client.tracking_uri, headers)

    # authenticate with the newly created user
    headers = {
        "Authorization": f"Bearer {jwt.encode({'username': username}, 'secret', algorithm='HS256')}"
    }
    _mlflow_search_experiments_rest(client.tracking_uri, headers)

    # invalid token
    bearer_token = jwt.encode({"username": username}, "invalid", algorithm="HS256")
    headers = {"Authorization": f"Bearer {bearer_token}"}
    with pytest.raises(requests.HTTPError, match=r"401 Client Error: UNAUTHORIZED") as e:
        _mlflow_search_experiments_rest(client.tracking_uri, headers)
    assert e.value.response.status_code == 401  # Unauthorized


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_search_experiments(client, monkeypatch):
    """
    Use user1 to create 10 experiments, grant READ permission to user2 on
    experiments [0, 3, 4, 5, 6, 8]. Test whether user2 can search only the
    readable experiments, both paged and un-paged.

    Runs against ``default_permission=NO_PERMISSIONS`` so experiments without
    an explicit READ grant are hidden from user2; the simplified model no
    longer accepts ``NO_PERMISSIONS`` as a per-resource grant.
    """
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)

    readable = [0, 3, 4, 5, 6, 8]

    with User(username1, password1, monkeypatch):
        for i in range(10):
            experiment_id = client.create_experiment(f"exp{i}")
            if i in readable:
                grant_role_permission(
                    client.tracking_uri,
                    username2,
                    "experiment",
                    experiment_id,
                    "READ",
                )

    # test un-paged search
    with User(username1, password1, monkeypatch):
        experiments = client.search_experiments(
            max_results=100,
            filter_string="name LIKE 'exp%'",
            order_by=["name ASC"],
        )
        names = sorted([exp.name for exp in experiments])
        assert names == [f"exp{i}" for i in range(10)]

    with User(username2, password2, monkeypatch):
        experiments = client.search_experiments(
            max_results=100,
            filter_string="name LIKE 'exp%'",
            order_by=["name ASC"],
        )
        names = sorted([exp.name for exp in experiments])
        assert names == [f"exp{i}" for i in readable]

    # test paged search
    with User(username1, password1, monkeypatch):
        page_token = ""
        experiments = []
        while True:
            res = client.search_experiments(
                max_results=4,
                filter_string="name LIKE 'exp%'",
                order_by=["name ASC"],
                page_token=page_token,
            )
            experiments.extend(res)
            page_token = res.token
            if not page_token:
                break

        names = sorted([exp.name for exp in experiments])
        assert names == [f"exp{i}" for i in range(10)]

    with User(username2, password2, monkeypatch):
        page_token = ""
        experiments = []
        while True:
            res = client.search_experiments(
                max_results=4,
                filter_string="name LIKE 'exp%'",
                order_by=["name ASC"],
                page_token=page_token,
            )
            experiments.extend(res)
            page_token = res.token
            if not page_token:
                break

        names = sorted([exp.name for exp in experiments])
        assert names == [f"exp{i}" for i in readable]


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_search_registered_models(client, monkeypatch):
    """
    Use user1 to create 10 registered_models, grant READ permission to user2
    on registered_models [0, 3, 4, 5, 6, 8]. Test whether user2 can search
    only the readable models, both paged and un-paged.

    Runs against ``default_permission=NO_PERMISSIONS`` so models without an
    explicit READ grant are hidden from user2.
    """
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)

    readable = [0, 3, 4, 5, 6, 8]

    with User(username1, password1, monkeypatch):
        for i in range(10):
            rm = client.create_registered_model(f"rm{i}")
            if i in readable:
                grant_role_permission(
                    client.tracking_uri,
                    username2,
                    "registered_model",
                    rm.name,
                    "READ",
                )

    # test un-paged search
    with User(username1, password1, monkeypatch):
        registered_models = client.search_registered_models(
            max_results=100,
            filter_string="name LIKE 'rm%'",
            order_by=["name ASC"],
        )
        names = sorted([rm.name for rm in registered_models])
        assert names == [f"rm{i}" for i in range(10)]

    with User(username2, password2, monkeypatch):
        registered_models = client.search_registered_models(
            max_results=100,
            filter_string="name LIKE 'rm%'",
            order_by=["name ASC"],
        )
        names = sorted([rm.name for rm in registered_models])
        assert names == [f"rm{i}" for i in readable]

    # test paged search
    with User(username1, password1, monkeypatch):
        page_token = ""
        registered_models = []
        while True:
            res = client.search_registered_models(
                max_results=4,
                filter_string="name LIKE 'rm%'",
                order_by=["name ASC"],
                page_token=page_token,
            )
            registered_models.extend(res)
            page_token = res.token
            if not page_token:
                break

        names = sorted([rm.name for rm in registered_models])
        assert names == [f"rm{i}" for i in range(10)]

    with User(username2, password2, monkeypatch):
        page_token = ""
        registered_models = []
        while True:
            res = client.search_registered_models(
                max_results=4,
                filter_string="name LIKE 'rm%'",
                order_by=["name ASC"],
                page_token=page_token,
            )
            registered_models.extend(res)
            page_token = res.token
            if not page_token:
                break

        names = sorted([rm.name for rm in registered_models])
        assert names == [f"rm{i}" for i in readable]


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_search_model_versions(client, monkeypatch):
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)

    readable = [0, 2, 4]

    with User(username1, password1, monkeypatch):
        experiment_id = client.create_experiment("mv_test_exp")
        run = client.create_run(experiment_id)
        run_id = run.info.run_id
        for i in range(5):
            rm = client.create_registered_model(f"mv_model{i}")
            client.create_model_version(rm.name, f"runs:/{run_id}/model", run_id=run_id)
            if i in readable:
                grant_role_permission(
                    client.tracking_uri,
                    username2,
                    "registered_model",
                    rm.name,
                    "READ",
                )

    # user1 (owner) sees all model versions
    with User(username1, password1, monkeypatch):
        versions = client.search_model_versions(filter_string="name LIKE 'mv_model%'")
        names = sorted({mv.name for mv in versions})
        assert names == [f"mv_model{i}" for i in range(5)]

    # user2 only sees model versions for readable models
    with User(username2, password2, monkeypatch):
        versions = client.search_model_versions(filter_string="name LIKE 'mv_model%'")
        names = sorted({mv.name for mv in versions})
        assert names == [f"mv_model{i}" for i in readable]


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_graphql_search_model_versions(client, monkeypatch):
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)

    readable = [0, 2, 4]

    with User(username1, password1, monkeypatch):
        experiment_id = client.create_experiment("gql_mv_test_exp")
        run = client.create_run(experiment_id)
        run_id = run.info.run_id
        for i in range(5):
            rm = client.create_registered_model(f"gql_mv_model{i}")
            client.create_model_version(rm.name, f"runs:/{run_id}/model", run_id=run_id)
            if i in readable:
                grant_role_permission(
                    client.tracking_uri,
                    username2,
                    "registered_model",
                    rm.name,
                    "READ",
                )

    query = """
    query SearchModelVersions($input: MlflowSearchModelVersionsInput){
      mlflowSearchModelVersions(input: $input){
        modelVersions { name version }
      }
    }
    """
    variables = {"input": {"filter": "name LIKE 'gql_mv_model%'"}}

    # user1 (owner) sees all via GraphQL
    resp = requests.post(
        f"{client.tracking_uri}/graphql",
        json={"query": query, "variables": variables},
        auth=(username1, password1),
    )
    resp.raise_for_status()
    payload = resp.json()
    assert payload.get("errors") in (None, [])
    names = sorted({
        mv["name"] for mv in payload["data"]["mlflowSearchModelVersions"]["modelVersions"]
    })
    assert names == [f"gql_mv_model{i}" for i in range(5)]

    # user2 only sees versions for readable models via GraphQL
    resp = requests.post(
        f"{client.tracking_uri}/graphql",
        json={"query": query, "variables": variables},
        auth=(username2, password2),
    )
    resp.raise_for_status()
    payload = resp.json()
    assert payload.get("errors") in (None, [])
    names = sorted({
        mv["name"] for mv in payload["data"]["mlflowSearchModelVersions"]["modelVersions"]
    })
    assert names == [f"gql_mv_model{i}" for i in readable]


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_create_model_version_requires_read_on_source_run(
    client: MlflowClient, monkeypatch: pytest.MonkeyPatch
):
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)

    with User(username1, password1, monkeypatch):
        exp_id = client.create_experiment("source-run-authz-exp")
        run = client.create_run(exp_id)
        run_id = run.info.run_id
        source = run.info.artifact_uri

    with User(username2, password2, monkeypatch):
        rm = client.create_registered_model("source-run-authz-model")

    # user2 owns the target model but has no read on user1's experiment/run:
    # anchoring a model version at user1's run artifact dir must be denied.
    response = _send_rest_tracking_post_request(
        client.tracking_uri,
        "/api/2.0/mlflow/model-versions/create",
        json_payload={"name": rm.name, "source": source, "run_id": run_id},
        auth=(username2, password2),
    )
    assert response.status_code == 403
    assert "Permission denied" in response.text

    # grant user2 READ on user1's experiment; creation should now succeed.
    grant_role_permission(
        client.tracking_uri,
        username2,
        "experiment",
        exp_id,
        "READ",
    )

    with User(username2, password2, monkeypatch):
        response = _send_rest_tracking_post_request(
            client.tracking_uri,
            "/api/2.0/mlflow/model-versions/create",
            json_payload={"name": rm.name, "source": source, "run_id": run_id},
            auth=(username2, password2),
        )
        assert response.status_code == 200


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_create_model_version_requires_read_on_source_model(
    client: MlflowClient, monkeypatch: pytest.MonkeyPatch
):
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)

    with User(username1, password1, monkeypatch):
        exp_id = client.create_experiment("source-model-authz-exp")
        model = client.create_logged_model(experiment_id=exp_id)
        model_id = model.model_id
        source = model.artifact_location

    with User(username2, password2, monkeypatch):
        rm = client.create_registered_model("source-model-authz-model")

    # user2 owns the target model but has no read on the source logged model.
    response = _send_rest_tracking_post_request(
        client.tracking_uri,
        "/api/2.0/mlflow/model-versions/create",
        json_payload={"name": rm.name, "source": source, "model_id": model_id},
        auth=(username2, password2),
    )
    assert response.status_code == 403
    assert "Permission denied" in response.text

    grant_role_permission(
        client.tracking_uri,
        username2,
        "experiment",
        exp_id,
        "READ",
    )

    with User(username2, password2, monkeypatch):
        response = _send_rest_tracking_post_request(
            client.tracking_uri,
            "/api/2.0/mlflow/model-versions/create",
            json_payload={"name": rm.name, "source": source, "model_id": model_id},
            auth=(username2, password2),
        )
        assert response.status_code == 200


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_create_model_version_from_own_source_succeeds(
    client: MlflowClient, monkeypatch: pytest.MonkeyPatch
):
    username1, password1 = create_user(client.tracking_uri)

    with User(username1, password1, monkeypatch):
        exp_id = client.create_experiment("own-source-authz-exp")
        run = client.create_run(exp_id)
        run_id = run.info.run_id
        rm = client.create_registered_model("own-source-authz-model")

    # Under no_permission_auth.ini the creator has no default read, so grant READ on the
    # source experiment explicitly — the create must succeed with the source-read guard active.
    grant_role_permission(client.tracking_uri, username1, "experiment", exp_id, "READ")

    with User(username1, password1, monkeypatch):
        mv = client.create_model_version(rm.name, f"{run.info.artifact_uri}/model", run_id=run_id)
        assert mv.name == rm.name


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_create_model_version_empty_source_id_does_not_bypass(
    client: MlflowClient, monkeypatch: pytest.MonkeyPatch
):
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)

    with User(username1, password1, monkeypatch):
        exp_id = client.create_experiment("empty-id-authz-exp")
        run = client.create_run(exp_id)
        source = run.info.artifact_uri

    with User(username2, password2, monkeypatch):
        rm = client.create_registered_model("empty-id-authz-model")

    # An explicitly-supplied empty run_id must not skip the source-read guard: the request
    # is denied rather than slipping past as if run_id were absent.
    response = _send_rest_tracking_post_request(
        client.tracking_uri,
        "/api/2.0/mlflow/model-versions/create",
        json_payload={"name": rm.name, "source": source, "run_id": ""},
        auth=(username2, password2),
    )
    assert response.status_code == 403
    assert "Permission denied" in response.text


def _wait(url: str, timeout: int = 10) -> None:
    t = time.time()
    while time.time() - t < timeout:
        try:
            if requests.get(f"{url}/health").ok:
                return
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(0.5)

    pytest.fail("Server did not start")


# flaky: auto-detected from CI re-runs; see the weekly flaky-test report
@pytest.mark.flaky(attempts=2)
def test_proxy_log_artifacts(monkeypatch, tmp_path):
    backend_uri = f"sqlite:///{tmp_path / 'sqlalchemy.db'}"
    port = get_safe_port()
    host = "localhost"
    env = _isolate_auth_config({MLFLOW_FLASK_SERVER_SECRET_KEY.name: "my-secret-key"}, tmp_path)
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlflow",
            "server",
            "--app-name",
            "basic-auth",
            "--backend-store-uri",
            backend_uri,
            "--host",
            host,
            "--port",
            str(port),
            "--workers",
            "1",
            "--gunicorn-opts",
            "--log-level debug",
        ],
        env=env,
    ) as prc:
        try:
            url = f"http://{host}:{port}"
            _wait(url)

            mlflow.set_tracking_uri(url)
            client = MlflowClient(url)
            tmp_file = tmp_path / "test.txt"
            tmp_file.touch()
            username1, password1 = create_user(url)
            with User(username1, password1, monkeypatch):
                exp_id = client.create_experiment("exp")
                run = client.create_run(exp_id)
                client.log_artifact(run.info.run_id, tmp_file)
                client.list_artifacts(run.info.run_id)

            username2, password2 = create_user(url)
            with User(username2, password2, monkeypatch):
                client.list_artifacts(run.info.run_id)
                with pytest.raises(requests.HTTPError, match="Permission denied"):
                    client.log_artifact(run.info.run_id, tmp_file)

                # Ensure that the regular expression captures an experiment ID correctly
                tmp_file_with_numbers = tmp_path / "123456.txt"
                tmp_file_with_numbers.touch()
                with pytest.raises(requests.HTTPError, match="Permission denied"):
                    client.log_artifact(run.info.run_id, tmp_file_with_numbers)
        finally:
            # Kill the server process to prevent `prc.wait()` (called when exiting the context
            # manager) from waiting forever.
            kill_process_tree(prc.pid)


def test_create_user_from_ui_fails_without_csrf_token(client):
    response = requests.post(
        client.tracking_uri + "/api/2.0/mlflow/users/create-ui",
        json={"username": "test", "password": "test"},
        auth=(ADMIN_USERNAME, ADMIN_PASSWORD),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    assert "The CSRF token is missing" in response.text


def test_create_user_ui(client):
    # needs to be a session as the CSRF protection will set some
    # cookies that need to be present for server side validation
    with requests.Session() as session:
        page = session.get(client.tracking_uri + "/signup", auth=(ADMIN_USERNAME, ADMIN_PASSWORD))

        csrf_regex = re.compile(r"name=\"csrf_token\" value=\"([\S]+)\"")
        match = csrf_regex.search(page.text)

        # assert that the CSRF token is sent in the form
        assert match is not None

        csrf_token = match.group(1)

        response = session.post(
            client.tracking_uri + "/api/2.0/mlflow/users/create-ui",
            data={
                "username": random_str(),
                "password": random_str(),
                "csrf_token": csrf_token,
            },
            auth=(ADMIN_USERNAME, ADMIN_PASSWORD),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        assert "Successfully signed up user" in response.text


def test_logged_model(client: MlflowClient, monkeypatch: pytest.MonkeyPatch):
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)

    class Model(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input):
            return model_input

    with User(username1, password1, monkeypatch):
        exp_id = client.create_experiment("exp")
        model = client.create_logged_model(experiment_id=exp_id)
        client.finalize_logged_model(model_id=model.model_id, status=LoggedModelStatus.READY)
        client.set_logged_model_tags(model_id=model.model_id, tags={"key": "value"})
        client.delete_logged_model_tag(model_id=model.model_id, key="key")
        models = client.search_logged_models(experiment_ids=[exp_id])
        assert len(models) == 1

    with User(username2, password2, monkeypatch):
        loaded_model = client.get_logged_model(model.model_id)
        assert loaded_model.model_id == model.model_id

        models = client.search_logged_models(experiment_ids=[exp_id])
        assert len(models) == 1

        with pytest.raises(MlflowException, match="Permission denied"):
            client.finalize_logged_model(model_id=model.model_id, status=LoggedModelStatus.READY)
        with pytest.raises(MlflowException, match="Permission denied"):
            client.set_logged_model_tags(model_id=model.model_id, tags={"key": "value"})
        with pytest.raises(MlflowException, match="Permission denied"):
            client.delete_logged_model_tag(model_id=model.model_id, key="key")
        with pytest.raises(MlflowException, match="Permission denied"):
            client.delete_logged_model(model_id=model.model_id)


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_logged_model_artifact_authorization(client: MlflowClient, monkeypatch: pytest.MonkeyPatch):
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)

    with User(username1, password1, monkeypatch):
        exp_id = client.create_experiment("logged-model-artifact-authz-test")
        model = client.create_logged_model(experiment_id=exp_id)

    # user1 (owner) should be able to access the artifact endpoint (404 since no artifact
    # exists, but should NOT be 403)
    response = requests.get(
        url=(
            client.tracking_uri
            + f"/ajax-api/2.0/mlflow/logged-models/{model.model_id}/artifacts/files"
        ),
        params={"artifact_file_path": "test.txt"},
        auth=(username1, password1),
    )
    assert response.status_code != 403

    # user2 has no permission on the experiment — expect 403
    response = requests.get(
        url=(
            client.tracking_uri
            + f"/ajax-api/2.0/mlflow/logged-models/{model.model_id}/artifacts/files"
        ),
        params={"artifact_file_path": "test.txt"},
        auth=(username2, password2),
    )
    assert response.status_code == 403

    # Also verify the list-artifacts (directories) endpoint
    # user1 (owner) should be able to list artifacts
    response = requests.get(
        url=(
            client.tracking_uri
            + f"/api/2.0/mlflow/logged-models/{model.model_id}/artifacts/directories"
        ),
        auth=(username1, password1),
    )
    assert response.status_code != 403

    # user2 has no permission — expect 403
    response = requests.get(
        url=(
            client.tracking_uri
            + f"/api/2.0/mlflow/logged-models/{model.model_id}/artifacts/directories"
        ),
        auth=(username2, password2),
    )
    assert response.status_code == 403


def test_logged_model_artifact_validator_respects_static_prefix(
    monkeypatch: pytest.MonkeyPatch,
):
    base = "/mlflow/logged-models/<model_id>/artifacts/files"

    # Without prefix — should match the bare path
    pat_no_prefix = _re_compile_path(_get_ajax_path(base))
    assert pat_no_prefix.fullmatch("/ajax-api/2.0/mlflow/logged-models/abc123/artifacts/files")

    # With prefix — should match the prefixed path
    monkeypatch.setenv(STATIC_PREFIX_ENV_VAR, "/custom-prefix")
    _re_compile_path.cache_clear()
    pat_with_prefix = _re_compile_path(_get_ajax_path(base))
    assert pat_with_prefix.fullmatch(
        "/custom-prefix/ajax-api/2.0/mlflow/logged-models/abc123/artifacts/files"
    )
    # bare path should NOT match the prefixed pattern
    assert not pat_with_prefix.fullmatch(
        "/ajax-api/2.0/mlflow/logged-models/abc123/artifacts/files"
    )

    _re_compile_path.cache_clear()


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_search_logged_models(client: MlflowClient, monkeypatch: pytest.MonkeyPatch):
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)
    readable = [0, 3, 4, 5, 6, 8]
    with User(username1, password1, monkeypatch):
        experiment_ids: list[str] = []
        for i in range(10):
            experiment_id = client.create_experiment(f"exp-{i}")
            experiment_ids.append(experiment_id)
            if i in readable:
                grant_role_permission(
                    client.tracking_uri,
                    username2,
                    "experiment",
                    experiment_id,
                    "READ",
                )
            client.create_logged_model(experiment_id=experiment_id)

        models = client.search_logged_models(experiment_ids=experiment_ids)
        assert len(models) == 10

        # Pagination
        models = client.search_logged_models(experiment_ids=experiment_ids, max_results=2)
        assert len(models) == 2
        assert models.token is not None

        models = client.search_logged_models(
            experiment_ids=experiment_ids, max_results=2, page_token=models.token
        )
        assert len(models) == 2
        assert models.token is not None

        models = client.search_logged_models(experiment_ids=experiment_ids, page_token=models.token)
        assert len(models) == 6
        assert models.token is None

    with User(username2, password2, monkeypatch):
        models = client.search_logged_models(experiment_ids=experiment_ids)
        assert len(models) == len(readable)

        # Pagination
        models = client.search_logged_models(experiment_ids=experiment_ids, max_results=2)
        assert len(models) == 2
        assert models.token is not None

        models = client.search_logged_models(
            experiment_ids=experiment_ids, max_results=2, page_token=models.token
        )
        assert len(models) == 2
        assert models.token is not None

        models = client.search_logged_models(experiment_ids=experiment_ids, page_token=models.token)
        assert len(models) == 2
        assert models.token is None


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_search_runs(client: MlflowClient, monkeypatch: pytest.MonkeyPatch):
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)

    readable = [0, 2]

    with User(username1, password1, monkeypatch):
        experiment_ids: list[str] = []
        run_counts = [8, 10, 7]
        all_runs = {}

        for i in range(3):
            experiment_id = client.create_experiment(f"exp-{i}")
            experiment_ids.append(experiment_id)
            if i in readable:
                grant_role_permission(
                    client.tracking_uri,
                    username2,
                    "experiment",
                    experiment_id,
                    "READ",
                )

            all_runs[experiment_id] = []
            for _ in range(run_counts[i]):
                run = client.create_run(experiment_id)
                all_runs[experiment_id].append(run.info.run_id)

    expected_readable_runs = set(all_runs[experiment_ids[0]] + all_runs[experiment_ids[2]])

    with User(username1, password1, monkeypatch):
        runs = client.search_runs(experiment_ids=experiment_ids)
        assert len(runs) == sum(run_counts)

    with User(username2, password2, monkeypatch):
        runs = client.search_runs(experiment_ids=experiment_ids)
        returned_run_ids = {run.info.run_id for run in runs}
        assert returned_run_ids == expected_readable_runs
        assert len(runs) == len(expected_readable_runs)

        page_token = None
        all_paginated_runs = []
        while True:
            runs = client.search_runs(
                experiment_ids=experiment_ids,
                max_results=3,
                page_token=page_token,
            )
            all_paginated_runs.extend([run.info.run_id for run in runs])
            page_token = runs.token
            if not page_token:
                break

        assert len(all_paginated_runs) == len(set(all_paginated_runs))
        assert set(all_paginated_runs) == expected_readable_runs

        inaccessible_runs = set(all_runs[experiment_ids[1]])
        returned_inaccessible = set(all_paginated_runs) & inaccessible_runs
        assert len(returned_inaccessible) == 0


def test_log_inputs_authorization(client: MlflowClient, monkeypatch: pytest.MonkeyPatch):
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)

    dataset_inputs = [
        DatasetInput(
            dataset=Dataset(
                name="name1",
                digest="digest1",
                source_type="source_type1",
                source="source1",
            ),
            tags=[InputTag(key="context", value="training")],
        )
    ]

    with User(username1, password1, monkeypatch):
        experiment_id = client.create_experiment("log_inputs_authz")
        run_id = client.create_run(experiment_id).info.run_id
        client.log_inputs(run_id, dataset_inputs)

    with User(username2, password2, monkeypatch):
        with pytest.raises(MlflowException, match="Permission denied"):
            client.log_inputs(run_id, dataset_inputs)

    grant_role_permission(client.tracking_uri, username2, "experiment", experiment_id, "EDIT")

    with User(username2, password2, monkeypatch):
        client.log_inputs(run_id, dataset_inputs)


def test_log_outputs_authorization(client: MlflowClient, monkeypatch: pytest.MonkeyPatch):
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)

    with User(username1, password1, monkeypatch):
        experiment_id = client.create_experiment("log_outputs_authz")
        run_id = client.create_run(experiment_id).info.run_id
        model = client.create_logged_model(experiment_id=experiment_id)

    model_outputs = [LoggedModelOutput(model.model_id, 1)]

    with User(username2, password2, monkeypatch):
        with pytest.raises(MlflowException, match="Permission denied"):
            client.log_outputs(run_id, model_outputs)

    grant_role_permission(client.tracking_uri, username2, "experiment", experiment_id, "EDIT")

    with User(username2, password2, monkeypatch):
        client.log_outputs(run_id, model_outputs)


def test_reregister_scorer_does_not_raise(client, monkeypatch):
    username1, password1 = create_user(client.tracking_uri)

    with User(username1, password1, monkeypatch):
        experiment_id = client.create_experiment("test_experiment")

    scorer_json = '{"name": "test_scorer", "type": "pyfunc"}'

    # First registration
    with User(username1, password1, monkeypatch):
        response = _send_rest_tracking_post_request(
            client.tracking_uri,
            "/api/3.0/mlflow/scorers/register",
            json_payload={
                "experiment_id": experiment_id,
                "name": "test_scorer",
                "serialized_scorer": scorer_json,
            },
            auth=(username1, password1),
        )
    assert response.status_code == 200
    assert response.json()["version"] == 1

    # Re-registration with the same name should succeed (not raise RESOURCE_ALREADY_EXISTS)
    updated_scorer_json = '{"name": "test_scorer", "type": "pyfunc", "updated": true}'
    with User(username1, password1, monkeypatch):
        response = _send_rest_tracking_post_request(
            client.tracking_uri,
            "/api/3.0/mlflow/scorers/register",
            json_payload={
                "experiment_id": experiment_id,
                "name": "test_scorer",
                "serialized_scorer": updated_scorer_json,
            },
            auth=(username1, password1),
        )
    assert response.status_code == 200
    assert response.json()["version"] == 2


def test_scorer_permission_denial(client, monkeypatch):
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)

    with User(username1, password1, monkeypatch):
        experiment_id = client.create_experiment("test_experiment")

    scorer_json = '{"name": "test_scorer", "type": "pyfunc"}'

    with User(username1, password1, monkeypatch):
        response = _send_rest_tracking_post_request(
            client.tracking_uri,
            "/api/3.0/mlflow/scorers/register",
            json_payload={
                "experiment_id": experiment_id,
                "name": "test_scorer",
                "serialized_scorer": scorer_json,
            },
            auth=(username1, password1),
        )

    scorer_name = response.json()["name"]

    with User(username2, password2, monkeypatch):
        # user2 has default READ permission, so they CAN read the scorer
        response = requests.get(
            url=client.tracking_uri + "/api/3.0/mlflow/scorers/get",
            params={
                "experiment_id": experiment_id,
                "name": scorer_name,
            },
            auth=(username2, password2),
        )
        response.raise_for_status()
        assert response.json()["scorer"]["scorer_name"] == scorer_name

        # But they CANNOT delete it (READ permission doesn't allow delete)
        response = requests.delete(
            url=client.tracking_uri + "/api/3.0/mlflow/scorers/delete",
            json={
                "experiment_id": experiment_id,
                "name": scorer_name,
            },
            auth=(username2, password2),
        )
        with pytest.raises(requests.HTTPError, match="403"):
            response.raise_for_status()


def test_scorer_read_permission(client, monkeypatch):
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)

    with User(username1, password1, monkeypatch):
        experiment_id = client.create_experiment("test_experiment")

    scorer_json = '{"name": "test_scorer", "type": "pyfunc"}'

    with User(username1, password1, monkeypatch):
        response = _send_rest_tracking_post_request(
            client.tracking_uri,
            "/api/3.0/mlflow/scorers/register",
            json_payload={
                "experiment_id": experiment_id,
                "name": "test_scorer",
                "serialized_scorer": scorer_json,
            },
            auth=(username1, password1),
        )

    scorer_name = response.json()["name"]

    grant_role_permission(
        client.tracking_uri,
        username2,
        "scorer",
        f"{experiment_id}/{scorer_name}",
        "READ",
    )

    with User(username2, password2, monkeypatch):
        response = requests.get(
            url=client.tracking_uri + "/api/3.0/mlflow/scorers/get",
            params={
                "experiment_id": experiment_id,
                "name": scorer_name,
            },
            auth=(username2, password2),
        )
        response.raise_for_status()
        assert response.json()["scorer"]["scorer_name"] == scorer_name

    with User(username2, password2, monkeypatch):
        response = requests.delete(
            url=client.tracking_uri + "/api/3.0/mlflow/scorers/delete",
            json={
                "experiment_id": experiment_id,
                "name": scorer_name,
            },
            auth=(username2, password2),
        )
        with pytest.raises(requests.HTTPError, match="403"):
            response.raise_for_status()


def _graphql_query(tracking_uri, query, variables=None, auth=None):
    return requests.post(
        f"{tracking_uri}/graphql",
        json={"query": query, "variables": variables or {}},
        auth=auth,
    )


def test_graphql_requires_authentication(client, monkeypatch):
    monkeypatch.delenv(MLFLOW_TRACKING_USERNAME.name, raising=False)
    monkeypatch.delenv(MLFLOW_TRACKING_PASSWORD.name, raising=False)

    query = """
    query {
        mlflowGetExperiment(input: {experimentId: "0"}) {
            experiment {
                experimentId
                name
            }
        }
    }
    """
    response = _graphql_query(client.tracking_uri, query)
    assert response.status_code == 401


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_graphql_get_experiment_authorization(client, monkeypatch):
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)

    with User(username1, password1, monkeypatch):
        experiment_id = client.create_experiment("graphql_test_exp")
        # No grant for user2; default_permission=NO_PERMISSIONS denies access.

    query = """
    query($expId: String!) {
        mlflowGetExperiment(input: {experimentId: $expId}) {
            experiment {
                experimentId
                name
            }
        }
    }
    """

    # user1 (creator) should be able to read the experiment
    response = _graphql_query(
        client.tracking_uri,
        query,
        variables={"expId": experiment_id},
        auth=(username1, password1),
    )
    assert response.status_code == 200
    data = response.json()
    experiment_data = data["data"]["mlflowGetExperiment"]["experiment"]
    assert experiment_data["experimentId"] == experiment_id
    assert experiment_data["name"] == "graphql_test_exp"

    # user2 (NO_PERMISSIONS) should NOT be able to read the experiment
    response = _graphql_query(
        client.tracking_uri,
        query,
        variables={"expId": experiment_id},
        auth=(username2, password2),
    )
    assert response.status_code == 200
    data = response.json()
    # With authorization denied, the result should be null
    assert data.get("data", {}).get("mlflowGetExperiment") is None


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_graphql_get_run_authorization(client, monkeypatch):
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)

    with User(username1, password1, monkeypatch):
        experiment_id = client.create_experiment("graphql_run_test_exp")
        run = client.create_run(experiment_id)
        run_id = run.info.run_id
        client.set_terminated(run_id)
        # No grant for user2; default_permission=NO_PERMISSIONS denies access.

    query = """
    query($runId: String!) {
        mlflowGetRun(input: {runId: $runId}) {
            run {
                info {
                    runId
                    experimentId
                }
            }
        }
    }
    """

    # user1 (creator) should be able to read the run
    response = _graphql_query(
        client.tracking_uri,
        query,
        variables={"runId": run_id},
        auth=(username1, password1),
    )
    assert response.status_code == 200
    data = response.json()
    run_data = data["data"]["mlflowGetRun"]["run"]
    assert run_data["info"]["runId"] == run_id
    assert run_data["info"]["experimentId"] == experiment_id

    # user2 (NO_PERMISSIONS) should NOT be able to read the run
    response = _graphql_query(
        client.tracking_uri,
        query,
        variables={"runId": run_id},
        auth=(username2, password2),
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("data", {}).get("mlflowGetRun") is None


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_graphql_search_runs_authorization(client, monkeypatch):
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)

    with User(username1, password1, monkeypatch):
        exp1_id = client.create_experiment("graphql_search_exp1")
        exp2_id = client.create_experiment("graphql_search_exp2")

        run1 = client.create_run(exp1_id)
        client.set_terminated(run1.info.run_id)

        run2 = client.create_run(exp2_id)
        client.set_terminated(run2.info.run_id)

        # Grant READ on exp1 to user2; no grant on exp2 (default_permission
        # is NO_PERMISSIONS, so absence of a grant denies access).
        grant_role_permission(
            client.tracking_uri,
            username2,
            "experiment",
            exp1_id,
            "READ",
        )

    query = """
    query($expIds: [String]!) {
        mlflowSearchRuns(input: {experimentIds: $expIds}) {
            runs {
                info {
                    runId
                    experimentId
                }
            }
        }
    }
    """

    # user1 should see both runs
    response = _graphql_query(
        client.tracking_uri,
        query,
        variables={"expIds": [exp1_id, exp2_id]},
        auth=(username1, password1),
    )
    assert response.status_code == 200
    data = response.json()
    runs = data.get("data", {}).get("mlflowSearchRuns", {}).get("runs", [])
    assert len(runs) == 2

    # user2 should only see run from exp1 (exp2 is filtered out)
    response = _graphql_query(
        client.tracking_uri,
        query,
        variables={"expIds": [exp1_id, exp2_id]},
        auth=(username2, password2),
    )
    assert response.status_code == 200
    data = response.json()
    runs = data.get("data", {}).get("mlflowSearchRuns", {}).get("runs", [])
    assert len(runs) == 1
    assert runs[0]["info"]["experimentId"] == exp1_id


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_graphql_list_artifacts_authorization(client, monkeypatch):
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)

    with User(username1, password1, monkeypatch):
        experiment_id = client.create_experiment("graphql_artifacts_test_exp")
        run = client.create_run(experiment_id)
        run_id = run.info.run_id
        client.set_terminated(run_id)
        # No grant for user2; default_permission=NO_PERMISSIONS denies access.

    query = """
    query($runId: String!) {
        mlflowListArtifacts(input: {runId: $runId}) {
            rootUri
            files {
                path
            }
        }
    }
    """

    # user1 (creator) should be able to list artifacts
    response = _graphql_query(
        client.tracking_uri,
        query,
        variables={"runId": run_id},
        auth=(username1, password1),
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("data", {}).get("mlflowListArtifacts") is not None

    # user2 (NO_PERMISSIONS) should NOT be able to list artifacts
    response = _graphql_query(
        client.tracking_uri,
        query,
        variables={"runId": run_id},
        auth=(username2, password2),
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("data", {}).get("mlflowListArtifacts") is None


def test_graphql_nonexistent_experiment(client, monkeypatch):
    username, password = create_user(client.tracking_uri)

    query = """
    query($expId: String!) {
        mlflowGetExperiment(input: {experimentId: $expId}) {
            experiment {
                experimentId
                name
            }
        }
    }
    """

    response = _graphql_query(
        client.tracking_uri,
        query,
        variables={"expId": "999999999"},
        auth=(username, password),
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("data", {}).get("mlflowGetExperiment") is None


def test_graphql_nonexistent_run(client, monkeypatch):
    username, password = create_user(client.tracking_uri)

    query = """
    query($runId: String!) {
        mlflowGetRun(input: {runId: $runId}) {
            run {
                info {
                    runId
                    experimentId
                }
            }
        }
    }
    """

    response = _graphql_query(
        client.tracking_uri,
        query,
        variables={"runId": "00000000000000000000000000000000"},
        auth=(username, password),
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("data", {}).get("mlflowGetRun") is None


def test_get_metric_history_bulk_interval_auth(client: MlflowClient, monkeypatch):
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)

    with User(username1, password1, monkeypatch):
        experiment_id = client.create_experiment("test_metric_history_experiment")
        run = client.create_run(experiment_id)
        run_id = run.info.run_id
        client.log_metric(run_id, "test_metric", 1.0, step=0)
        client.log_metric(run_id, "test_metric", 2.0, step=1)

        grant_role_permission(
            client.tracking_uri,
            username2,
            "experiment",
            experiment_id,
            "READ",
        )

    with User(username2, password2, monkeypatch):
        response = requests.get(
            url=client.tracking_uri + "/ajax-api/2.0/mlflow/metrics/get-history-bulk-interval",
            params={
                "run_ids": run_id,
                "metric_key": "test_metric",
                "max_results": 100,
            },
            auth=(username2, password2),
        )
        response.raise_for_status()
        data = response.json()
        assert "metrics" in data
        assert len(data["metrics"]) == 2


def test_gateway_secrets_permissions(client, monkeypatch):
    user1, password1 = create_user(client.tracking_uri)
    user2, password2 = create_user(client.tracking_uri)

    with User(user1, password1, monkeypatch):
        response = requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/secrets/create",
            json={
                "secret_name": "user1_secret",
                "secret_value": {"api_key": "test-key"},
                "provider": "openai",
            },
            auth=(user1, password1),
        )
        response.raise_for_status()
        user1_secret_id = response.json()["secret"]["secret_id"]

    with User(user1, password1, monkeypatch):
        response = requests.get(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/secrets/get",
            params={"secret_id": user1_secret_id},
            auth=(user1, password1),
        )
        response.raise_for_status()

    with User(user1, password1, monkeypatch):
        response = requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/secrets/update",
            json={
                "secret_id": user1_secret_id,
                "secret_value": {"api_key": "updated-key"},
            },
            auth=(user1, password1),
        )
        response.raise_for_status()

    # User2 can read secrets by default (READ permission is default)
    with User(user2, password2, monkeypatch):
        response = requests.get(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/secrets/get",
            params={"secret_id": user1_secret_id},
            auth=(user2, password2),
        )
        response.raise_for_status()

    # User2 cannot update secrets without explicit permission
    with User(user2, password2, monkeypatch):
        response = requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/secrets/update",
            json={
                "secret_id": user1_secret_id,
                "secret_value": {"api_key": "hacked-key"},
            },
            auth=(user2, password2),
        )
        assert response.status_code == 403

    # User2 cannot delete secrets without explicit permission
    with User(user2, password2, monkeypatch):
        response = requests.delete(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/secrets/delete",
            json={"secret_id": user1_secret_id},
            auth=(user2, password2),
        )
        assert response.status_code == 403

    with User(user1, password1, monkeypatch):
        response = requests.get(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/secrets/list",
            auth=(user1, password1),
        )
        response.raise_for_status()

    with User(user1, password1, monkeypatch):
        response = requests.get(
            url=client.tracking_uri + "/ajax-api/3.0/mlflow/gateway/secrets/config",
            auth=(user1, password1),
        )
        response.raise_for_status()
        assert "secrets_available" in response.json()

    with User(user1, password1, monkeypatch):
        response = requests.delete(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/secrets/delete",
            json={"secret_id": user1_secret_id},
            auth=(user1, password1),
        )
        response.raise_for_status()


def test_gateway_endpoints_permissions(client, monkeypatch):
    user1, password1 = create_user(client.tracking_uri)
    user2, password2 = create_user(client.tracking_uri)

    with User(user1, password1, monkeypatch):
        response = requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/secrets/create",
            json={
                "secret_name": "user1_secret_for_endpoint",
                "secret_value": {"api_key": "test-key"},
                "provider": "openai",
            },
            auth=(user1, password1),
        )
        response.raise_for_status()
        secret_id = response.json()["secret"]["secret_id"]

    with User(user1, password1, monkeypatch):
        response = requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/model-definitions/create",
            json={
                "name": "user1_model_def",
                "secret_id": secret_id,
                "provider": "openai",
                "model_name": "gpt-4",
            },
            auth=(user1, password1),
        )
        response.raise_for_status()
        model_definition_id = response.json()["model_definition"]["model_definition_id"]

    with User(user1, password1, monkeypatch):
        response = requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/endpoints/create",
            json={
                "name": "user1_endpoint",
                "model_configs": [
                    {
                        "model_definition_id": model_definition_id,
                        "linkage_type": "PRIMARY",
                    }
                ],
            },
            auth=(user1, password1),
        )
        response.raise_for_status()
        endpoint_id = response.json()["endpoint"]["endpoint_id"]

    with User(user1, password1, monkeypatch):
        response = requests.get(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/endpoints/list",
            auth=(user1, password1),
        )
        response.raise_for_status()

    with User(user1, password1, monkeypatch):
        response = requests.get(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/endpoints/get",
            params={"endpoint_id": endpoint_id},
            auth=(user1, password1),
        )
        response.raise_for_status()

    with User(user1, password1, monkeypatch):
        response = requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/endpoints/update",
            json={
                "endpoint_id": endpoint_id,
                "name": "updated_endpoint",
            },
            auth=(user1, password1),
        )
        response.raise_for_status()

    # User2 can read endpoints by default (READ permission is default)
    with User(user2, password2, monkeypatch):
        response = requests.get(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/endpoints/get",
            params={"endpoint_id": endpoint_id},
            auth=(user2, password2),
        )
        response.raise_for_status()

    # User2 cannot update endpoints without explicit permission
    with User(user2, password2, monkeypatch):
        response = requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/endpoints/update",
            json={
                "endpoint_id": endpoint_id,
                "name": "hacked_endpoint",
            },
            auth=(user2, password2),
        )
        assert response.status_code == 403

    # User2 cannot delete endpoints without explicit permission
    with User(user2, password2, monkeypatch):
        response = requests.delete(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/endpoints/delete",
            json={"endpoint_id": endpoint_id},
            auth=(user2, password2),
        )
        assert response.status_code == 403

    with User(user1, password1, monkeypatch):
        response = requests.delete(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/endpoints/delete",
            json={"endpoint_id": endpoint_id},
            auth=(user1, password1),
        )
        response.raise_for_status()

    with User(user1, password1, monkeypatch):
        response = requests.delete(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/model-definitions/delete",
            json={"model_definition_id": model_definition_id},
            auth=(user1, password1),
        )
        response.raise_for_status()

    with User(user1, password1, monkeypatch):
        response = requests.delete(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/secrets/delete",
            json={"secret_id": secret_id},
            auth=(user1, password1),
        )
        response.raise_for_status()


def test_gateway_model_definitions_permissions(client, monkeypatch):
    user1, password1 = create_user(client.tracking_uri)
    user2, password2 = create_user(client.tracking_uri)

    with User(user1, password1, monkeypatch):
        response = requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/secrets/create",
            json={
                "secret_name": "user1_secret_for_model_def",
                "secret_value": {"api_key": "test-key"},
                "provider": "openai",
            },
            auth=(user1, password1),
        )
        response.raise_for_status()
        secret_id = response.json()["secret"]["secret_id"]

    with User(user1, password1, monkeypatch):
        response = requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/model-definitions/create",
            json={
                "name": "user1_model_def",
                "secret_id": secret_id,
                "provider": "openai",
                "model_name": "gpt-4",
            },
            auth=(user1, password1),
        )
        response.raise_for_status()
        model_definition_id = response.json()["model_definition"]["model_definition_id"]

    with User(user1, password1, monkeypatch):
        response = requests.get(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/model-definitions/list",
            auth=(user1, password1),
        )
        response.raise_for_status()

    with User(user1, password1, monkeypatch):
        response = requests.get(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/model-definitions/get",
            params={"model_definition_id": model_definition_id},
            auth=(user1, password1),
        )
        response.raise_for_status()

    with User(user1, password1, monkeypatch):
        response = requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/model-definitions/update",
            json={
                "model_definition_id": model_definition_id,
                "name": "updated_model_def",
            },
            auth=(user1, password1),
        )
        response.raise_for_status()

    # User2 can read model definitions by default (READ permission is default)
    with User(user2, password2, monkeypatch):
        response = requests.get(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/model-definitions/get",
            params={"model_definition_id": model_definition_id},
            auth=(user2, password2),
        )
        response.raise_for_status()

    # User2 cannot update model definitions without explicit permission
    with User(user2, password2, monkeypatch):
        response = requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/model-definitions/update",
            json={
                "model_definition_id": model_definition_id,
                "name": "hacked_model_def",
            },
            auth=(user2, password2),
        )
        assert response.status_code == 403

    # User2 cannot delete model definitions without explicit permission
    with User(user2, password2, monkeypatch):
        response = requests.delete(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/model-definitions/delete",
            json={"model_definition_id": model_definition_id},
            auth=(user2, password2),
        )
        assert response.status_code == 403

    with User(user1, password1, monkeypatch):
        response = requests.delete(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/model-definitions/delete",
            json={"model_definition_id": model_definition_id},
            auth=(user1, password1),
        )
        response.raise_for_status()

    with User(user1, password1, monkeypatch):
        response = requests.delete(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/secrets/delete",
            json={"secret_id": secret_id},
            auth=(user1, password1),
        )
        response.raise_for_status()


def test_gateway_budget_policy_admin_only(client, monkeypatch):
    user1, password1 = create_user(client.tracking_uri)

    # Admin creates a budget policy
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        response = requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/budgets/create",
            json={
                "budget_unit": "USD",
                "budget_amount": 100.0,
                "duration": {"unit": "DAYS", "value": 30},
                "target_scope": "GLOBAL",
                "budget_action": "ALERT",
            },
            auth=(ADMIN_USERNAME, ADMIN_PASSWORD),
        )
        response.raise_for_status()
        budget_policy_id = response.json()["budget_policy"]["budget_policy_id"]

    # Non-admin can list budget policies
    with User(user1, password1, monkeypatch):
        response = requests.get(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/budgets/list",
            auth=(user1, password1),
        )
        response.raise_for_status()

    # Non-admin can get a budget policy
    with User(user1, password1, monkeypatch):
        response = requests.get(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/budgets/get",
            params={"budget_policy_id": budget_policy_id},
            auth=(user1, password1),
        )
        response.raise_for_status()

    # Non-admin cannot create a budget policy
    with User(user1, password1, monkeypatch):
        response = requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/budgets/create",
            json={
                "budget_unit": "USD",
                "budget_amount": 50.0,
                "duration": {"unit": "DAYS", "value": 7},
                "target_scope": "GLOBAL",
                "budget_action": "REJECT",
            },
            auth=(user1, password1),
        )
        assert response.status_code == 403

    # Non-admin cannot update a budget policy
    with User(user1, password1, monkeypatch):
        response = requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/budgets/update",
            json={
                "budget_policy_id": budget_policy_id,
                "budget_amount": 200.0,
            },
            auth=(user1, password1),
        )
        assert response.status_code == 403

    # Non-admin cannot delete a budget policy
    with User(user1, password1, monkeypatch):
        response = requests.delete(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/budgets/delete",
            json={"budget_policy_id": budget_policy_id},
            auth=(user1, password1),
        )
        assert response.status_code == 403

    # Admin can delete the budget policy
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        response = requests.delete(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/budgets/delete",
            json={"budget_policy_id": budget_policy_id},
            auth=(ADMIN_USERNAME, ADMIN_PASSWORD),
        )
        response.raise_for_status()


def test_gateway_ajax_routes_permissions(client, monkeypatch):
    username, password = create_user(client.tracking_uri)

    with User(username, password, monkeypatch):
        response = requests.get(
            url=client.tracking_uri + "/ajax-api/3.0/mlflow/gateway/supported-providers",
            auth=(username, password),
        )
        response.raise_for_status()
        assert "providers" in response.json()

    with User(username, password, monkeypatch):
        response = requests.get(
            url=client.tracking_uri + "/ajax-api/3.0/mlflow/gateway/supported-models",
            auth=(username, password),
        )
        response.raise_for_status()

    with User(username, password, monkeypatch):
        response = requests.get(
            url=client.tracking_uri + "/ajax-api/3.0/mlflow/gateway/provider-config",
            params={"provider": "openai"},
            auth=(username, password),
        )
        response.raise_for_status()

    with User(username, password, monkeypatch):
        response = requests.get(
            url=client.tracking_uri + "/ajax-api/3.0/mlflow/gateway/secrets/config",
            auth=(username, password),
        )
        response.raise_for_status()
        assert "secrets_available" in response.json()


def test_gateway_unauthenticated_access_denied(client, monkeypatch):
    monkeypatch.delenv(MLFLOW_TRACKING_USERNAME.name, raising=False)
    monkeypatch.delenv(MLFLOW_TRACKING_PASSWORD.name, raising=False)

    response = requests.get(
        url=client.tracking_uri + "/api/3.0/mlflow/gateway/secrets/list",
    )
    assert response.status_code == 401

    response = requests.get(
        url=client.tracking_uri + "/api/3.0/mlflow/gateway/endpoints/list",
    )
    assert response.status_code == 401

    response = requests.get(
        url=client.tracking_uri + "/api/3.0/mlflow/gateway/model-definitions/list",
    )
    assert response.status_code == 401

    response = requests.get(
        url=client.tracking_uri + "/ajax-api/3.0/mlflow/gateway/supported-providers",
    )
    assert response.status_code == 401


def test_gateway_endpoint_use_permission(fastapi_client, monkeypatch):
    user1, password1 = create_user(fastapi_client.tracking_uri)
    user2, password2 = create_user(fastapi_client.tracking_uri)

    # User1 creates a secret, model definition, and endpoint
    with User(user1, password1, monkeypatch):
        response = requests.post(
            url=fastapi_client.tracking_uri + "/api/3.0/mlflow/gateway/secrets/create",
            json={
                "secret_name": "test_secret",
                "secret_value": {"api_key": "test-key"},
                "provider": "openai",
            },
            auth=(user1, password1),
        )
        response.raise_for_status()
        secret_id = response.json()["secret"]["secret_id"]

    with User(user1, password1, monkeypatch):
        response = requests.post(
            url=fastapi_client.tracking_uri + "/api/3.0/mlflow/gateway/model-definitions/create",
            json={
                "name": "test_model_def",
                "secret_id": secret_id,
                "provider": "openai",
                "model_name": "gpt-4",
            },
            auth=(user1, password1),
        )
        response.raise_for_status()
        model_definition_id = response.json()["model_definition"]["model_definition_id"]

    with User(user1, password1, monkeypatch):
        response = requests.post(
            url=fastapi_client.tracking_uri + "/api/3.0/mlflow/gateway/endpoints/create",
            json={
                "name": "test_endpoint",
                "model_configs": [
                    {
                        "model_definition_id": model_definition_id,
                        "linkage_type": "PRIMARY",
                    }
                ],
            },
            auth=(user1, password1),
        )
        response.raise_for_status()
        endpoint_id = response.json()["endpoint"]["endpoint_id"]
        endpoint_name = response.json()["endpoint"]["name"]

    # User2 without permission cannot invoke the endpoint
    with User(user2, password2, monkeypatch):
        response = requests.post(
            url=fastapi_client.tracking_uri + f"/gateway/{endpoint_name}/mlflow/invocations",
            json={"messages": [{"role": "user", "content": "test"}]},
            auth=(user2, password2),
        )
        assert response.status_code == 403

    # Grant USE permission to user2
    with User(user1, password1, monkeypatch):
        grant_role_permission(
            fastapi_client.tracking_uri,
            user2,
            "gateway_endpoint",
            endpoint_id,
            "USE",
        )

    # User2 with USE permission can invoke
    with User(user2, password2, monkeypatch):
        response = requests.post(
            url=fastapi_client.tracking_uri + f"/gateway/{endpoint_name}/mlflow/invocations",
            json={"messages": [{"role": "user", "content": "test"}]},
            auth=(user2, password2),
        )
        # Will fail because we don't have real LLM credentials, but should pass auth (not 403)
        assert response.status_code != 403

    # Cleanup
    with User(user1, password1, monkeypatch):
        requests.delete(
            url=fastapi_client.tracking_uri + "/api/3.0/mlflow/gateway/endpoints/delete",
            json={"endpoint_id": endpoint_id},
            auth=(user1, password1),
        ).raise_for_status()
        requests.delete(
            url=fastapi_client.tracking_uri + "/api/3.0/mlflow/gateway/model-definitions/delete",
            json={"model_definition_id": model_definition_id},
            auth=(user1, password1),
        ).raise_for_status()
        requests.delete(
            url=fastapi_client.tracking_uri + "/api/3.0/mlflow/gateway/secrets/delete",
            json={"secret_id": secret_id},
            auth=(user1, password1),
        ).raise_for_status()


def test_gateway_endpoint_use_permission_with_workspaces(fastapi_workspace_client):
    tracking_uri = fastapi_workspace_client.tracking_uri
    admin_auth = (ADMIN_USERNAME, ADMIN_PASSWORD)
    workspace_headers = {"X-MLFLOW-WORKSPACE": DEFAULT_WORKSPACE_NAME}
    user, password = create_user(tracking_uri)

    response = requests.post(
        url=tracking_uri + "/api/3.0/mlflow/gateway/secrets/create",
        json={
            "secret_name": "test_secret",
            "secret_value": {"api_key": "test-key"},
            "provider": "openai",
        },
        auth=admin_auth,
        headers=workspace_headers,
    )
    response.raise_for_status()
    secret_id = response.json()["secret"]["secret_id"]

    response = requests.post(
        url=tracking_uri + "/api/3.0/mlflow/gateway/model-definitions/create",
        json={
            "name": "test_model_def",
            "secret_id": secret_id,
            "provider": "openai",
            "model_name": "gpt-4",
        },
        auth=admin_auth,
        headers=workspace_headers,
    )
    response.raise_for_status()
    model_definition_id = response.json()["model_definition"]["model_definition_id"]

    response = requests.post(
        url=tracking_uri + "/api/3.0/mlflow/gateway/endpoints/create",
        json={
            "name": "test_endpoint",
            "model_configs": [
                {
                    "model_definition_id": model_definition_id,
                    "linkage_type": "PRIMARY",
                }
            ],
        },
        auth=admin_auth,
        headers=workspace_headers,
    )
    response.raise_for_status()
    endpoint_id = response.json()["endpoint"]["endpoint_id"]
    endpoint_name = response.json()["endpoint"]["name"]

    # Without a grant the invocation is denied.
    response = requests.post(
        url=tracking_uri + f"/gateway/{endpoint_name}/mlflow/invocations",
        json={"messages": [{"role": "user", "content": "test"}]},
        auth=(user, password),
        headers=workspace_headers,
    )
    assert response.status_code == 403

    grant_role_permission(tracking_uri, user, "gateway_endpoint", endpoint_id, "USE")

    # With USE granted the request must clear authorization. It then fails on the
    # fake provider credentials, so anything but 403 means authorization passed.
    response = requests.post(
        url=tracking_uri + f"/gateway/{endpoint_name}/mlflow/invocations",
        json={"messages": [{"role": "user", "content": "test"}]},
        auth=(user, password),
        headers=workspace_headers,
    )
    assert response.status_code != 403


def test_gateway_proxy_authenticates_via_mlflow_auth_header(fastapi_client, monkeypatch):
    user1, password1 = create_user(fastapi_client.tracking_uri)
    user2, password2 = create_user(fastapi_client.tracking_uri)

    with User(user1, password1, monkeypatch):
        response = requests.post(
            url=fastapi_client.tracking_uri + "/api/3.0/mlflow/gateway/secrets/create",
            json={
                "secret_name": "proxy_secret",
                "secret_value": {"api_key": "test-key"},
                "provider": "openai",
            },
            auth=(user1, password1),
        )
        response.raise_for_status()
        secret_id = response.json()["secret"]["secret_id"]

        response = requests.post(
            url=fastapi_client.tracking_uri + "/api/3.0/mlflow/gateway/model-definitions/create",
            json={
                "name": "proxy_model_def",
                "secret_id": secret_id,
                "provider": "openai",
                "model_name": "gpt-4",
            },
            auth=(user1, password1),
        )
        response.raise_for_status()
        model_definition_id = response.json()["model_definition"]["model_definition_id"]

        response = requests.post(
            url=fastapi_client.tracking_uri + "/api/3.0/mlflow/gateway/endpoints/create",
            json={
                "name": "proxy_endpoint",
                "model_configs": [
                    {"model_definition_id": model_definition_id, "linkage_type": "PRIMARY"}
                ],
            },
            auth=(user1, password1),
        )
        response.raise_for_status()
        endpoint_id = response.json()["endpoint"]["endpoint_id"]
        endpoint_name = response.json()["endpoint"]["name"]

    with User(user1, password1, monkeypatch):
        grant_role_permission(
            fastapi_client.tracking_uri,
            user2,
            "gateway_endpoint",
            endpoint_id,
            "USE",
        )

    mlflow_auth = "Basic " + base64.b64encode(f"{user2}:{password2}".encode()).decode("ascii")
    proxy_url = fastapi_client.tracking_uri + f"/gateway/proxy/{endpoint_name}/v1/responses"

    # The coding agent's own provider key occupies Authorization; MLflow creds ride in
    # X-MLflow-Authorization. Auth must clear the middleware (the upstream call then fails
    # on the fake key, but that is NOT the middleware's 401/403).
    response = requests.post(
        proxy_url,
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={
            "Authorization": "Bearer sk-decoy-provider-key",
            "X-MLflow-Authorization": mlflow_auth,
            "User-Agent": "codex_cli_rs/1.0",
        },
    )
    assert "You are not authenticated" not in response.text
    assert "Permission denied" not in response.text

    # Without the MLflow auth header, the decoy Bearer alone must be rejected by the middleware.
    response = requests.post(
        proxy_url,
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={
            "Authorization": "Bearer sk-decoy-provider-key",
            "User-Agent": "codex_cli_rs/1.0",
        },
    )
    assert response.status_code == 401
    assert "You are not authenticated" in response.text

    with User(user1, password1, monkeypatch):
        requests.delete(
            url=fastapi_client.tracking_uri + "/api/3.0/mlflow/gateway/endpoints/delete",
            json={"endpoint_id": endpoint_id},
            auth=(user1, password1),
        ).raise_for_status()
        requests.delete(
            url=fastapi_client.tracking_uri + "/api/3.0/mlflow/gateway/model-definitions/delete",
            json={"model_definition_id": model_definition_id},
            auth=(user1, password1),
        ).raise_for_status()
        requests.delete(
            url=fastapi_client.tracking_uri + "/api/3.0/mlflow/gateway/secrets/delete",
            json={"secret_id": secret_id},
            auth=(user1, password1),
        ).raise_for_status()


def test_gateway_model_definition_requires_secret_use_permission(client, monkeypatch):
    user1, password1 = create_user(client.tracking_uri)
    user2, password2 = create_user(client.tracking_uri)

    # User1 creates a secret
    with User(user1, password1, monkeypatch):
        response = requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/secrets/create",
            json={
                "secret_name": "user1_secret",
                "secret_value": {"api_key": "test-key"},
                "provider": "openai",
            },
            auth=(user1, password1),
        )
        response.raise_for_status()
        secret_id = response.json()["secret"]["secret_id"]

    # User2 cannot create a model definition using user1's secret (no permission)
    with User(user2, password2, monkeypatch):
        response = requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/model-definitions/create",
            json={
                "name": "model_def_1",
                "secret_id": secret_id,
                "provider": "openai",
                "model_name": "gpt-4",
            },
            auth=(user2, password2),
        )
        assert response.status_code == 403

    # Grant USE permission to user2
    with User(user1, password1, monkeypatch):
        grant_role_permission(
            client.tracking_uri,
            user2,
            "gateway_secret",
            secret_id,
            "USE",
        )

    # User2 can now create a model definition using user1's secret
    with User(user2, password2, monkeypatch):
        response = requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/model-definitions/create",
            json={
                "name": "model_def_1",
                "secret_id": secret_id,
                "provider": "openai",
                "model_name": "gpt-4",
            },
            auth=(user2, password2),
        )
        response.raise_for_status()
        model_def_id = response.json()["model_definition"]["model_definition_id"]

    # User1 creates another secret
    with User(user1, password1, monkeypatch):
        response = requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/secrets/create",
            json={
                "secret_name": "user1_secret_2",
                "secret_value": {"api_key": "test-key-2"},
                "provider": "anthropic",
            },
            auth=(user1, password1),
        )
        response.raise_for_status()
        secret_id_2 = response.json()["secret"]["secret_id"]

    # User2 cannot update the model definition to use secret_id_2 (no permission on that secret)
    with User(user2, password2, monkeypatch):
        response = requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/model-definitions/update",
            json={
                "model_definition_id": model_def_id,
                "secret_id": secret_id_2,
                "provider": "anthropic",
            },
            auth=(user2, password2),
        )
        assert response.status_code == 403

    # Grant USE permission to user2 on secret_id_2
    with User(user1, password1, monkeypatch):
        grant_role_permission(
            client.tracking_uri,
            user2,
            "gateway_secret",
            secret_id_2,
            "USE",
        )

    # User2 can now update the model definition to use secret_id_2
    with User(user2, password2, monkeypatch):
        response = requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/model-definitions/update",
            json={
                "model_definition_id": model_def_id,
                "secret_id": secret_id_2,
                "provider": "anthropic",
            },
            auth=(user2, password2),
        )
        response.raise_for_status()

    # Cleanup
    with User(user2, password2, monkeypatch):
        requests.delete(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/model-definitions/delete",
            json={"model_definition_id": model_def_id},
            auth=(user2, password2),
        ).raise_for_status()

    with User(user1, password1, monkeypatch):
        requests.delete(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/secrets/delete",
            json={"secret_id": secret_id},
            auth=(user1, password1),
        ).raise_for_status()
        requests.delete(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/secrets/delete",
            json={"secret_id": secret_id_2},
            auth=(user1, password1),
        ).raise_for_status()


def test_gateway_endpoint_requires_model_definition_use_permission(client, monkeypatch):
    user1, password1 = create_user(client.tracking_uri)
    user2, password2 = create_user(client.tracking_uri)

    # User1 creates a secret and model definition
    with User(user1, password1, monkeypatch):
        response = requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/secrets/create",
            json={
                "secret_name": "user1_secret",
                "secret_value": {"api_key": "test-key"},
                "provider": "openai",
            },
            auth=(user1, password1),
        )
        response.raise_for_status()
        secret_id = response.json()["secret"]["secret_id"]

        response = requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/model-definitions/create",
            json={
                "name": "model_def_1",
                "secret_id": secret_id,
                "provider": "openai",
                "model_name": "gpt-4",
            },
            auth=(user1, password1),
        )
        response.raise_for_status()
        model_def_id = response.json()["model_definition"]["model_definition_id"]

    # User2 cannot create an endpoint using user1's model definition (no permission)
    with User(user2, password2, monkeypatch):
        response = requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/endpoints/create",
            json={
                "name": "endpoint_1",
                "model_configs": [
                    {
                        "model_definition_id": model_def_id,
                        "linkage_type": "PRIMARY",
                    }
                ],
            },
            auth=(user2, password2),
        )
        assert response.status_code == 403

    # Grant USE permission to user2 on the model definition
    with User(user1, password1, monkeypatch):
        grant_role_permission(
            client.tracking_uri,
            user2,
            "gateway_model_definition",
            model_def_id,
            "USE",
        )

    # User2 can now create an endpoint using user1's model definition
    with User(user2, password2, monkeypatch):
        response = requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/endpoints/create",
            json={
                "name": "endpoint_1",
                "model_configs": [
                    {
                        "model_definition_id": model_def_id,
                        "linkage_type": "PRIMARY",
                    }
                ],
            },
            auth=(user2, password2),
        )
        response.raise_for_status()
        endpoint_id = response.json()["endpoint"]["endpoint_id"]

    # User1 creates another model definition
    with User(user1, password1, monkeypatch):
        response = requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/model-definitions/create",
            json={
                "name": "model_def_2",
                "secret_id": secret_id,
                "provider": "openai",
                "model_name": "gpt-3.5-turbo",
            },
            auth=(user1, password1),
        )
        response.raise_for_status()
        model_def_id_2 = response.json()["model_definition"]["model_definition_id"]

    # User2 cannot update the endpoint to use model_def_id_2 (no permission on that model def)
    with User(user2, password2, monkeypatch):
        response = requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/endpoints/update",
            json={
                "endpoint_id": endpoint_id,
                "model_configs": [
                    {
                        "model_definition_id": model_def_id_2,
                        "linkage_type": "PRIMARY",
                    }
                ],
            },
            auth=(user2, password2),
        )
        assert response.status_code == 403

    # Grant USE permission to user2 on model_def_id_2
    with User(user1, password1, monkeypatch):
        grant_role_permission(
            client.tracking_uri,
            user2,
            "gateway_model_definition",
            model_def_id_2,
            "USE",
        )

    # User2 can now update the endpoint to use model_def_id_2
    with User(user2, password2, monkeypatch):
        response = requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/endpoints/update",
            json={
                "endpoint_id": endpoint_id,
                "model_configs": [
                    {
                        "model_definition_id": model_def_id_2,
                        "linkage_type": "PRIMARY",
                    }
                ],
            },
            auth=(user2, password2),
        )
        response.raise_for_status()

    # Cleanup
    with User(user2, password2, monkeypatch):
        requests.delete(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/endpoints/delete",
            json={"endpoint_id": endpoint_id},
            auth=(user2, password2),
        ).raise_for_status()

    with User(user1, password1, monkeypatch):
        requests.delete(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/model-definitions/delete",
            json={"model_definition_id": model_def_id},
            auth=(user1, password1),
        ).raise_for_status()
        requests.delete(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/model-definitions/delete",
            json={"model_definition_id": model_def_id_2},
            auth=(user1, password1),
        ).raise_for_status()
        requests.delete(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/secrets/delete",
            json={"secret_id": secret_id},
            auth=(user1, password1),
        ).raise_for_status()


def test_gateway_endpoint_requires_fallback_model_definition_use_permission(client, monkeypatch):
    user1, password1 = create_user(client.tracking_uri)
    user2, password2 = create_user(client.tracking_uri)

    # User1 creates secrets and model definitions
    with User(user1, password1, monkeypatch):
        response = requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/secrets/create",
            json={
                "secret_name": "user1_secret",
                "secret_value": {"api_key": "test-key"},
                "provider": "openai",
            },
            auth=(user1, password1),
        )
        response.raise_for_status()
        secret_id = response.json()["secret"]["secret_id"]

        # Create primary model definition
        response = requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/model-definitions/create",
            json={
                "name": "primary_model",
                "secret_id": secret_id,
                "provider": "openai",
                "model_name": "gpt-4",
            },
            auth=(user1, password1),
        )
        response.raise_for_status()
        primary_model_def_id = response.json()["model_definition"]["model_definition_id"]

        # Create fallback model definition
        response = requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/model-definitions/create",
            json={
                "name": "fallback_model",
                "secret_id": secret_id,
                "provider": "openai",
                "model_name": "gpt-3.5-turbo",
            },
            auth=(user1, password1),
        )
        response.raise_for_status()
        fallback_model_def_id = response.json()["model_definition"]["model_definition_id"]

    # Grant USE permission to user2 on primary model but not fallback
    with User(user1, password1, monkeypatch):
        grant_role_permission(
            client.tracking_uri,
            user2,
            "gateway_model_definition",
            primary_model_def_id,
            "USE",
        )

    # User2 cannot create an endpoint with fallback model (no permission on fallback)
    with User(user2, password2, monkeypatch):
        response = requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/endpoints/create",
            json={
                "name": "endpoint_with_fallback",
                "model_configs": [
                    {
                        "model_definition_id": primary_model_def_id,
                        "linkage_type": "PRIMARY",
                    },
                    {
                        "model_definition_id": fallback_model_def_id,
                        "linkage_type": "FALLBACK",
                        "fallback_order": 1,
                    },
                ],
            },
            auth=(user2, password2),
        )
        assert response.status_code == 403

    # Grant USE permission to user2 on fallback model
    with User(user1, password1, monkeypatch):
        grant_role_permission(
            client.tracking_uri,
            user2,
            "gateway_model_definition",
            fallback_model_def_id,
            "USE",
        )

    # User2 can now create an endpoint with both primary and fallback models
    with User(user2, password2, monkeypatch):
        response = requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/endpoints/create",
            json={
                "name": "endpoint_with_fallback",
                "model_configs": [
                    {
                        "model_definition_id": primary_model_def_id,
                        "linkage_type": "PRIMARY",
                    },
                    {
                        "model_definition_id": fallback_model_def_id,
                        "linkage_type": "FALLBACK",
                        "fallback_order": 1,
                    },
                ],
            },
            auth=(user2, password2),
        )
        response.raise_for_status()
        endpoint_id = response.json()["endpoint"]["endpoint_id"]

    # Cleanup
    with User(user2, password2, monkeypatch):
        requests.delete(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/endpoints/delete",
            json={"endpoint_id": endpoint_id},
            auth=(user2, password2),
        ).raise_for_status()

    with User(user1, password1, monkeypatch):
        requests.delete(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/model-definitions/delete",
            json={"model_definition_id": primary_model_def_id},
            auth=(user1, password1),
        ).raise_for_status()
        requests.delete(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/model-definitions/delete",
            json={"model_definition_id": fallback_model_def_id},
            auth=(user1, password1),
        ).raise_for_status()
        requests.delete(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/secrets/delete",
            json={"secret_id": secret_id},
            auth=(user1, password1),
        ).raise_for_status()


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_prompt_optimization_job_search_permissions(client, monkeypatch):
    user1, password1 = create_user(client.tracking_uri)
    user2, password2 = create_user(client.tracking_uri)

    # user1 creates an experiment. With default_permission=NO_PERMISSIONS,
    # user2 has no access to it until an explicit grant is created below.
    with User(user1, password1, monkeypatch):
        experiment_id = client.create_experiment("prompt_optimization_search_test")

    # user1 can search jobs in the experiment
    response = requests.post(
        url=client.tracking_uri + "/api/3.0/mlflow/prompt-optimization/jobs/search",
        json={"experiment_id": experiment_id},
        auth=(user1, password1),
    )
    assert response.status_code != 403

    # user2 cannot search jobs in the experiment (no grant + default deny)
    response = requests.post(
        url=client.tracking_uri + "/api/3.0/mlflow/prompt-optimization/jobs/search",
        json={"experiment_id": experiment_id},
        auth=(user2, password2),
    )
    assert response.status_code == 403

    # Grant READ permission to user2
    grant_role_permission(
        client.tracking_uri,
        user2,
        "experiment",
        experiment_id,
        "READ",
    )

    # user2 can now search jobs (READ grants can_read)
    response = requests.post(
        url=client.tracking_uri + "/api/3.0/mlflow/prompt-optimization/jobs/search",
        json={"experiment_id": experiment_id},
        auth=(user2, password2),
    )
    assert response.status_code != 403


def test_prompt_optimization_job_create_permissions(client, monkeypatch):
    user1, password1 = create_user(client.tracking_uri)
    user2, password2 = create_user(client.tracking_uri)

    # user1 creates an experiment
    with User(user1, password1, monkeypatch):
        experiment_id = client.create_experiment("prompt_optimization_create_test")

    # Grant READ permission to user2 (not enough for create)
    grant_role_permission(
        client.tracking_uri,
        user2,
        "experiment",
        experiment_id,
        "READ",
    )

    # user2 cannot create jobs (READ doesn't grant update)
    response = requests.post(
        url=client.tracking_uri + "/api/3.0/mlflow/prompt-optimization/jobs",
        json={
            "experiment_id": experiment_id,
            "source_prompt_uri": "prompts:/test/1",
            "config": {
                "optimizer_type": 1,  # GEPA
                "dataset_id": "test-dataset",
                "scorers": ["Correctness"],
            },
        },
        auth=(user2, password2),
    )
    assert response.status_code == 403

    # Grant EDIT permission to user2
    grant_role_permission(
        client.tracking_uri,
        user2,
        "experiment",
        experiment_id,
        "EDIT",
    )

    # user2 can now create jobs (EDIT grants can_update)
    # The request will fail for other reasons (missing prompt, dataset, etc.)
    # but should pass the permission check
    response = requests.post(
        url=client.tracking_uri + "/api/3.0/mlflow/prompt-optimization/jobs",
        json={
            "experiment_id": experiment_id,
            "source_prompt_uri": "prompts:/test/1",
            "config": {
                "optimizer_type": 1,  # GEPA
                "dataset_id": "test-dataset",
                "scorers": ["Correctness"],
            },
        },
        auth=(user2, password2),
    )
    # Should not be 403 (permission denied)
    assert response.status_code != 403


def test_gateway_endpoint_invocation_requires_use_permission(fastapi_client, monkeypatch):
    user1, password1 = create_user(fastapi_client.tracking_uri)
    user2, password2 = create_user(fastapi_client.tracking_uri)

    # User1 creates a secret, model definition, and endpoint
    with User(user1, password1, monkeypatch):
        response = requests.post(
            url=fastapi_client.tracking_uri + "/api/3.0/mlflow/gateway/secrets/create",
            json={
                "secret_name": "user1_secret",
                "secret_value": {"api_key": "test-key"},
                "provider": "openai",
            },
            auth=(user1, password1),
        )
        response.raise_for_status()
        secret_id = response.json()["secret"]["secret_id"]

        response = requests.post(
            url=fastapi_client.tracking_uri + "/api/3.0/mlflow/gateway/model-definitions/create",
            json={
                "name": "test_model_def",
                "secret_id": secret_id,
                "provider": "openai",
                "model_name": "gpt-4",
            },
            auth=(user1, password1),
        )
        response.raise_for_status()
        model_def_id = response.json()["model_definition"]["model_definition_id"]

        response = requests.post(
            url=fastapi_client.tracking_uri + "/api/3.0/mlflow/gateway/endpoints/create",
            json={
                "name": "test_endpoint",
                "model_configs": [
                    {
                        "model_definition_id": model_def_id,
                        "linkage_type": "PRIMARY",
                    }
                ],
            },
            auth=(user1, password1),
        )
        response.raise_for_status()
        endpoint_id = response.json()["endpoint"]["endpoint_id"]

    # User2 cannot invoke the endpoint (no permission)
    with User(user2, password2, monkeypatch):
        response = requests.post(
            url=fastapi_client.tracking_uri + "/gateway/test_endpoint/mlflow/invocations",
            json={"messages": [{"role": "user", "content": "Hello"}]},
            auth=(user2, password2),
        )
        assert response.status_code == 403

    # Grant READ permission to user2 (not enough for invocation)
    with User(user1, password1, monkeypatch):
        grant_role_permission(
            fastapi_client.tracking_uri,
            user2,
            "gateway_endpoint",
            endpoint_id,
            "READ",
        )

    # User2 still cannot invoke (READ is not sufficient)
    with User(user2, password2, monkeypatch):
        response = requests.post(
            url=fastapi_client.tracking_uri + "/gateway/test_endpoint/mlflow/invocations",
            json={"messages": [{"role": "user", "content": "Hello"}]},
            auth=(user2, password2),
        )
        assert response.status_code == 403

    # Upgrade to USE permission
    with User(user1, password1, monkeypatch):
        grant_role_permission(
            fastapi_client.tracking_uri,
            user2,
            "gateway_endpoint",
            endpoint_id,
            "USE",
        )

    # User2 can now invoke the endpoint (though it will fail due to invalid API key)
    # We just check that we get past the permission check (403) to a different error
    with User(user2, password2, monkeypatch):
        response = requests.post(
            url=fastapi_client.tracking_uri + "/gateway/test_endpoint/mlflow/invocations",
            json={"messages": [{"role": "user", "content": "Hello"}]},
            auth=(user2, password2),
        )
        # Should not be 403 anymore (permission granted)
        # Will likely be 400 or 500 due to invalid API key, but that's fine
        assert response.status_code != 403

    # Cleanup
    with User(user1, password1, monkeypatch):
        requests.delete(
            url=fastapi_client.tracking_uri + "/api/3.0/mlflow/gateway/endpoints/delete",
            json={"endpoint_id": endpoint_id},
            auth=(user1, password1),
        ).raise_for_status()
        requests.delete(
            url=fastapi_client.tracking_uri + "/api/3.0/mlflow/gateway/model-definitions/delete",
            json={"model_definition_id": model_def_id},
            auth=(user1, password1),
        ).raise_for_status()
        requests.delete(
            url=fastapi_client.tracking_uri + "/api/3.0/mlflow/gateway/secrets/delete",
            json={"secret_id": secret_id},
            auth=(user1, password1),
        ).raise_for_status()


def test_otel_unauthenticated_access_denied(fastapi_client, monkeypatch):
    monkeypatch.delenv(MLFLOW_TRACKING_USERNAME.name, raising=False)
    monkeypatch.delenv(MLFLOW_TRACKING_PASSWORD.name, raising=False)

    response = requests.post(
        url=fastapi_client.tracking_uri + "/v1/traces",
        headers={
            "Content-Type": "application/x-protobuf",
            "X-Mlflow-Experiment-Id": "1",
        },
        data=b"",
    )
    assert response.status_code == 401


def test_otel_experiment_permission(fastapi_client, monkeypatch):
    user1, password1 = create_user(fastapi_client.tracking_uri)
    user2, password2 = create_user(fastapi_client.tracking_uri)

    # user1 creates an experiment
    with User(user1, password1, monkeypatch):
        experiment_id = fastapi_client.create_experiment("otel_permission_test")

    # Grant READ permission to user2 (not enough for writing traces)
    grant_role_permission(
        fastapi_client.tracking_uri,
        user2,
        "experiment",
        experiment_id,
        "READ",
    )

    # user2 cannot write traces (READ doesn't grant can_update)
    response = requests.post(
        url=fastapi_client.tracking_uri + "/v1/traces",
        headers={
            "Content-Type": "application/x-protobuf",
            "X-Mlflow-Experiment-Id": experiment_id,
        },
        data=b"",
        auth=(user2, password2),
    )
    assert response.status_code == 403

    # Grant EDIT permission to user2
    grant_role_permission(
        fastapi_client.tracking_uri,
        user2,
        "experiment",
        experiment_id,
        "EDIT",
    )

    # user2 can now write traces (EDIT grants can_update)
    # The request may fail for other reasons (invalid protobuf) but should pass permission check
    response = requests.post(
        url=fastapi_client.tracking_uri + "/v1/traces",
        headers={
            "Content-Type": "application/x-protobuf",
            "X-Mlflow-Experiment-Id": experiment_id,
        },
        data=b"",
        auth=(user2, password2),
    )
    assert response.status_code != 403


def test_job_api_unauthenticated_access_denied(fastapi_client, monkeypatch):
    monkeypatch.delenv(MLFLOW_TRACKING_USERNAME.name, raising=False)
    monkeypatch.delenv(MLFLOW_TRACKING_PASSWORD.name, raising=False)

    response = requests.post(
        url=fastapi_client.tracking_uri + "/ajax-api/3.0/jobs/search",
        json={},
    )
    assert response.status_code == 401


def test_assistant_unauthenticated_access_denied(fastapi_client, monkeypatch):
    monkeypatch.delenv(MLFLOW_TRACKING_USERNAME.name, raising=False)
    monkeypatch.delenv(MLFLOW_TRACKING_PASSWORD.name, raising=False)

    response = requests.post(
        url=fastapi_client.tracking_uri + "/ajax-api/3.0/mlflow/assistant/chat",
        json={"messages": []},
    )
    assert response.status_code == 401


def test_get_online_scoring_configs_with_auth(client, monkeypatch):
    username, password = create_user(client.tracking_uri)

    with User(username, password, monkeypatch):
        experiment_id = client.create_experiment("test_experiment")

        scorer_json = '{"name": "test_scorer", "type": "pyfunc"}'
        response = _send_rest_tracking_post_request(
            client.tracking_uri,
            "/api/3.0/mlflow/scorers/register",
            json_payload={
                "experiment_id": experiment_id,
                "name": "test_scorer",
                "serialized_scorer": scorer_json,
            },
            auth=(username, password),
        )
        scorer_id = response.json()["scorer_id"]

        response = requests.get(
            url=client.tracking_uri + "/ajax-api/3.0/mlflow/scorers/online-configs",
            params={"scorer_ids": scorer_id},
            auth=(username, password),
        )

        assert response.status_code == 200
        data = response.json()
        assert "configs" in data
        assert isinstance(data["configs"], list)


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_online_scoring_config_endpoints_reject_unauthorized_user(client, monkeypatch):
    owner_user, owner_pw = create_user(client.tracking_uri)
    attacker_user, attacker_pw = create_user(client.tracking_uri)

    with User(owner_user, owner_pw, monkeypatch):
        experiment_id = client.create_experiment("online_scoring_auth_exp")

        scorer_json = '{"name": "target_scorer", "type": "pyfunc"}'
        register_resp = _send_rest_tracking_post_request(
            client.tracking_uri,
            "/api/3.0/mlflow/scorers/register",
            json_payload={
                "experiment_id": experiment_id,
                "name": "target_scorer",
                "serialized_scorer": scorer_json,
            },
            auth=(owner_user, owner_pw),
        )
        scorer_id = register_resp.json()["scorer_id"]

        # Under no_permission_auth.ini the default permission is NO_PERMISSIONS, so the
        # attacker (who is never granted access to this experiment) is unauthorized by
        # default. The owner auto-receives MANAGE on the experiment they create.

        # Seed a config so validate_can_read_online_scoring_configs has a row
        # to resolve ownership against (empty results short circuit to allow).
        # sample_rate=0.0 skips the handler's gateway model check on the scorer.
        seed_resp = requests.put(
            url=client.tracking_uri + "/api/3.0/mlflow/scorers/online-config",
            json={
                "experiment_id": experiment_id,
                "name": "target_scorer",
                "sample_rate": 0.0,
            },
            auth=(owner_user, owner_pw),
        )
        assert seed_resp.status_code == 200

    for path in (
        "/api/3.0/mlflow/scorers/online-configs",
        "/ajax-api/3.0/mlflow/scorers/online-configs",
    ):
        response = requests.get(
            url=client.tracking_uri + path,
            params={"scorer_ids": scorer_id},
            auth=(attacker_user, attacker_pw),
        )
        assert response.status_code == 403

    for path in (
        "/api/3.0/mlflow/scorers/online-config",
        "/ajax-api/3.0/mlflow/scorers/online-config",
    ):
        response = requests.put(
            url=client.tracking_uri + path,
            json={
                "experiment_id": experiment_id,
                "name": "target_scorer",
                "sample_rate": 0.0,
            },
            auth=(attacker_user, attacker_pw),
        )
        assert response.status_code == 403

    with User(owner_user, owner_pw, monkeypatch):
        response = requests.get(
            url=client.tracking_uri + "/api/3.0/mlflow/scorers/online-configs",
            params={"scorer_ids": scorer_id},
            auth=(owner_user, owner_pw),
        )
        assert response.status_code == 200

        response = requests.put(
            url=client.tracking_uri + "/api/3.0/mlflow/scorers/online-config",
            json={
                "experiment_id": experiment_id,
                "name": "target_scorer",
                "sample_rate": 0.0,
            },
            auth=(owner_user, owner_pw),
        )
        assert response.status_code == 200


def test_list_users(client):
    username1, password1 = create_user(client.tracking_uri)
    username2, _password2 = create_user(client.tracking_uri)

    # Admin can list all users
    response = requests.get(
        url=client.tracking_uri + LIST_USERS,
        auth=(ADMIN_USERNAME, ADMIN_PASSWORD),
    )
    assert response.status_code == 200
    data = response.json()
    assert "users" in data
    usernames = [u["username"] for u in data["users"]]
    assert ADMIN_USERNAME in usernames
    assert username1 in usernames
    assert username2 in usernames
    for user in data["users"]:
        assert "id" in user
        assert "username" in user
        assert "password" not in user
        assert "password_hash" not in user

    # Unauthenticated request should fail
    response = requests.get(url=client.tracking_uri + LIST_USERS)
    assert response.status_code == 401

    # Any authenticated user may list users (the review-queue assignment UI
    # needs the roster); assigning a reviewer still requires elevated permission.
    response = requests.get(
        url=client.tracking_uri + LIST_USERS,
        auth=(username1, password1),
    )
    assert response.status_code == 200
    assert username1 in [u["username"] for u in response.json()["users"]]

    # Ajax API path should also work for admin
    response = requests.get(
        url=client.tracking_uri + AJAX_LIST_USERS,
        auth=(ADMIN_USERNAME, ADMIN_PASSWORD),
    )
    assert response.status_code == 200
    data = response.json()
    assert "users" in data
    assert len(data["users"]) >= 3


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY": Fernet.generate_key().decode("utf-8")}],
    indirect=True,
)
def test_webhook_admin_only_permissions(client, monkeypatch):
    user1, password1 = create_user(client.tracking_uri)

    # Non-admin: create webhook should be forbidden
    with User(user1, password1, monkeypatch):
        response = requests.post(
            url=client.tracking_uri + "/api/2.0/mlflow/webhooks",
            json={
                "name": "test-webhook",
                "url": "https://example.com/webhook",
                "events": [{"entity": "MODEL_VERSION", "action": "CREATED"}],
            },
            auth=(user1, password1),
        )
        assert response.status_code == 403

    # Non-admin: list webhooks should be forbidden
    with User(user1, password1, monkeypatch):
        response = requests.get(
            url=client.tracking_uri + "/api/2.0/mlflow/webhooks",
            auth=(user1, password1),
        )
        assert response.status_code == 403

    # Admin: create webhook should succeed
    response = requests.post(
        url=client.tracking_uri + "/api/2.0/mlflow/webhooks",
        json={
            "name": "admin-webhook",
            "url": "https://example.com/webhook",
            "events": [{"entity": "MODEL_VERSION", "action": "CREATED"}],
        },
        auth=(ADMIN_USERNAME, ADMIN_PASSWORD),
    )
    response.raise_for_status()
    webhook_id = response.json()["webhook"]["webhook_id"]

    # Admin: list webhooks should succeed
    response = requests.get(
        url=client.tracking_uri + "/api/2.0/mlflow/webhooks",
        auth=(ADMIN_USERNAME, ADMIN_PASSWORD),
    )
    response.raise_for_status()

    # Non-admin: get webhook should be forbidden
    with User(user1, password1, monkeypatch):
        response = requests.get(
            url=client.tracking_uri + f"/api/2.0/mlflow/webhooks/{webhook_id}",
            auth=(user1, password1),
        )
        assert response.status_code == 403

    # Admin: get webhook should succeed
    response = requests.get(
        url=client.tracking_uri + f"/api/2.0/mlflow/webhooks/{webhook_id}",
        auth=(ADMIN_USERNAME, ADMIN_PASSWORD),
    )
    response.raise_for_status()

    # Non-admin: update webhook should be forbidden
    with User(user1, password1, monkeypatch):
        response = requests.patch(
            url=client.tracking_uri + f"/api/2.0/mlflow/webhooks/{webhook_id}",
            json={"name": "updated-name"},
            auth=(user1, password1),
        )
        assert response.status_code == 403

    # Admin: update webhook should succeed
    response = requests.patch(
        url=client.tracking_uri + f"/api/2.0/mlflow/webhooks/{webhook_id}",
        json={"name": "updated-name"},
        auth=(ADMIN_USERNAME, ADMIN_PASSWORD),
    )
    response.raise_for_status()

    # Non-admin: test webhook should be forbidden
    with User(user1, password1, monkeypatch):
        response = requests.post(
            url=client.tracking_uri + f"/api/2.0/mlflow/webhooks/{webhook_id}/test",
            json={},
            auth=(user1, password1),
        )
        assert response.status_code == 403

    # Admin: test webhook should succeed
    response = requests.post(
        url=client.tracking_uri + f"/api/2.0/mlflow/webhooks/{webhook_id}/test",
        json={},
        auth=(ADMIN_USERNAME, ADMIN_PASSWORD),
    )
    response.raise_for_status()

    # Non-admin: delete webhook should be forbidden
    with User(user1, password1, monkeypatch):
        response = requests.delete(
            url=client.tracking_uri + f"/api/2.0/mlflow/webhooks/{webhook_id}",
            auth=(user1, password1),
        )
        assert response.status_code == 403

    # Admin: delete webhook should succeed
    response = requests.delete(
        url=client.tracking_uri + f"/api/2.0/mlflow/webhooks/{webhook_id}",
        auth=(ADMIN_USERNAME, ADMIN_PASSWORD),
    )
    response.raise_for_status()


# -- Unit tests for _authenticate_fastapi_request --


@pytest.fixture
def mock_auth_store():
    if auth_module._USER_AUTH_CACHE is not None:
        with auth_module._USER_AUTH_CACHE_LOCK:
            auth_module._USER_AUTH_CACHE.clear()
    with mock.patch("mlflow.server.auth.store") as mock_store:
        mock_store.get_user.side_effect = lambda username: mock.Mock(username=username)
        mock_store.authenticate_user.return_value = True
        yield mock_store
    if auth_module._USER_AUTH_CACHE is not None:
        with auth_module._USER_AUTH_CACHE_LOCK:
            auth_module._USER_AUTH_CACHE.clear()


@pytest.fixture
def mock_auth_config():
    with mock.patch("mlflow.server.auth.auth_config") as mock_config:
        mock_config.admin_username = "admin"
        yield mock_config


@pytest.fixture
def enable_auth_cache():
    # The credential cache is disabled by default; cache-behavior tests must opt in.
    cache = TTLCache(maxsize=10000, ttl=60)
    with mock.patch("mlflow.server.auth._USER_AUTH_CACHE", cache):
        yield cache


def _make_request(path, authorization=None, mlflow_authorization=None, *, scope_path=None):
    request = mock.Mock()
    request.scope = {"path": scope_path or path}
    request.url.path = path
    request.headers = {}
    if authorization:
        request.headers["Authorization"] = authorization
    if mlflow_authorization:
        request.headers["X-MLflow-Authorization"] = mlflow_authorization
    return request


# -- Basic auth with internal token (trusted internal requests) --


def test_get_fastapi_request_path_prefers_scope_path():
    request = _make_request("/reconstructed/path", scope_path="/routed/path")

    assert get_routed_asgi_path(request) == "/routed/path"


@pytest.mark.parametrize("scope", [None, {}, {"path": ""}, {"path": 123}])
def test_get_fastapi_request_path_falls_back_to_url(scope):
    request = _make_request("/reconstructed/path")
    request.scope = scope

    assert get_routed_asgi_path(request) == "/reconstructed/path"


def test_basic_auth_with_internal_token_returns_user(
    mock_auth_store, mock_auth_config, monkeypatch
):
    monkeypatch.setenv(_MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN.name, "internal-secret")
    credentials = base64.b64encode(b"alice:internal-secret").decode("ascii")
    request = _make_request("/gateway/mlflow/v1/chat", f"Basic {credentials}")

    user = _authenticate_fastapi_request(request)

    assert user.username == "alice"
    mock_auth_store.get_user.assert_called_once_with("alice")
    mock_auth_store.authenticate_user.assert_not_called()


def test_basic_auth_with_internal_token_deleted_user_returns_none(
    mock_auth_store, mock_auth_config, monkeypatch
):
    monkeypatch.setenv(_MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN.name, "internal-secret")
    mock_auth_store.get_user.side_effect = MlflowException("User not found")
    credentials = base64.b64encode(b"deleted_user:internal-secret").decode("ascii")
    request = _make_request("/gateway/mlflow/v1/chat", f"Basic {credentials}")

    user = _authenticate_fastapi_request(request)

    assert user is None


def test_basic_auth_with_internal_token_uses_scope_path(
    mock_auth_store, mock_auth_config, monkeypatch
):
    monkeypatch.setenv(_MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN.name, "internal-secret")
    credentials = base64.b64encode(b"alice:internal-secret").decode("ascii")
    request = _make_request(
        "/gateway/mlflow/v1/chat",
        f"Basic {credentials}",
        scope_path="/api/3.0/mlflow/experiments/list",
    )

    user = _authenticate_fastapi_request(request)

    assert user.username == "alice"
    mock_auth_store.authenticate_user.assert_called_once_with("alice", "internal-secret")
    mock_auth_store.get_user.assert_called_once_with("alice")


@pytest.mark.parametrize(
    "fastapi_client",
    [{"MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE": "true"}],
    indirect=True,
)
def test_malformed_host_does_not_skip_fastapi_auth(fastapi_client, monkeypatch):
    monkeypatch.delenv(MLFLOW_TRACKING_USERNAME.name, raising=False)
    monkeypatch.delenv(MLFLOW_TRACKING_PASSWORD.name, raising=False)

    response = requests.post(
        url=fastapi_client.tracking_uri + "/ajax-api/3.0/jobs/search",
        headers={"Host": "example.com/health?x="},
        json={},
    )

    assert response.status_code == 401


def test_basic_auth_with_wrong_password_falls_through_to_authenticate(
    mock_auth_store, mock_auth_config, monkeypatch
):
    monkeypatch.setenv(_MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN.name, "internal-secret")
    credentials = base64.b64encode(b"alice:wrong-password").decode("ascii")
    request = _make_request("/gateway/mlflow/v1/chat", f"Basic {credentials}")

    user = _authenticate_fastapi_request(request)

    assert user.username == "alice"
    mock_auth_store.authenticate_user.assert_called_once_with("alice", "wrong-password")


def test_basic_auth_internal_token_rejected_on_non_gateway_route(
    mock_auth_store, mock_auth_config, monkeypatch
):
    monkeypatch.setenv(_MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN.name, "internal-secret")
    credentials = base64.b64encode(b"alice:internal-secret").decode("ascii")
    request = _make_request("/api/3.0/mlflow/experiments/list", f"Basic {credentials}")

    _authenticate_fastapi_request(request)

    # Internal token should NOT be accepted on non-gateway routes — falls through
    # to store.authenticate_user instead
    mock_auth_store.authenticate_user.assert_called_once_with("alice", "internal-secret")


def test_basic_auth_no_internal_token_uses_normal_auth(
    mock_auth_store, mock_auth_config, monkeypatch
):
    monkeypatch.delenv(_MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN.name, raising=False)
    credentials = base64.b64encode(b"alice:password123").decode("ascii")
    request = _make_request("/gateway/mlflow/v1/chat", f"Basic {credentials}")

    user = _authenticate_fastapi_request(request)

    assert user.username == "alice"
    mock_auth_store.authenticate_user.assert_called_once_with("alice", "password123")


# -- X-MLflow-Authorization header for gateway routes (OpenAI-protocol coding agents) --


def test_gateway_auth_header_authenticates(mock_auth_store, mock_auth_config, monkeypatch):
    monkeypatch.delenv(_MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN.name, raising=False)
    credentials = base64.b64encode(b"alice:password123").decode("ascii")
    request = _make_request(
        "/gateway/proxy/my-endpoint/v1/responses",
        mlflow_authorization=f"Basic {credentials}",
    )

    user = _authenticate_fastapi_request(request)

    assert user.username == "alice"
    mock_auth_store.authenticate_user.assert_called_once_with("alice", "password123")


def test_gateway_auth_header_takes_precedence_over_bearer(
    mock_auth_store, mock_auth_config, monkeypatch
):
    monkeypatch.delenv(_MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN.name, raising=False)
    credentials = base64.b64encode(b"alice:password123").decode("ascii")
    request = _make_request(
        "/gateway/proxy/my-endpoint/v1/responses",
        authorization="Bearer sk-provider-key",
        mlflow_authorization=f"Basic {credentials}",
    )

    user = _authenticate_fastapi_request(request)

    assert user.username == "alice"
    mock_auth_store.authenticate_user.assert_called_once_with("alice", "password123")


def test_gateway_auth_header_ignored_on_non_gateway_route(
    mock_auth_store, mock_auth_config, monkeypatch
):
    monkeypatch.delenv(_MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN.name, raising=False)
    credentials = base64.b64encode(b"alice:password123").decode("ascii")
    request = _make_request(
        "/api/3.0/mlflow/experiments/list",
        authorization="Bearer sk-provider-key",
        mlflow_authorization=f"Basic {credentials}",
    )

    user = _authenticate_fastapi_request(request)

    assert user is None
    mock_auth_store.authenticate_user.assert_not_called()


def test_gateway_basic_auth_still_works_without_new_header(
    mock_auth_store, mock_auth_config, monkeypatch
):
    monkeypatch.delenv(_MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN.name, raising=False)
    credentials = base64.b64encode(b"alice:password123").decode("ascii")
    request = _make_request(
        "/gateway/proxy/my-endpoint/v1/responses",
        authorization=f"Basic {credentials}",
    )

    user = _authenticate_fastapi_request(request)

    assert user.username == "alice"
    mock_auth_store.authenticate_user.assert_called_once_with("alice", "password123")


def test_gateway_auth_header_honors_internal_token(mock_auth_store, mock_auth_config, monkeypatch):
    monkeypatch.setenv(_MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN.name, "internal-secret")
    credentials = base64.b64encode(b"alice:internal-secret").decode("ascii")
    request = _make_request(
        "/gateway/proxy/my-endpoint/v1/responses",
        mlflow_authorization=f"Basic {credentials}",
    )

    user = _authenticate_fastapi_request(request)

    assert user.username == "alice"
    mock_auth_store.get_user.assert_called_once_with("alice")
    mock_auth_store.authenticate_user.assert_not_called()


def test_gateway_auth_header_malformed_returns_none(mock_auth_store, mock_auth_config):
    request = _make_request(
        "/gateway/proxy/my-endpoint/v1/responses",
        mlflow_authorization="garbage-not-basic",
    )

    user = _authenticate_fastapi_request(request)

    assert user is None


def test_gateway_empty_auth_header_falls_back_to_authorization(
    mock_auth_store, mock_auth_config, monkeypatch
):
    monkeypatch.delenv(_MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN.name, raising=False)
    credentials = base64.b64encode(b"alice:password123").decode("ascii")
    request = _make_request(
        "/gateway/proxy/my-endpoint/v1/responses",
        authorization=f"Basic {credentials}",
    )
    # A present-but-empty X-MLflow-Authorization must not shadow a valid Authorization.
    request.headers["X-MLflow-Authorization"] = ""

    user = _authenticate_fastapi_request(request)

    assert user.username == "alice"
    mock_auth_store.authenticate_user.assert_called_once_with("alice", "password123")


# -- Standard Basic auth --


def test_fastapi_valid_basic_auth(mock_auth_store, mock_auth_config, monkeypatch):
    monkeypatch.delenv(_MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN.name, raising=False)
    credentials = base64.b64encode(b"alice:password123").decode("ascii")
    request = _make_request("/api/3.0/mlflow/experiments/list", f"Basic {credentials}")

    user = _authenticate_fastapi_request(request)

    assert user.username == "alice"
    mock_auth_store.authenticate_user.assert_called_once_with("alice", "password123")


def test_fastapi_invalid_basic_auth(mock_auth_store, mock_auth_config, monkeypatch):
    monkeypatch.delenv(_MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN.name, raising=False)
    mock_auth_store.authenticate_user.return_value = False
    credentials = base64.b64encode(b"alice:wrong").decode("ascii")
    request = _make_request("/api/3.0/mlflow/experiments/list", f"Basic {credentials}")

    user = _authenticate_fastapi_request(request)

    assert user is None


# -- Non-Basic auth schemes --


def test_bearer_returns_none(mock_auth_store, mock_auth_config, monkeypatch):
    monkeypatch.setenv(_MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN.name, "abc123")
    request = _make_request("/gateway/mlflow/v1/chat", "Bearer abc123")

    user = _authenticate_fastapi_request(request)

    assert user is None
    mock_auth_store.get_user.assert_not_called()


# -- No auth header --


def test_fastapi_no_authorization_header(mock_auth_store, mock_auth_config):
    request = _make_request("/api/3.0/mlflow/experiments/list")

    user = _authenticate_fastapi_request(request)

    assert user is None


def test_fastapi_malformed_authorization_header(mock_auth_store, mock_auth_config):
    request = _make_request("/api/3.0/mlflow/experiments/list", "garbage")

    user = _authenticate_fastapi_request(request)

    assert user is None


# -- Basic auth credential cache --


def test_basic_auth_caches_successful_credentials(
    enable_auth_cache, mock_auth_store, mock_auth_config, monkeypatch
):
    monkeypatch.delenv(_MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN.name, raising=False)
    credentials = base64.b64encode(b"alice:password123").decode("ascii")
    request = _make_request("/api/3.0/mlflow/experiments/list", f"Basic {credentials}")

    user_a = _authenticate_fastapi_request(request)
    user_b = _authenticate_fastapi_request(request)

    assert user_a.username == "alice"
    assert user_b.username == "alice"
    # Both PBKDF2 check and user fetch should run exactly once across the two requests.
    mock_auth_store.authenticate_user.assert_called_once_with("alice", "password123")
    mock_auth_store.get_user.assert_called_once_with("alice")


def test_basic_auth_cache_does_not_store_failed_credentials(
    enable_auth_cache, mock_auth_store, mock_auth_config, monkeypatch
):
    monkeypatch.delenv(_MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN.name, raising=False)
    mock_auth_store.authenticate_user.return_value = False
    credentials = base64.b64encode(b"alice:wrong").decode("ascii")
    request = _make_request("/api/3.0/mlflow/experiments/list", f"Basic {credentials}")

    assert _authenticate_fastapi_request(request) is None
    assert _authenticate_fastapi_request(request) is None
    assert mock_auth_store.authenticate_user.call_count == 2


def test_basic_auth_cache_keyed_by_username_and_password(
    enable_auth_cache, mock_auth_store, mock_auth_config, monkeypatch
):
    monkeypatch.delenv(_MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN.name, raising=False)
    alice = base64.b64encode(b"alice:password123").decode("ascii")
    bob = base64.b64encode(b"bob:password123").decode("ascii")
    alice_wrong = base64.b64encode(b"alice:other-password").decode("ascii")

    _authenticate_fastapi_request(_make_request("/x", f"Basic {alice}"))
    _authenticate_fastapi_request(_make_request("/x", f"Basic {bob}"))
    _authenticate_fastapi_request(_make_request("/x", f"Basic {alice_wrong}"))

    assert mock_auth_store.authenticate_user.call_args_list == [
        mock.call("alice", "password123"),
        mock.call("bob", "password123"),
        mock.call("alice", "other-password"),
    ]


def test_basic_auth_returns_none_when_user_deleted_between_authenticate_and_get(
    enable_auth_cache, mock_auth_store, mock_auth_config, monkeypatch
):
    # TOCTOU: authenticate_user returned True but the user disappeared before get_user.
    monkeypatch.delenv(_MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN.name, raising=False)
    mock_auth_store.get_user.side_effect = MlflowException("User not found")
    credentials = base64.b64encode(b"ghost:password123").decode("ascii")
    request = _make_request("/x", f"Basic {credentials}")

    # Flask and FastAPI paths both must treat this as an auth failure, not surface
    # a 500 and, critically, must not cache the (ghost, password123) pair.
    assert _authenticate_fastapi_request(request) is None
    if auth_module._USER_AUTH_CACHE is not None:
        assert (
            auth_module._auth_cache_key("ghost", "password123") not in auth_module._USER_AUTH_CACHE
        )


def test_flask_basic_auth_skips_get_user_when_cache_disabled(
    mock_auth_store, mock_auth_config, monkeypatch
):
    monkeypatch.delenv(_MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN.name, raising=False)
    fake_flask_request = mock.Mock()
    fake_flask_request.authorization.username = "alice"
    fake_flask_request.authorization.password = "password123"

    with (
        mock.patch("mlflow.server.auth._USER_AUTH_CACHE", None),
        mock.patch("mlflow.server.auth.request", fake_flask_request),
    ):
        result = auth_module.authenticate_request_basic_auth()

    assert result is fake_flask_request.authorization
    mock_auth_store.authenticate_user.assert_called_once_with("alice", "password123")
    # Cache disabled + Flask path only needs the yes/no answer → no user fetch.
    mock_auth_store.get_user.assert_not_called()


def test_flask_basic_auth_shares_cache_with_fastapi_path(
    enable_auth_cache, mock_auth_store, mock_auth_config, monkeypatch
):
    monkeypatch.delenv(_MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN.name, raising=False)
    # Prime the cache via the FastAPI path.
    credentials = base64.b64encode(b"alice:password123").decode("ascii")
    _authenticate_fastapi_request(_make_request("/x", f"Basic {credentials}"))
    mock_auth_store.authenticate_user.assert_called_once_with("alice", "password123")

    # A subsequent Flask-side call for the same credentials must be served from
    # cache — no second PBKDF2 verification, no second user fetch.
    fake_flask_request = mock.Mock()
    fake_flask_request.authorization.username = "alice"
    fake_flask_request.authorization.password = "password123"
    with mock.patch("mlflow.server.auth.request", fake_flask_request):
        result = auth_module.authenticate_request_basic_auth()

    assert result is fake_flask_request.authorization
    mock_auth_store.authenticate_user.assert_called_once_with("alice", "password123")


def test_invalidate_user_auth_cache_drops_only_matching_username(
    enable_auth_cache, mock_auth_store, mock_auth_config, monkeypatch
):
    monkeypatch.delenv(_MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN.name, raising=False)
    alice = base64.b64encode(b"alice:password123").decode("ascii")
    alice_alt = base64.b64encode(b"alice:other-password").decode("ascii")
    bob = base64.b64encode(b"bob:password123").decode("ascii")

    _authenticate_fastapi_request(_make_request("/x", f"Basic {alice}"))
    _authenticate_fastapi_request(_make_request("/x", f"Basic {alice_alt}"))
    _authenticate_fastapi_request(_make_request("/x", f"Basic {bob}"))
    assert mock_auth_store.authenticate_user.call_count == 3

    auth_module._invalidate_user_auth_cache("alice")

    # Alice's two cached credentials are re-checked; bob's cache entry stays hot.
    _authenticate_fastapi_request(_make_request("/x", f"Basic {alice}"))
    _authenticate_fastapi_request(_make_request("/x", f"Basic {alice_alt}"))
    _authenticate_fastapi_request(_make_request("/x", f"Basic {bob}"))
    assert mock_auth_store.authenticate_user.call_count == 5


def _create_trace(tracking_uri: str, experiment_id: str, auth: tuple[str, str]) -> str:
    """Create a trace and return its request_id."""
    resp = requests.post(
        url=tracking_uri + "/api/2.0/mlflow/traces",
        json={
            "experiment_id": experiment_id,
            "timestamp_ms": int(time.time() * 1000),
            "execution_time_ms": 10,
            "status": "OK",
            "request_metadata": [],
            "tags": [],
        },
        auth=auth,
    )
    resp.raise_for_status()
    return resp.json()["trace_info"]["request_id"]


def _grant_experiment_permission(
    tracking_uri: str,
    experiment_id: str,
    username: str,
    permission: str,
    auth: tuple[str, str],
) -> None:
    # ``grant`` is not upsert — issue a best-effort revoke first so this helper
    # behaves like the legacy upsert semantics tests relied on.
    requests.post(
        url=tracking_uri + "/api/3.0/mlflow/users/permissions/revoke",
        json={
            "username": username,
            "resource_type": "experiment",
            "resource_id": experiment_id,
        },
        auth=auth,
    )
    _send_rest_tracking_post_request(
        tracking_uri,
        "/api/3.0/mlflow/users/permissions/grant",
        json_payload={
            "username": username,
            "resource_type": "experiment",
            "resource_id": experiment_id,
            "permission": permission,
        },
        auth=auth,
    )


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_trace_search_permission(client, monkeypatch):
    user1, password1 = create_user(client.tracking_uri)
    user2, password2 = create_user(client.tracking_uri)

    with User(user1, password1, monkeypatch):
        experiment_id = client.create_experiment("trace_search_test")

    # user2 has no grant; default_permission=NO_PERMISSIONS denies access

    # user1 can search traces
    resp = requests.get(
        url=client.tracking_uri + "/api/2.0/mlflow/traces",
        params={"experiment_ids": [experiment_id]},
        auth=(user1, password1),
    )
    assert resp.status_code == 200

    # user2 is denied
    resp = requests.get(
        url=client.tracking_uri + "/api/2.0/mlflow/traces",
        params={"experiment_ids": [experiment_id]},
        auth=(user2, password2),
    )
    assert resp.status_code == 403

    # Grant READ; user2 can now search
    _grant_experiment_permission(
        client.tracking_uri, experiment_id, user2, "READ", (user1, password1)
    )
    resp = requests.get(
        url=client.tracking_uri + "/api/2.0/mlflow/traces",
        params={"experiment_ids": [experiment_id]},
        auth=(user2, password2),
    )
    assert resp.status_code == 200


def test_trace_delete_permission(client, monkeypatch):
    user1, password1 = create_user(client.tracking_uri)
    user2, password2 = create_user(client.tracking_uri)

    with User(user1, password1, monkeypatch):
        experiment_id = client.create_experiment("trace_delete_test")

    _grant_experiment_permission(
        client.tracking_uri, experiment_id, user2, "READ", (user1, password1)
    )

    def delete_traces(auth):
        return requests.post(
            url=client.tracking_uri + "/api/2.0/mlflow/traces/delete-traces",
            json={
                "experiment_id": experiment_id,
                "max_timestamp_millis": 9999999999999,
            },
            auth=auth,
        )

    # user2 with READ is denied
    assert delete_traces((user2, password2)).status_code == 403

    # user1 (MANAGE) can delete
    assert delete_traces((user1, password1)).status_code == 200

    # Upgrade user2 to MANAGE; now allowed
    _grant_experiment_permission(
        client.tracking_uri, experiment_id, user2, "MANAGE", (user1, password1)
    )
    assert delete_traces((user2, password2)).status_code == 200


def test_trace_tag_permission(client, monkeypatch):
    user1, password1 = create_user(client.tracking_uri)
    user2, password2 = create_user(client.tracking_uri)

    with User(user1, password1, monkeypatch):
        experiment_id = client.create_experiment("trace_tag_test")

    request_id = _create_trace(client.tracking_uri, experiment_id, (user1, password1))

    _grant_experiment_permission(
        client.tracking_uri, experiment_id, user2, "READ", (user1, password1)
    )

    def set_tag(auth):
        return requests.patch(
            url=client.tracking_uri + f"/api/2.0/mlflow/traces/{request_id}/tags",
            json={"key": "env", "value": "test"},
            auth=auth,
        )

    def delete_tag(auth):
        return requests.delete(
            url=client.tracking_uri + f"/api/2.0/mlflow/traces/{request_id}/tags",
            json={"key": "env"},
            auth=auth,
        )

    # READ is not enough for tag mutation
    assert set_tag((user2, password2)).status_code == 403
    assert delete_tag((user2, password2)).status_code == 403

    # Upgrade to EDIT; tag operations now allowed
    _grant_experiment_permission(
        client.tracking_uri, experiment_id, user2, "EDIT", (user1, password1)
    )
    assert set_tag((user2, password2)).status_code == 200
    assert delete_tag((user2, password2)).status_code == 200


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_trace_get_info_permission(client, monkeypatch):
    user1, password1 = create_user(client.tracking_uri)
    user2, password2 = create_user(client.tracking_uri)

    with User(user1, password1, monkeypatch):
        experiment_id = client.create_experiment("trace_get_info_test")

    request_id = _create_trace(client.tracking_uri, experiment_id, (user1, password1))

    # user2 has no grant; default_permission=NO_PERMISSIONS denies access

    def get_info(auth):
        return requests.get(
            url=client.tracking_uri + f"/api/2.0/mlflow/traces/{request_id}/info",
            auth=auth,
        )

    # user2 with no grant is denied
    assert get_info((user2, password2)).status_code == 403

    # user1 can read
    assert get_info((user1, password1)).status_code == 200

    # Grant READ; user2 can now read
    _grant_experiment_permission(
        client.tracking_uri, experiment_id, user2, "READ", (user1, password1)
    )
    assert get_info((user2, password2)).status_code == 200


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_trace_get_v3_permission(client, monkeypatch):
    user1, password1 = create_user(client.tracking_uri)
    user2, password2 = create_user(client.tracking_uri)

    with User(user1, password1, monkeypatch):
        experiment_id = client.create_experiment("trace_get_v3_test")

    trace_id = _create_trace(client.tracking_uri, experiment_id, (user1, password1))

    # user2 has no grant; default_permission=NO_PERMISSIONS denies access

    def get_trace_v3(auth):
        return requests.get(
            url=client.tracking_uri + f"/api/3.0/mlflow/traces/{trace_id}",
            auth=auth,
        )

    assert get_trace_v3((user2, password2)).status_code == 403
    assert get_trace_v3((user1, password1)).status_code == 200

    _grant_experiment_permission(
        client.tracking_uri, experiment_id, user2, "READ", (user1, password1)
    )
    assert get_trace_v3((user2, password2)).status_code == 200


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
@pytest.mark.parametrize("api_version", ["2.0", "3.0"])
def test_trace_artifact_authorization(
    client: MlflowClient, monkeypatch: pytest.MonkeyPatch, api_version: str
):
    user1, password1 = create_user(client.tracking_uri)
    user2, password2 = create_user(client.tracking_uri)

    with User(user1, password1, monkeypatch):
        experiment_id = client.create_experiment(f"trace_artifact_authz_test_v{api_version}")

    request_id = _create_trace(client.tracking_uri, experiment_id, (user1, password1))

    def get_artifact(auth):
        return requests.get(
            url=client.tracking_uri + f"/ajax-api/{api_version}/mlflow/get-trace-artifact",
            params={"request_id": request_id},
            auth=auth,
        )

    # user1 (owner) should be able to access the artifact endpoint (may be 404 if
    # no artifact has been uploaded, but should NOT be 403)
    assert get_artifact((user1, password1)).status_code != 403

    # user2 has no permission on the experiment, expect 403
    assert get_artifact((user2, password2)).status_code == 403

    # Grant READ; user2 can now access the artifact endpoint
    _grant_experiment_permission(
        client.tracking_uri, experiment_id, user2, "READ", (user1, password1)
    )
    assert get_artifact((user2, password2)).status_code != 403


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_trace_batch_get_permission(client, monkeypatch):
    user1, password1 = create_user(client.tracking_uri)
    user2, password2 = create_user(client.tracking_uri)

    with User(user1, password1, monkeypatch):
        experiment_id = client.create_experiment("trace_batch_get_test")

    trace_id = _create_trace(client.tracking_uri, experiment_id, (user1, password1))

    # user2 has no grant; default_permission=NO_PERMISSIONS denies access

    def batch_get(auth):
        return requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/traces/batchGetInfos",
            json={"trace_ids": [trace_id]},
            auth=auth,
        )

    assert batch_get((user2, password2)).status_code == 403
    assert batch_get((user1, password1)).status_code == 200

    _grant_experiment_permission(
        client.tracking_uri, experiment_id, user2, "READ", (user1, password1)
    )
    assert batch_get((user2, password2)).status_code == 200


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_trace_link_to_run_permission(client, monkeypatch):
    user1, password1 = create_user(client.tracking_uri)
    user2, password2 = create_user(client.tracking_uri)

    with User(user1, password1, monkeypatch):
        exp_a = client.create_experiment("link_test_exp_a")
        exp_b = client.create_experiment("link_test_exp_b")

    trace_id = _create_trace(client.tracking_uri, exp_b, (user1, password1))

    with User(user1, password1, monkeypatch):
        run = client.create_run(exp_a)
    run_id = run.info.run_id

    # user2: UPDATE on exp_a but no grant on exp_b → denied (can't read traces in B)
    # default_permission=NO_PERMISSIONS means absence of a grant on exp_b is a deny
    _grant_experiment_permission(client.tracking_uri, exp_a, user2, "EDIT", (user1, password1))

    def link(auth):
        return requests.post(
            url=client.tracking_uri + "/api/2.0/mlflow/traces/link-to-run",
            json={"trace_ids": [trace_id], "run_id": run_id},
            auth=auth,
        )

    assert link((user2, password2)).status_code == 403

    # Grant READ on exp_b → now allowed
    _grant_experiment_permission(client.tracking_uri, exp_b, user2, "READ", (user1, password1))
    assert link((user2, password2)).status_code == 200


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_trace_search_v3_permission(client, monkeypatch):
    user1, password1 = create_user(client.tracking_uri)
    user2, password2 = create_user(client.tracking_uri)

    with User(user1, password1, monkeypatch):
        experiment_id = client.create_experiment("trace_search_v3_test")

    # user2 has no grant; default_permission=NO_PERMISSIONS denies access

    def search_v3(auth):
        return requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/traces/search",
            json={
                "locations": [{"mlflow_experiment": {"experiment_id": experiment_id}}],
            },
            auth=auth,
        )

    assert search_v3((user2, password2)).status_code == 403
    assert search_v3((user1, password1)).status_code == 200

    _grant_experiment_permission(
        client.tracking_uri, experiment_id, user2, "READ", (user1, password1)
    )
    assert search_v3((user2, password2)).status_code == 200


_MCP_AJAX_PREFIX = "/ajax-api/3.0/mlflow/mcp-servers"
_MCP_REST_PREFIX = "/api/3.0/mlflow/mcp-servers"

_MCP_SUBPATHS = [
    "",
    "/com.test/my-server",
    "/my-server/versions",
    "/my-server/versions/1",
    "/my-server/versions/1/tags",
    "/my-server/versions/1/tags/k",
    "/endpoints",
    "/my-server/endpoints",
    "/my-server/endpoints/123",
    "/my-server/tags",
    "/my-server/tags/k",
    "/my-server/aliases",
    "/my-server/aliases/latest",
]


@pytest.mark.parametrize(
    "path",
    [f"{prefix}{sub}" for prefix in (_MCP_AJAX_PREFIX, _MCP_REST_PREFIX) for sub in _MCP_SUBPATHS],
)
def test_mcp_server_routes_have_validators(path):
    validator = _find_fastapi_validator(path)
    assert validator is not None


@pytest.mark.parametrize(
    "path",
    [f"{prefix}{sub}" for prefix in (_MCP_AJAX_PREFIX, _MCP_REST_PREFIX) for sub in _MCP_SUBPATHS],
)
def test_mcp_server_routes_return_validator_with_custom_auth(path):
    with mock.patch("mlflow.server.auth.auth_config") as cfg:
        cfg.authorization_function = "custom_auth:authorize"
        validator = _find_fastapi_validator(path)
    assert validator is not None


@pytest.mark.parametrize("prefix", [_MCP_AJAX_PREFIX, _MCP_REST_PREFIX])
def test_mcp_server_unauthenticated_returns_401(fastapi_client, monkeypatch, prefix):
    monkeypatch.delenv(MLFLOW_TRACKING_USERNAME.name, raising=False)
    monkeypatch.delenv(MLFLOW_TRACKING_PASSWORD.name, raising=False)

    response = requests.get(
        url=fastapi_client.tracking_uri + prefix,
    )
    assert response.status_code == 401


@pytest.mark.parametrize("prefix", [_MCP_AJAX_PREFIX, _MCP_REST_PREFIX])
def test_mcp_server_forbidden_returns_403(fastapi_client, monkeypatch, prefix):
    creator, creator_pw = create_user(fastapi_client.tracking_uri)
    other, other_pw = create_user(fastapi_client.tracking_uri)

    with User(creator, creator_pw, monkeypatch):
        requests.post(
            url=fastapi_client.tracking_uri + prefix,
            json={"name": "com.test/forbidden-server"},
            auth=(creator, creator_pw),
        ).raise_for_status()

    # A different user with default_permission=READ → can_update/can_delete=False
    with User(other, other_pw, monkeypatch):
        response = requests.patch(
            url=fastapi_client.tracking_uri + f"{prefix}/com.test/forbidden-server",
            json={"description": "test"},
            auth=(other, other_pw),
        )
        assert response.status_code == 403

        response = requests.delete(
            url=fastapi_client.tracking_uri + f"{prefix}/com.test/forbidden-server",
            auth=(other, other_pw),
        )
        assert response.status_code == 403


@pytest.mark.parametrize("prefix", [_MCP_AJAX_PREFIX, _MCP_REST_PREFIX])
def test_mcp_server_read_passes_auth(fastapi_client, monkeypatch, prefix):
    username, password = create_user(fastapi_client.tracking_uri)

    with User(username, password, monkeypatch):
        requests.post(
            url=fastapi_client.tracking_uri + prefix,
            json={"name": "com.test/read-server"},
            auth=(username, password),
        ).raise_for_status()

        # No explicit grant; default_permission=READ → can_read=True
        response = requests.get(
            url=fastapi_client.tracking_uri + f"{prefix}/com.test/read-server",
            auth=(username, password),
        )
        assert response.status_code == 200


@pytest.mark.parametrize("prefix", [_MCP_AJAX_PREFIX, _MCP_REST_PREFIX])
def test_mcp_server_edit_passes_with_grant(fastapi_client, monkeypatch, prefix):
    username, password = create_user(fastapi_client.tracking_uri)

    with User(username, password, monkeypatch):
        requests.post(
            url=fastapi_client.tracking_uri + prefix,
            json={"name": "com.test/edit-server"},
            auth=(username, password),
        ).raise_for_status()

    grant_role_permission(
        fastapi_client.tracking_uri,
        username,
        "mcp_server",
        "com.test/edit-server",
        "EDIT",
    )

    with User(username, password, monkeypatch):
        response = requests.patch(
            url=fastapi_client.tracking_uri + f"{prefix}/com.test/edit-server",
            json={"description": "updated via grant"},
            auth=(username, password),
        )
        assert response.status_code == 200


@pytest.mark.parametrize("prefix", [_MCP_AJAX_PREFIX, _MCP_REST_PREFIX])
def test_mcp_server_creator_gets_manage(fastapi_client, monkeypatch, prefix):
    username, password = create_user(fastapi_client.tracking_uri)

    with User(username, password, monkeypatch):
        requests.post(
            url=fastapi_client.tracking_uri + prefix,
            json={"name": "com.test/manage-server"},
            auth=(username, password),
        ).raise_for_status()

        # Creator should have MANAGE → can_update=True
        response = requests.patch(
            url=fastapi_client.tracking_uri + f"{prefix}/com.test/manage-server",
            json={"description": "updated by creator"},
            auth=(username, password),
        )
        assert response.status_code == 200

        # Creator should have MANAGE → can_delete=True
        response = requests.delete(
            url=fastapi_client.tracking_uri + f"{prefix}/com.test/manage-server",
            auth=(username, password),
        )
        assert response.status_code == 200


@pytest.mark.parametrize("prefix", [_MCP_AJAX_PREFIX, _MCP_REST_PREFIX])
def test_mcp_server_delete_cascades_grants(fastapi_client, monkeypatch, prefix):
    creator, creator_pw = create_user(fastapi_client.tracking_uri)
    other, other_pw = create_user(fastapi_client.tracking_uri)

    # Creator creates server → auto-grant gives MANAGE
    with User(creator, creator_pw, monkeypatch):
        requests.post(
            url=fastapi_client.tracking_uri + prefix,
            json={"name": "com.test/cascade-server"},
            auth=(creator, creator_pw),
        ).raise_for_status()

    # Grant MANAGE to other so they can delete
    admin_auth = (ADMIN_USERNAME, ADMIN_PASSWORD)
    requests.post(
        url=f"{fastapi_client.tracking_uri}/api/3.0/mlflow/users/permissions/grant",
        json={
            "username": other,
            "resource_type": "mcp_server",
            "resource_id": "com.test/cascade-server",
            "permission": "MANAGE",
        },
        auth=admin_auth,
    ).raise_for_status()

    # Non-admin deletes the server → should cascade-delete auto-granted permissions
    with User(other, other_pw, monkeypatch):
        requests.delete(
            url=fastapi_client.tracking_uri + f"{prefix}/com.test/cascade-server",
            auth=(other, other_pw),
        ).raise_for_status()

    # Re-create the server as a non-admin
    with User(creator, creator_pw, monkeypatch):
        requests.post(
            url=fastapi_client.tracking_uri + prefix,
            json={"name": "com.test/cascade-server"},
            auth=(creator, creator_pw),
        ).raise_for_status()

    # Other user's MANAGE was cleaned up by cascade → PATCH denied
    with User(other, other_pw, monkeypatch):
        response = requests.patch(
            url=fastapi_client.tracking_uri + f"{prefix}/com.test/cascade-server",
            json={"description": "should fail"},
            auth=(other, other_pw),
        )
        assert response.status_code == 403


@pytest.mark.parametrize("prefix", [_MCP_AJAX_PREFIX, _MCP_REST_PREFIX])
def test_mcp_server_admin_delete_cascades_grants(fastapi_client, monkeypatch, prefix):
    """Admin delete must still run ``_mcp_server_after_delete`` grant cleanup.

    Admins skip FastAPI validators (full access) but must not skip after-request
    handlers — otherwise recreating the same server name restores stale grants.
    """
    creator, creator_pw = create_user(fastapi_client.tracking_uri)
    other, other_pw = create_user(fastapi_client.tracking_uri)
    admin_auth = (ADMIN_USERNAME, ADMIN_PASSWORD)

    with User(creator, creator_pw, monkeypatch):
        requests.post(
            url=fastapi_client.tracking_uri + prefix,
            json={"name": "com.test/admin-cascade-server"},
            auth=(creator, creator_pw),
        ).raise_for_status()

    requests.post(
        url=f"{fastapi_client.tracking_uri}/api/3.0/mlflow/users/permissions/grant",
        json={
            "username": other,
            "resource_type": "mcp_server",
            "resource_id": "com.test/admin-cascade-server",
            "permission": "MANAGE",
        },
        auth=admin_auth,
    ).raise_for_status()

    # Admin deletes the server — after-handler must cascade-delete grants.
    requests.delete(
        url=fastapi_client.tracking_uri + f"{prefix}/com.test/admin-cascade-server",
        auth=admin_auth,
    ).raise_for_status()

    with User(creator, creator_pw, monkeypatch):
        requests.post(
            url=fastapi_client.tracking_uri + prefix,
            json={"name": "com.test/admin-cascade-server"},
            auth=(creator, creator_pw),
        ).raise_for_status()

    with User(other, other_pw, monkeypatch):
        response = requests.patch(
            url=fastapi_client.tracking_uri + f"{prefix}/com.test/admin-cascade-server",
            json={"description": "should fail"},
            auth=(other, other_pw),
        )
        assert response.status_code == 403


@pytest.mark.parametrize("prefix", [_MCP_AJAX_PREFIX, _MCP_REST_PREFIX])
def test_mcp_server_tracks_created_by(fastapi_client, monkeypatch, prefix):
    username, password = create_user(fastapi_client.tracking_uri)

    # Creator should be recorded on create
    with User(username, password, monkeypatch):
        resp = requests.post(
            url=fastapi_client.tracking_uri + prefix,
            json={"name": "com.test/audit-server"},
            auth=(username, password),
        )
        resp.raise_for_status()
        data = resp.json()
        assert data["created_by"] == username
        assert data["last_updated_by"] == username

    # Creator's update should set last_updated_by
    with User(username, password, monkeypatch):
        resp = requests.patch(
            url=fastapi_client.tracking_uri + f"{prefix}/com.test/audit-server",
            json={"description": "user update"},
            auth=(username, password),
        )
        resp.raise_for_status()
        assert resp.json()["last_updated_by"] == username

    # Admin's update should change last_updated_by to admin
    resp = requests.patch(
        url=fastapi_client.tracking_uri + f"{prefix}/com.test/audit-server",
        json={"description": "admin update"},
        auth=(ADMIN_USERNAME, ADMIN_PASSWORD),
    )
    resp.raise_for_status()
    data = resp.json()
    assert data["created_by"] == username
    assert data["last_updated_by"] == ADMIN_USERNAME


@pytest.mark.parametrize("prefix", [_MCP_AJAX_PREFIX, _MCP_REST_PREFIX])
def test_mcp_server_unified_permission_grant(fastapi_client, monkeypatch, prefix):
    admin_auth = (ADMIN_USERNAME, ADMIN_PASSWORD)
    user, password = create_user(fastapi_client.tracking_uri)
    server_name = "com.test/unified-perm"

    requests.post(
        url=fastapi_client.tracking_uri + prefix,
        json={"name": server_name},
        auth=admin_auth,
    ).raise_for_status()

    requests.post(
        url=f"{fastapi_client.tracking_uri}/api/3.0/mlflow/users/permissions/grant",
        json={
            "username": user,
            "resource_type": "mcp_server",
            "resource_id": server_name,
            "permission": "EDIT",
        },
        auth=admin_auth,
    ).raise_for_status()

    resp = requests.get(
        url=f"{fastapi_client.tracking_uri}/api/3.0/mlflow/users/permissions/get",
        params={
            "username": user,
            "resource_type": "mcp_server",
            "resource_id": server_name,
        },
        auth=admin_auth,
    )
    assert resp.status_code == 200
    assert resp.json()["permission"] == "EDIT"

    requests.post(
        url=f"{fastapi_client.tracking_uri}/api/3.0/mlflow/users/permissions/revoke",
        json={
            "username": user,
            "resource_type": "mcp_server",
            "resource_id": server_name,
        },
        auth=admin_auth,
    ).raise_for_status()

    # After revoke, the user falls back to default_permission (READ).
    resp = requests.get(
        url=f"{fastapi_client.tracking_uri}/api/3.0/mlflow/users/permissions/get",
        params={
            "username": user,
            "resource_type": "mcp_server",
            "resource_id": server_name,
        },
        auth=admin_auth,
    )
    assert resp.status_code == 200
    assert resp.json()["permission"] == "READ"


@pytest.mark.parametrize("prefix", [_MCP_AJAX_PREFIX, _MCP_REST_PREFIX])
def test_mcp_server_root_post_enforces_workspace_create_authz(prefix, monkeypatch, tmp_path):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    monkeypatch.setattr(
        auth_module,
        "auth_config",
        auth_module.auth_config._replace(default_permission=NO_PERMISSIONS.name),
    )

    db_uri = f"sqlite:///{tmp_path / 'auth-ws.db'}"
    auth_store = SqlAlchemyStore()
    auth_store.init_db(db_uri)
    monkeypatch.setattr(auth_module, "store", auth_store, raising=False)

    username = "workspace-test-user"
    auth_store.create_user(username, "supersecurepassword", is_admin=False)

    # Without any workspace permission, create must be denied.
    with workspace_context.WorkspaceContext("team-a"):
        assert auth_module.validate_can_create_mcp_server(username) is False

    # Grant workspace-level USE — the same grant that enables
    # validate_can_create_registered_model via _user_can_create_in_workspace.
    auth_store.set_workspace_permission("team-a", username, USE.name)

    with workspace_context.WorkspaceContext("team-a"):
        assert auth_module.validate_can_create_mcp_server(username) is True

    # Revoke the grant and verify denial is restored.
    auth_store.delete_workspace_permission("team-a", username)

    with workspace_context.WorkspaceContext("team-a"):
        assert auth_module.validate_can_create_mcp_server(username) is False

    # The FastAPI validator for root POST dispatches to validate_can_create_mcp_server.
    validator = _find_fastapi_validator(prefix)
    assert validator is not None

    auth_store.engine.dispose()


def test_validate_can_create_mcp_server_delegates_to_shared_helper():
    with mock.patch.object(
        auth_module, "_can_create_in_workspace", return_value=True
    ) as mock_helper:
        result = auth_module.validate_can_create_mcp_server("alice")
        mock_helper.assert_called_once_with("alice")
        assert result is True


@pytest.mark.parametrize(
    ("resource_type", "resource_id"),
    [
        ("mcp_server", "com.test/some-server"),
        ("registered_model", "my-model"),
        ("experiment", "123"),
    ],
)
def test_read_predicate_honors_grant_default_workspace_access(
    monkeypatch, resource_type, resource_id
):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    default_workspace = "team-default"
    monkeypatch.setattr(
        auth_module,
        "auth_config",
        auth_module.auth_config._replace(
            default_permission=READ.name,
            grant_default_workspace_access=True,
        ),
        raising=False,
    )
    monkeypatch.setattr(auth_module, "_get_workspace_store", lambda: None, raising=False)
    monkeypatch.setattr(
        auth_module,
        "get_default_workspace_optional",
        lambda *args, **kwargs: (SimpleNamespace(name=default_workspace), True),
        raising=False,
    )

    class DummyStore:
        def get_user(self, username):
            return SimpleNamespace(id=42, username=username)

        def list_role_grants_for_user_in_workspace(self, user_id, workspace, resource_type):
            return []

    monkeypatch.setattr(auth_module, "store", DummyStore(), raising=False)

    with workspace_context.WorkspaceContext(default_workspace):
        predicate = auth_module._role_based_read_predicate("alice", resource_type)
        assert predicate(resource_id) is True


@pytest.mark.parametrize(
    "path",
    [f"{prefix}/" for prefix in (_MCP_AJAX_PREFIX, _MCP_REST_PREFIX)]
    + [f"{prefix}/endpoints/" for prefix in (_MCP_AJAX_PREFIX, _MCP_REST_PREFIX)],
)
def test_response_filter_matches_trailing_slash(path):
    assert _find_fastapi_response_filter(path, "GET") is not None


@pytest.mark.parametrize("prefix", [_MCP_AJAX_PREFIX, _MCP_REST_PREFIX])
@pytest.mark.parametrize("path_suffix", ["/com.test/server", "/endpoints/123"])
def test_response_filter_requires_exact_collection_route_match(prefix, path_suffix):
    assert _find_fastapi_response_filter(f"{prefix}{path_suffix}", "GET") is None


def test_apply_fastapi_response_filter_fails_closed():
    request = SimpleNamespace(method="GET")
    response = SimpleNamespace(
        status_code=200,
        headers={"content-length": "2", "x-test": "1"},
        media_type="application/json",
    )

    filtered = auth_module._apply_fastapi_response_filter(
        response_filter=lambda *_: (_ for _ in ()).throw(ValueError("boom")),
        username="alice",
        body=b'{"mcp_servers":[]}',
        request=request,
        response=response,
        path=_MCP_REST_PREFIX,
    )

    assert filtered.status_code == 500
    payload = json.loads(filtered.body)
    assert payload["error_code"] == "INTERNAL_ERROR"
    assert "Failed to filter response" in payload["message"]


@pytest.mark.parametrize("prefix", [_MCP_AJAX_PREFIX, _MCP_REST_PREFIX])
@pytest.mark.parametrize("sub", ["endpoints", "tags", "aliases"])
def test_non_version_nested_post_requires_can_update(fastapi_client, monkeypatch, prefix, sub):
    user, pw = create_user(fastapi_client.tracking_uri)

    with User(user, pw, monkeypatch):
        response = requests.post(
            url=f"{fastapi_client.tracking_uri}{prefix}/com.test/no-such-server/{sub}",
            json={},
            auth=(user, pw),
        )
    assert response.status_code == 403


@pytest.mark.parametrize("prefix", [_MCP_AJAX_PREFIX, _MCP_REST_PREFIX])
def test_implicit_parent_create_grants_manage_despite_wildcard(fastapi_client, monkeypatch, prefix):
    user, pw = create_user(fastapi_client.tracking_uri)
    admin_auth = (ADMIN_USERNAME, ADMIN_PASSWORD)
    server_name = "com.test/wildcard-implicit"

    requests.post(
        url=f"{fastapi_client.tracking_uri}/api/3.0/mlflow/users/permissions/grant",
        json={
            "username": user,
            "resource_type": "mcp_server",
            "resource_id": "*",
            "permission": "EDIT",
        },
        auth=admin_auth,
    ).raise_for_status()

    with User(user, pw, monkeypatch):
        requests.post(
            url=f"{fastapi_client.tracking_uri}{prefix}/{server_name}/versions",
            json=_version_create_body(server_name),
            auth=(user, pw),
        ).raise_for_status()

    resp = requests.get(
        url=f"{fastapi_client.tracking_uri}/api/3.0/mlflow/users/permissions/get",
        params={
            "username": user,
            "resource_type": "mcp_server",
            "resource_id": server_name,
        },
        auth=admin_auth,
    )
    assert resp.status_code == 200
    assert resp.json()["permission"] == "MANAGE"


@pytest.mark.parametrize("prefix", [_MCP_AJAX_PREFIX, _MCP_REST_PREFIX])
def test_version_create_validator_stores_live_update_recheck(monkeypatch, prefix):
    validator = _find_fastapi_validator(f"{prefix}/com.test/race-server/versions")
    assert validator is not None

    class _Store:
        exists = False

        def get_mcp_server(self, name):
            if self.exists:
                return SimpleNamespace(name=name)
            raise MlflowException("not found", error_code=RESOURCE_DOES_NOT_EXIST)

    store = _Store()
    permission_helper = mock.Mock(
        side_effect=lambda name, username: SimpleNamespace(
            can_read=False,
            can_update=store.exists,
            can_delete=False,
        )
    )
    monkeypatch.setattr(auth_module, "_get_tracking_store", lambda: store)
    monkeypatch.setattr(auth_module, "_get_mcp_server_permission", permission_helper)
    monkeypatch.setattr(auth_module, "validate_can_create_mcp_server", lambda username: True)

    request = SimpleNamespace(method="POST", state=SimpleNamespace())
    assert asyncio.run(validator("alice", request)) is True
    assert request.state.mcp_server_parent_auto_created is True
    assert permission_helper.call_count == 0

    store.exists = True
    assert request.state.mcp_server_can_update_existing_recheck() is True
    permission_helper.assert_called_once_with("com.test/race-server", "alice")


@pytest.mark.parametrize("prefix", [_MCP_AJAX_PREFIX, _MCP_REST_PREFIX])
def test_mcp_server_nested_delete_requires_can_delete(fastapi_client, monkeypatch, prefix):
    owner, owner_pw = create_user(fastapi_client.tracking_uri)
    editor, editor_pw = create_user(fastapi_client.tracking_uri)
    server_name = "com.test/nested-delete"

    with User(owner, owner_pw, monkeypatch):
        requests.post(
            url=fastapi_client.tracking_uri + prefix,
            json={"name": server_name},
            auth=(owner, owner_pw),
        ).raise_for_status()
        requests.post(
            url=f"{fastapi_client.tracking_uri}{prefix}/{server_name}/versions",
            json=_version_create_body(server_name),
            auth=(owner, owner_pw),
        ).raise_for_status()

    grant_role_permission(
        fastapi_client.tracking_uri,
        editor,
        "mcp_server",
        server_name,
        "EDIT",
    )

    with User(editor, editor_pw, monkeypatch):
        resp = requests.delete(
            url=f"{fastapi_client.tracking_uri}{prefix}/{server_name}/versions/1.0.0",
            auth=(editor, editor_pw),
        )
        assert resp.status_code == 403


@pytest.mark.parametrize("prefix", [_MCP_AJAX_PREFIX, _MCP_REST_PREFIX])
def test_nested_post_does_not_re_promote_downgraded_creator(fastapi_client, monkeypatch, prefix):
    creator, creator_pw = create_user(fastapi_client.tracking_uri)
    server_name = "com.test/re-promote"
    admin_auth = (ADMIN_USERNAME, ADMIN_PASSWORD)

    # Creator creates the server (auto-granted MANAGE via synthetic role).
    with User(creator, creator_pw, monkeypatch):
        requests.post(
            url=fastapi_client.tracking_uri + prefix,
            json={"name": server_name},
            auth=(creator, creator_pw),
        ).raise_for_status()

    # Admin downgrades creator: revoke MANAGE, then grant EDIT on the synthetic role.
    _send_rest_tracking_post_request(
        fastapi_client.tracking_uri,
        "/api/3.0/mlflow/users/permissions/revoke",
        {"username": creator, "resource_type": "mcp_server", "resource_id": server_name},
        auth=admin_auth,
    ).raise_for_status()
    _send_rest_tracking_post_request(
        fastapi_client.tracking_uri,
        "/api/3.0/mlflow/users/permissions/grant",
        {
            "username": creator,
            "resource_type": "mcp_server",
            "resource_id": server_name,
            "permission": "EDIT",
        },
        auth=admin_auth,
    ).raise_for_status()

    # Creator does a nested POST (version create) — should NOT restore MANAGE.
    with User(creator, creator_pw, monkeypatch):
        requests.post(
            url=f"{fastapi_client.tracking_uri}{prefix}/{server_name}/versions",
            json=_version_create_body(server_name),
            auth=(creator, creator_pw),
        ).raise_for_status()

    # Verify creator still cannot delete the server (requires MANAGE/can_delete).
    with User(creator, creator_pw, monkeypatch):
        resp = requests.delete(
            url=f"{fastapi_client.tracking_uri}{prefix}/{server_name}",
            auth=(creator, creator_pw),
        )
        assert resp.status_code == 403


@pytest.mark.parametrize("prefix", [_MCP_AJAX_PREFIX, _MCP_REST_PREFIX])
def test_non_version_nested_post_does_not_re_promote_downgraded_creator(
    fastapi_client, monkeypatch, prefix
):
    creator, creator_pw = create_user(fastapi_client.tracking_uri)
    server_name = "com.test/re-promote-tag"
    admin_auth = (ADMIN_USERNAME, ADMIN_PASSWORD)

    with User(creator, creator_pw, monkeypatch):
        requests.post(
            url=fastapi_client.tracking_uri + prefix,
            json={"name": server_name},
            auth=(creator, creator_pw),
        ).raise_for_status()

    _send_rest_tracking_post_request(
        fastapi_client.tracking_uri,
        "/api/3.0/mlflow/users/permissions/revoke",
        {"username": creator, "resource_type": "mcp_server", "resource_id": server_name},
        auth=admin_auth,
    ).raise_for_status()
    _send_rest_tracking_post_request(
        fastapi_client.tracking_uri,
        "/api/3.0/mlflow/users/permissions/grant",
        {
            "username": creator,
            "resource_type": "mcp_server",
            "resource_id": server_name,
            "permission": "EDIT",
        },
        auth=admin_auth,
    ).raise_for_status()

    # A nested POST like /tags should be allowed with EDIT, but must not
    # restore MANAGE to the original creator.
    with User(creator, creator_pw, monkeypatch):
        requests.post(
            url=f"{fastapi_client.tracking_uri}{prefix}/{server_name}/tags",
            json={"key": "env", "value": "dev"},
            auth=(creator, creator_pw),
        ).raise_for_status()

    resp = requests.get(
        url=f"{fastapi_client.tracking_uri}/api/3.0/mlflow/users/permissions/get",
        params={
            "username": creator,
            "resource_type": "mcp_server",
            "resource_id": server_name,
        },
        auth=admin_auth,
    )
    assert resp.status_code == 200
    assert resp.json()["permission"] == "EDIT"

    with User(creator, creator_pw, monkeypatch):
        resp = requests.delete(
            url=f"{fastapi_client.tracking_uri}{prefix}/{server_name}",
            auth=(creator, creator_pw),
        )
        assert resp.status_code == 403


def _version_create_body(name):
    return {
        "server_json": {"name": name, "version": "1.0.0"},
        "source": "https://example.com/server.py",
    }


@pytest.mark.parametrize("prefix", [_MCP_AJAX_PREFIX, _MCP_REST_PREFIX])
def test_mcp_server_version_create_implicitly_creates_parent(fastapi_client, monkeypatch, prefix):
    """POST /{name}/versions on a nonexistent server should succeed when the
    user has create rights, because the store auto-creates the parent.
    """
    creator, creator_pw = create_user(fastapi_client.tracking_uri)
    server_name = "com.test/implicit-parent"

    # Version create on a server that doesn't exist yet — should succeed
    # because any authenticated user can create servers (non-workspace mode).
    with User(creator, creator_pw, monkeypatch):
        resp = requests.post(
            url=f"{fastapi_client.tracking_uri}{prefix}/{server_name}/versions",
            json=_version_create_body(server_name),
            auth=(creator, creator_pw),
        )
        assert resp.status_code == 200, resp.text

    # Creator should have received MANAGE auto-grant on the implicitly
    # created parent, just as if they had called POST /mcp-servers directly.
    with User(creator, creator_pw, monkeypatch):
        resp = requests.patch(
            url=f"{fastapi_client.tracking_uri}{prefix}/{server_name}",
            json={"description": "creator can update"},
            auth=(creator, creator_pw),
        )
        assert resp.status_code == 200

    # A different user without a grant should still be denied updates.
    other, other_pw = create_user(fastapi_client.tracking_uri)
    with User(other, other_pw, monkeypatch):
        resp = requests.patch(
            url=f"{fastapi_client.tracking_uri}{prefix}/{server_name}",
            json={"description": "should fail"},
            auth=(other, other_pw),
        )
        assert resp.status_code == 403


@pytest.mark.parametrize("prefix", [_MCP_AJAX_PREFIX, _MCP_REST_PREFIX])
def test_mcp_server_version_create_on_existing_requires_update(fastapi_client, monkeypatch, prefix):
    """POST /{name}/versions on an existing server still requires can_update,
    not just create rights.
    """
    owner, owner_pw = create_user(fastapi_client.tracking_uri)
    reader, reader_pw = create_user(fastapi_client.tracking_uri)
    server_name = "com.test/existing-parent"

    # Owner creates the server explicitly.
    with User(owner, owner_pw, monkeypatch):
        requests.post(
            url=fastapi_client.tracking_uri + prefix,
            json={"name": server_name},
            auth=(owner, owner_pw),
        ).raise_for_status()

    # Reader has default READ permission — can_update=False.
    with User(reader, reader_pw, monkeypatch):
        resp = requests.post(
            url=f"{fastapi_client.tracking_uri}{prefix}/{server_name}/versions",
            json=_version_create_body(server_name),
            auth=(reader, reader_pw),
        )
        assert resp.status_code == 403

    # Owner has MANAGE — can_update=True.
    with User(owner, owner_pw, monkeypatch):
        resp = requests.post(
            url=f"{fastapi_client.tracking_uri}{prefix}/{server_name}/versions",
            json=_version_create_body(server_name),
            auth=(owner, owner_pw),
        )
        assert resp.status_code == 200


@pytest.mark.parametrize("prefix", [_MCP_AJAX_PREFIX, _MCP_REST_PREFIX])
def test_mcp_server_nested_post_does_not_escalate_existing_grant(
    fastapi_client, monkeypatch, prefix
):
    admin_auth = (ADMIN_USERNAME, ADMIN_PASSWORD)
    owner, owner_pw = create_user(fastapi_client.tracking_uri)
    editor, editor_pw = create_user(fastapi_client.tracking_uri)
    server_name = "com.test/no-escalate"

    # Owner creates the server (gets MANAGE auto-grant).
    with User(owner, owner_pw, monkeypatch):
        requests.post(
            url=fastapi_client.tracking_uri + prefix,
            json={"name": server_name},
            auth=(owner, owner_pw),
        ).raise_for_status()

    # Admin grants editor EDIT permission on the server.
    requests.post(
        url=f"{fastapi_client.tracking_uri}/api/3.0/mlflow/users/permissions/grant",
        json={
            "username": editor,
            "resource_type": "mcp_server",
            "resource_id": server_name,
            "permission": "EDIT",
        },
        auth=admin_auth,
    ).raise_for_status()

    # Editor creates a version on the existing server — should succeed (can_update).
    with User(editor, editor_pw, monkeypatch):
        resp = requests.post(
            url=f"{fastapi_client.tracking_uri}{prefix}/{server_name}/versions",
            json=_version_create_body(server_name),
            auth=(editor, editor_pw),
        )
        assert resp.status_code == 200

    # Editor's permission must still be EDIT, NOT escalated to MANAGE.
    resp = requests.get(
        url=f"{fastapi_client.tracking_uri}/api/3.0/mlflow/users/permissions/get",
        params={
            "username": editor,
            "resource_type": "mcp_server",
            "resource_id": server_name,
        },
        auth=admin_auth,
    )
    assert resp.status_code == 200
    assert resp.json()["permission"] == "EDIT"


@pytest.mark.parametrize("prefix", [_MCP_AJAX_PREFIX, _MCP_REST_PREFIX])
def test_mcp_server_nested_post_body_name_does_not_grant_manage(
    fastapi_client, monkeypatch, prefix
):
    # Nested POSTs must not auto-grant MANAGE from an injected body ``name``.
    admin_auth = (ADMIN_USERNAME, ADMIN_PASSWORD)
    editor, editor_pw = create_user(fastapi_client.tracking_uri)
    owned = "com.test/owned-for-tags"
    victim = "com.test/victim-server"

    requests.post(
        url=fastapi_client.tracking_uri + prefix,
        json={"name": owned},
        auth=admin_auth,
    ).raise_for_status()
    requests.post(
        url=fastapi_client.tracking_uri + prefix,
        json={"name": victim},
        auth=admin_auth,
    ).raise_for_status()
    requests.post(
        url=f"{fastapi_client.tracking_uri}/api/3.0/mlflow/users/permissions/grant",
        json={
            "username": editor,
            "resource_type": "mcp_server",
            "resource_id": owned,
            "permission": "EDIT",
        },
        auth=admin_auth,
    ).raise_for_status()

    with User(editor, editor_pw, monkeypatch):
        requests.post(
            url=f"{fastapi_client.tracking_uri}{prefix}/{owned}/tags",
            json={"key": "k", "value": "v", "name": victim},
            auth=(editor, editor_pw),
        ).raise_for_status()

    # No grant on the victim server — DELETE must still be forbidden.
    with User(editor, editor_pw, monkeypatch):
        resp = requests.delete(
            url=f"{fastapi_client.tracking_uri}{prefix}/{victim}",
            auth=(editor, editor_pw),
        )
        assert resp.status_code == 403


@pytest.mark.parametrize(
    "fastapi_client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
@pytest.mark.parametrize("prefix", [_MCP_AJAX_PREFIX, _MCP_REST_PREFIX])
def test_mcp_server_search_filters_unreadable(fastapi_client, monkeypatch, prefix):
    owner, owner_pw = create_user(fastapi_client.tracking_uri)
    reader, reader_pw = create_user(fastapi_client.tracking_uri)
    admin_auth = (ADMIN_USERNAME, ADMIN_PASSWORD)

    readable_names = ["com.test/visible-1", "com.test/visible-2"]
    hidden_name = "com.test/hidden"

    for name in readable_names + [hidden_name]:
        requests.post(
            url=fastapi_client.tracking_uri + prefix,
            json={"name": name},
            auth=admin_auth,
        ).raise_for_status()

    for name in readable_names:
        grant_role_permission(fastapi_client.tracking_uri, reader, "mcp_server", name, "READ")

    # Admin sees all servers.
    resp = requests.get(
        url=fastapi_client.tracking_uri + prefix,
        auth=admin_auth,
    )
    assert resp.status_code == 200
    admin_names = {s["name"] for s in resp.json()["mcp_servers"]}
    assert readable_names[0] in admin_names
    assert hidden_name in admin_names

    # Reader sees only servers with an explicit READ grant.
    with User(reader, reader_pw, monkeypatch):
        resp = requests.get(
            url=fastapi_client.tracking_uri + prefix,
            auth=(reader, reader_pw),
        )
        assert resp.status_code == 200
        reader_names = {s["name"] for s in resp.json()["mcp_servers"]}
        assert reader_names == set(readable_names)
        assert hidden_name not in reader_names


@pytest.mark.parametrize(
    "fastapi_client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
@pytest.mark.parametrize("prefix", [_MCP_AJAX_PREFIX, _MCP_REST_PREFIX])
def test_mcp_server_endpoint_search_filters_by_parent(fastapi_client, monkeypatch, prefix):
    reader, reader_pw = create_user(fastapi_client.tracking_uri)
    admin_auth = (ADMIN_USERNAME, ADMIN_PASSWORD)

    visible = "com.test/bind-visible"
    hidden = "com.test/bind-hidden"

    for name in [visible, hidden]:
        requests.post(
            url=fastapi_client.tracking_uri + prefix,
            json={"name": name},
            auth=admin_auth,
        ).raise_for_status()
        ver_resp = requests.post(
            url=f"{fastapi_client.tracking_uri}{prefix}/{name}/versions",
            json=_version_create_body(name),
            auth=admin_auth,
        )
        ver_resp.raise_for_status()
        version = ver_resp.json()["version"]
        requests.post(
            url=f"{fastapi_client.tracking_uri}{prefix}/{name}/endpoints",
            json={
                "server_version": version,
                "url": f"https://example.com/{name}",
            },
            auth=admin_auth,
        ).raise_for_status()

    grant_role_permission(fastapi_client.tracking_uri, reader, "mcp_server", visible, "READ")

    # Reader sees only endpoints whose parent server is readable.
    with User(reader, reader_pw, monkeypatch):
        resp = requests.get(
            url=f"{fastapi_client.tracking_uri}{prefix}/endpoints",
            auth=(reader, reader_pw),
        )
        assert resp.status_code == 200
        endpoint_servers = {e["server_name"] for e in resp.json()["mcp_access_endpoints"]}
        assert endpoint_servers == {visible}
        assert hidden not in endpoint_servers


@pytest.mark.parametrize(
    "fastapi_client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
@pytest.mark.parametrize("prefix", [_MCP_AJAX_PREFIX, _MCP_REST_PREFIX])
def test_mcp_server_search_backfills_after_filtering(fastapi_client, monkeypatch, prefix):
    reader, reader_pw = create_user(fastapi_client.tracking_uri)
    admin_auth = (ADMIN_USERNAME, ADMIN_PASSWORD)

    # 3 readable servers so backfill must break mid-backend-page and still
    # return z-read3 on the next client request (not skip it).
    readable = ["com.test/z-read1", "com.test/z-read2", "com.test/z-read3"]
    hidden = ["com.test/a-hid1", "com.test/a-hid2"]
    for name in readable + hidden:
        requests.post(
            url=fastapi_client.tracking_uri + prefix,
            json={"name": name},
            auth=admin_auth,
        ).raise_for_status()

    for name in readable:
        grant_role_permission(fastapi_client.tracking_uri, reader, "mcp_server", name, "READ")

    # Request max_results=2. Without backfill the first page might contain a
    # mix of readable/hidden servers and return fewer than 2 readable rows.
    with User(reader, reader_pw, monkeypatch):
        all_readable = []
        page_token = None
        while True:
            params = {"max_results": 2}
            if page_token:
                params["page_token"] = page_token
            resp = requests.get(
                url=fastapi_client.tracking_uri + prefix,
                params=params,
                auth=(reader, reader_pw),
            )
            assert resp.status_code == 200
            data = resp.json()
            page = data["mcp_servers"]
            all_readable.extend(page)
            page_token = data.get("next_page_token")
            if not page_token:
                break
            # Each non-final page must be full (max_results items).
            assert len(page) == 2

    assert {s["name"] for s in all_readable} == set(readable)
