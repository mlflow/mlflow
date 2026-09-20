import asyncio
import base64
import json
import logging
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
    MLFLOW_AUTH_ADMIN_PASSWORD,
    MLFLOW_ENABLE_WORKSPACES,
    MLFLOW_FLASK_SERVER_SECRET_KEY,
    MLFLOW_TRACKING_PASSWORD,
    MLFLOW_TRACKING_USERNAME,
    MLFLOW_WORKSPACE_STORE_URI,
)
from mlflow.exceptions import MlflowException
from mlflow.prompt.constants import IS_PROMPT_TAG_KEY
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
from mlflow.server.mcp_server_api import (
    get_mcp_server,
    get_mcp_server_version,
    search_all_access_endpoints,
    search_mcp_server_versions,
    search_mcp_servers,
)
from mlflow.store.jobs.sqlalchemy_store import SqlAlchemyJobStore
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

    Neither the packaged config nor the fixtures carry an admin password (MLflow
    ships none), so the bootstrap password is supplied through
    ``MLFLOW_AUTH_ADMIN_PASSWORD`` unless ``extra_env`` already sets it.
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
    return {
        MLFLOW_AUTH_ADMIN_PASSWORD.name: ADMIN_PASSWORD,
        **extra_env,
        "MLFLOW_AUTH_CONFIG_PATH": str(dst_path),
    }


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
    if extra_env.get("_MLFLOW_SERVER_SERVE_ARTIFACTS") == "true":
        extra_env.setdefault(
            "_MLFLOW_SERVER_ARTIFACT_DESTINATION",
            str(tmp_path / "served_artifacts"),
        )

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


def test_find_fastapi_validator_handles_static_prefix(monkeypatch):
    monkeypatch.delenv(STATIC_PREFIX_ENV_VAR, raising=False)
    assert _find_fastapi_validator("/gateway/mlflow/v1/chat/completions", "GET") is not None
    assert _find_fastapi_validator("/gateway/mlflow/v1/models", "GET") is not None
    assert _find_fastapi_validator("/v1/traces", "GET") is not None
    assert _find_fastapi_validator("/ajax-api/3.0/jobs/search", "GET") is not None
    assert _find_fastapi_validator("/ajax-api/3.0/mlflow/assistant/config", "GET") is not None
    assert _find_fastapi_validator("/mlflow/gateway/mlflow/v1/chat/completions", "GET") is None

    monkeypatch.setenv(STATIC_PREFIX_ENV_VAR, "/mlflow")
    assert _find_fastapi_validator("/mlflow/gateway/mlflow/v1/chat/completions", "GET") is not None
    assert _find_fastapi_validator("/mlflow/gateway/mlflow/v1/models", "GET") is not None
    assert _find_fastapi_validator("/mlflow/v1/traces", "GET") is not None
    assert _find_fastapi_validator("/mlflow/ajax-api/3.0/jobs/search", "GET") is not None
    assert (
        _find_fastapi_validator("/mlflow/ajax-api/3.0/mlflow/assistant/config", "GET") is not None
    )
    # An unprefixed route root still resolves a validator: nothing serves that path
    # once a prefix is configured, and this errs toward requiring auth.
    assert _find_fastapi_validator("/gateway/mlflow/v1/chat/completions", "GET") is not None
    assert _find_fastapi_validator("/mlflow/api/2.0/mlflow/experiments/search", "GET") is None


def test_find_fastapi_validator_leaves_prefixed_artifact_paths_to_flask(monkeypatch):
    monkeypatch.setenv(STATIC_PREFIX_ENV_VAR, "/mlflow")
    artifact_path = "/api/2.0/mlflow-artifacts/artifacts/1/run-id/artifacts/model.pkl"

    native = _find_fastapi_validator(artifact_path, "GET")
    assert native.__qualname__ == "_get_fastapi_proxy_artifact_validator.<locals>.validator"

    assert _find_fastapi_validator(f"/mlflow{artifact_path}", "GET") is None


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


def test_extract_experiment_id_from_artifact_proxy_path():
    assert (
        auth_module._extract_experiment_id_from_artifact_proxy_path(
            "/api/2.0/mlflow-artifacts/artifacts/42/run-id/artifacts/model.pkl"
        )
        == "42"
    )
    assert (
        auth_module._extract_experiment_id_from_artifact_proxy_path(
            "/ajax-api/2.0/mlflow-artifacts/artifacts/workspaces/team-a/7/run-id/artifacts/f"
        )
        == "7"
    )
    for action in ("create", "complete", "abort"):
        assert (
            auth_module._extract_experiment_id_from_artifact_proxy_path(
                f"/api/2.0/mlflow-artifacts/mpu/{action}/99/run-id/artifacts/model"
            )
            == "99"
        )
        assert (
            auth_module._extract_experiment_id_from_artifact_proxy_path(
                f"/ajax-api/2.0/mlflow-artifacts/mpu/{action}/workspaces/ws/3/run/artifacts/x"
            )
            == "3"
        )
    assert (
        auth_module._extract_experiment_id_from_artifact_proxy_path(
            "/api/2.0/mlflow-artifacts/artifacts",
            query_path="55/models/m-abc123/artifacts",
        )
        == "55"
    )
    assert (
        auth_module._extract_experiment_id_from_artifact_proxy_path(
            "/api/2.0/mlflow/experiments/get"
        )
        is None
    )


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


def test_after_request_handlers_contains_only_declared_handlers():
    declared = set(auth_module.AFTER_REQUEST_PATH_HANDLERS.values())

    leaked = {
        (path, method): handler
        for (path, method), handler in auth_module.AFTER_REQUEST_HANDLERS.items()
        if getattr(handler, "__module__", "") == "mlflow.server.handlers"
        and handler not in declared
    }

    assert leaked == {}


@pytest.mark.parametrize(
    ("path", "method"),
    [
        # invoke + demo gate via the exact-match table, so they leave this list. Jobs
        # stay: they gate via the regex JOB_BEFORE_REQUEST_VALIDATORS map, not the table.
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


def test_proxy_artifact_authorization_required_fastapi(fastapi_client, monkeypatch):
    username1, password1 = create_user(fastapi_client.tracking_uri)
    username2, password2 = create_user(fastapi_client.tracking_uri)

    with User(username1, password1, monkeypatch):
        experiment_id = fastapi_client.create_experiment("proxy-artifact-authz-test-fastapi")

    response = requests.put(
        url=(
            fastapi_client.tracking_uri
            + f"/ajax-api/2.0/mlflow-artifacts/artifacts/{experiment_id}/test.txt"
        ),
        data=b"forbidden",
        auth=(username2, password2),
    )
    assert response.status_code == 403


@pytest.mark.parametrize(
    ("path", "method"),
    [
        ("/api/2.0/mlflow-artifacts/artifacts/1/run-id/artifacts/model.pkl", "GET"),
        ("/ajax-api/2.0/mlflow-artifacts/artifacts/1/run-id/artifacts/model.pkl", "GET"),
        ("/api/2.0/mlflow-artifacts/artifacts/1/run-id/artifacts/model.pkl", "PUT"),
        ("/ajax-api/2.0/mlflow-artifacts/artifacts/1/run-id/artifacts/model.pkl", "PUT"),
    ],
)
def test_fastapi_validator_matches_native_artifact_routes(path, method):
    assert _find_fastapi_validator(path, method) is not None


@pytest.mark.parametrize(
    ("path", "method"),
    [
        ("/api/2.0/mlflow-artifacts/artifacts", "GET"),
        ("/ajax-api/2.0/mlflow-artifacts/artifacts", "GET"),
        ("/api/2.0/mlflow-artifacts/artifacts/1/run-id/artifacts/model.pkl", "DELETE"),
        ("/ajax-api/2.0/mlflow-artifacts/artifacts/1/run-id/artifacts/model.pkl", "DELETE"),
        ("/api/2.0/mlflow-artifacts/mpu/create/1/run-id/artifacts/model.pkl", "POST"),
        ("/ajax-api/2.0/mlflow-artifacts/mpu/abort/1/run-id/artifacts/model.pkl", "POST"),
    ],
)
def test_fastapi_validator_skips_flask_fallback_artifact_routes(path, method):
    assert _find_fastapi_validator(path, method) is None


def test_proxy_artifact_permission_reuses_authenticated_flask_user(monkeypatch):
    permission = SimpleNamespace(can_read=True, can_update=True, can_manage=False)
    authenticate_request = mock.Mock(side_effect=AssertionError("should not re-authenticate"))

    monkeypatch.setattr(auth_module, "authenticate_request", authenticate_request)
    monkeypatch.setattr(auth_module, "_get_experiment_id_from_view_args", lambda: "123")
    monkeypatch.setattr(auth_module, "_role_permission_for", lambda **_: permission)
    monkeypatch.setattr(auth_module, "_get_role_permission_or_default", lambda perm: perm)

    with auth_module.app.test_request_context("/api/2.0/mlflow-artifacts/artifacts"):
        auth_module.g.mlflow_authenticated_user = "alice"
        result = auth_module._get_permission_from_experiment_id_artifact_proxy()

    assert result is permission
    authenticate_request.assert_not_called()


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


@pytest.mark.parametrize(
    "fastapi_client",
    [
        {
            "MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini",
            "_MLFLOW_SERVER_SERVE_ARTIFACTS": "true",
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    ("list_path", "query_path_template"),
    [
        ("/api/2.0/mlflow-artifacts/artifacts", "{experiment_id}/models/m-abc123/artifacts"),
        ("/ajax-api/2.0/mlflow-artifacts/artifacts", "{experiment_id}/models/m-abc123/artifacts"),
        (
            "/api/2.0/mlflow-artifacts/artifacts",
            "workspaces/default/{experiment_id}/models/m-abc123/artifacts",
        ),
    ],
)
def test_proxy_artifact_list_query_param_uses_experiment_permission_on_fastapi_server(
    fastapi_client, monkeypatch, list_path, query_path_template
):
    # List-artifacts is still served by Flask on the FastAPI server, so this
    # exercises the fallback auth path rather than the native FastAPI router.
    username1, password1 = create_user(fastapi_client.tracking_uri)
    username2, password2 = create_user(fastapi_client.tracking_uri)

    with User(username1, password1, monkeypatch):
        experiment_id = fastapi_client.create_experiment(
            "proxy-artifact-list-query-param-test-fastapi"
        )

    query_path = query_path_template.format(experiment_id=experiment_id)
    response = requests.get(
        url=fastapi_client.tracking_uri + list_path,
        params={"path": query_path},
        auth=(username1, password1),
    )
    assert response.status_code == 200

    response = requests.get(
        url=fastapi_client.tracking_uri + list_path,
        params={"path": query_path},
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


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_presigned_upload_url_logged_model_authorization_required(client, monkeypatch):
    # Logged-model-scoped mints dispatch on model_id; permission is inherited from
    # the owning experiment, so a user without experiment permission must get 403.
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)

    with User(username1, password1, monkeypatch):
        experiment_id = client.create_experiment("presigned-upload-model-authz-test")
        logged_model = client.create_logged_model(experiment_id)
        model_id = logged_model.model_id

    response = requests.post(
        url=client.tracking_uri + "/api/2.0/mlflow/artifacts/presigned-upload-url",
        json={"model_id": model_id, "path": "model.pkl"},
        auth=(username2, password2),
    )
    assert response.status_code == 403

    # The owner passes the auth layer and reaches the handler, which rejects the
    # local (file://) logged-model artifact location with 501.
    response = requests.post(
        url=client.tracking_uri + "/api/2.0/mlflow/artifacts/presigned-upload-url",
        json={"model_id": model_id, "path": "model.pkl"},
        auth=(username1, password1),
    )
    assert response.status_code == 501


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
@pytest.mark.parametrize("scope", ["run", "model"])
def test_presigned_upload_url_authorizes_camel_case_field_aliases(client, monkeypatch, scope):
    # The handler parses the body through the proto, which accepts the canonical
    # camelCase aliases `runId` / `modelId`. The validator must resolve permissions
    # from the same parsed IDs — reading only the raw snake_case keys would let a
    # camelCase request slip past the exactly-one check (400) or the permission
    # lookup instead of being denied with 403.
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)
    with User(username1, password1, monkeypatch):
        experiment_id = client.create_experiment(f"presigned-upload-camel-{scope}-authz-test")
        if scope == "run":
            body = {"runId": client.create_run(experiment_id).info.run_id, "path": "model.pkl"}
        else:
            body = {"modelId": client.create_logged_model(experiment_id).model_id, "path": "a.pkl"}
    # user2 has no permission on user1's experiment: camelCase must still be denied.
    response = requests.post(
        url=client.tracking_uri + "/api/2.0/mlflow/artifacts/presigned-upload-url",
        json=body,
        auth=(username2, password2),
    )
    assert response.status_code == 403
    # The owner passes auth and reaches the handler (local file:// store -> 501).
    response = requests.post(
        url=client.tracking_uri + "/api/2.0/mlflow/artifacts/presigned-upload-url",
        json=body,
        auth=(username1, password1),
    )
    assert response.status_code == 501
    # Mixed-case both-IDs request must still hit the exactly-one 400, not 403/404.
    response = requests.post(
        url=client.tracking_uri + "/api/2.0/mlflow/artifacts/presigned-upload-url",
        json={**body, ("modelId" if scope == "run" else "runId"): "other"},
        auth=(username2, password2),
    )
    assert response.status_code == 400
    assert "Exactly one of run_id and model_id" in response.text


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_presigned_upload_url_exactly_one_scope_enforced_before_authorization(client, monkeypatch):
    # The auth validator runs before the handler, so it must enforce the
    # exactly-one-of run_id / model_id contract itself: a malformed request
    # carrying both IDs (or neither) must get the documented 400 even from a
    # user without any permission — not a 403/404 leaked by resolving either
    # resource's permission first.
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)

    with User(username1, password1, monkeypatch):
        experiment_id = client.create_experiment("presigned-upload-xor-authz-test")
        run = client.create_run(experiment_id)
        run_id = run.info.run_id
        logged_model = client.create_logged_model(experiment_id)
        model_id = logged_model.model_id

    # user2 has no permission on user1's experiment; both IDs → 400, not 403.
    response = requests.post(
        url=client.tracking_uri + "/api/2.0/mlflow/artifacts/presigned-upload-url",
        json={"run_id": run_id, "model_id": model_id, "path": "model.pkl"},
        auth=(username2, password2),
    )
    assert response.status_code == 400
    assert "Exactly one of run_id and model_id" in response.text

    # A nonexistent model id must not leak existence via 404 either.
    response = requests.post(
        url=client.tracking_uri + "/api/2.0/mlflow/artifacts/presigned-upload-url",
        json={"run_id": run_id, "model_id": "m-nonexistent", "path": "model.pkl"},
        auth=(username2, password2),
    )
    assert response.status_code == 400

    # Neither ID → 400 with the same message.
    response = requests.post(
        url=client.tracking_uri + "/api/2.0/mlflow/artifacts/presigned-upload-url",
        json={"path": "model.pkl"},
        auth=(username2, password2),
    )
    assert response.status_code == 400
    assert "Exactly one of run_id and model_id" in response.text


@pytest.mark.parametrize(
    "fastapi_client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
@pytest.mark.parametrize("mpu_action", ["create", "complete", "abort"])
def test_mpu_authorization_required_via_flask_fallback_on_fastapi_server(
    fastapi_client, monkeypatch, mpu_action
):
    username1, password1 = create_user(fastapi_client.tracking_uri)
    username2, password2 = create_user(fastapi_client.tracking_uri)

    with User(username1, password1, monkeypatch):
        experiment_id = fastapi_client.create_experiment(f"mpu-authz-fastapi-{mpu_action}")

    # MPU routes are still handled by Flask on the FastAPI server, so both
    # assertions here exercise the Flask fallback auth path.
    response = requests.post(
        url=(
            fastapi_client.tracking_uri
            + f"/api/2.0/mlflow-artifacts/mpu/{mpu_action}/{experiment_id}/artifacts/model"
        ),
        json={"path": "python_model.pkl", "num_parts": 1},
        auth=(username2, password2),
    )
    assert response.status_code == 403

    # If middleware failed to parse the experiment id, owner would also get 403
    # from default_permission=NO_PERMISSIONS.
    response = requests.post(
        url=(
            fastapi_client.tracking_uri
            + f"/api/2.0/mlflow-artifacts/mpu/{mpu_action}/{experiment_id}/artifacts/model"
        ),
        json={"path": "python_model.pkl", "num_parts": 1},
        auth=(username1, password1),
    )
    assert response.status_code != 403


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


@pytest.fixture
def jwt_fastapi_artifact_client(tmp_path):
    path = tmp_path.joinpath("sqlalchemy.db").as_uri()
    backend_uri = ("sqlite://" if is_windows() else "sqlite:////") + path[len("file://") :]
    artifact_dest = str(tmp_path / "jwt_artifacts")
    extra_env = _isolate_auth_config({"MLFLOW_AUTH_CONFIG_PATH": "fixtures/jwt_auth.ini"}, tmp_path)
    extra_env[MLFLOW_FLASK_SERVER_SECRET_KEY.name] = "my-secret-key"
    extra_env["_MLFLOW_SGI_NAME"] = "uvicorn"
    extra_env["PYTHONPATH"] = str(Path.cwd() / "examples" / "jwt_auth")
    extra_env["_MLFLOW_SERVER_SERVE_ARTIFACTS"] = "true"
    extra_env["_MLFLOW_SERVER_ARTIFACT_DESTINATION"] = artifact_dest

    with _init_server(
        backend_uri=backend_uri,
        root_artifact_uri=tmp_path.joinpath("artifacts").as_uri(),
        extra_env=extra_env,
        app="mlflow.server.auth:create_app",
        server_type="fastapi",
    ) as url:
        yield MlflowClient(url)


def _write_counting_custom_auth_setup(tmp_path: Path):
    counter_path = tmp_path / "auth_count.txt"
    counter_path.write_text("0")
    (tmp_path / "counting_auth.py").write_text(
        "from pathlib import Path\n"
        "from werkzeug.datastructures import Authorization\n\n"
        f"_COUNTER_PATH = Path({str(counter_path)!r})\n\n"
        "def authenticate_request():\n"
        "    _COUNTER_PATH.write_text(str(int(_COUNTER_PATH.read_text()) + 1))\n"
        f'    return Authorization("basic", {{"username": {ADMIN_USERNAME!r}}})\n'
    )
    config_path = tmp_path / "counting_auth.ini"
    config_path.write_text(
        "[mlflow]\n"
        "default_permission = READ\n"
        f"database_uri = sqlite:///{tmp_path / 'basic_auth.db'}\n"
        f"admin_username = {ADMIN_USERNAME}\n"
        f"admin_password = {ADMIN_PASSWORD}\n"
        "authorization_function = counting_auth:authenticate_request\n"
        "grant_default_workspace_access = false\n"
    )
    return counter_path, {
        "MLFLOW_AUTH_CONFIG_PATH": str(config_path),
        MLFLOW_FLASK_SERVER_SECRET_KEY.name: "my-secret-key",
        "PYTHONPATH": str(tmp_path),
        "_MLFLOW_SERVER_SERVE_ARTIFACTS": "true",
        "_MLFLOW_SERVER_ARTIFACT_DESTINATION": str(tmp_path / "served_artifacts"),
    }


@pytest.fixture
def counting_custom_auth_fastapi_client(tmp_path):
    tmp_path = tmp_path / "fastapi"
    tmp_path.mkdir()
    path = tmp_path.joinpath("sqlalchemy.db").as_uri()
    backend_uri = ("sqlite://" if is_windows() else "sqlite:////") + path[len("file://") :]
    counter_path, extra_env = _write_counting_custom_auth_setup(tmp_path)
    extra_env["_MLFLOW_SGI_NAME"] = "uvicorn"

    with _init_server(
        backend_uri=backend_uri,
        root_artifact_uri=tmp_path.joinpath("artifacts").as_uri(),
        extra_env=extra_env,
        app="mlflow.server.auth:create_app",
        server_type="fastapi",
    ) as url:
        yield MlflowClient(url), counter_path


@pytest.fixture
def counting_custom_auth_flask_client(tmp_path):
    tmp_path = tmp_path / "flask"
    tmp_path.mkdir()
    path = tmp_path.joinpath("sqlalchemy.db").as_uri()
    backend_uri = ("sqlite://" if is_windows() else "sqlite:////") + path[len("file://") :]
    counter_path, extra_env = _write_counting_custom_auth_setup(tmp_path)

    with _init_server(
        backend_uri=backend_uri,
        root_artifact_uri=tmp_path.joinpath("artifacts").as_uri(),
        extra_env=extra_env,
        app="mlflow.server.auth:create_app",
        server_type="flask",
    ) as url:
        yield MlflowClient(url), counter_path


def _count_custom_auth_calls_for_artifact_list(client, counter_path):
    experiment_id = client.create_experiment(f"counting-auth-fallback-{random_str()}")
    counter_path.write_text("0")
    response = requests.get(
        f"{client.tracking_uri}/api/2.0/mlflow-artifacts/artifacts",
        params={"path": f"{experiment_id}/models/m-abc123/artifacts"},
    )
    return response, int(counter_path.read_text())


def test_custom_auth_flask_fallback_artifact_list_matches_flask_auth_count_on_fastapi_server(
    counting_custom_auth_flask_client,
    counting_custom_auth_fastapi_client,
):
    flask_client, flask_counter_path = counting_custom_auth_flask_client
    fastapi_client, fastapi_counter_path = counting_custom_auth_fastapi_client

    flask_response, flask_count = _count_custom_auth_calls_for_artifact_list(
        flask_client, flask_counter_path
    )
    fastapi_response, fastapi_count = _count_custom_auth_calls_for_artifact_list(
        fastapi_client, fastapi_counter_path
    )

    assert flask_response.status_code == 200
    assert fastapi_response.status_code == 200
    assert fastapi_count == flask_count


def test_custom_auth_artifact_upload_download_fastapi(jwt_fastapi_artifact_client):
    client = jwt_fastapi_artifact_client
    admin_token = jwt.encode({"username": ADMIN_USERNAME}, "secret", algorithm="HS256")
    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    # Create experiment as admin
    response = requests.post(
        f"{client.tracking_uri}/api/2.0/mlflow/experiments/create",
        headers=admin_headers,
        json={"name": "jwt-artifact-e2e"},
    )
    response.raise_for_status()
    experiment_id = response.json()["experiment_id"]

    artifact_path = f"{experiment_id}/run-id/artifacts/model.pkl"
    artifact_url = f"{client.tracking_uri}/api/2.0/mlflow-artifacts/artifacts/{artifact_path}"
    payload = b"trained model weights v1"

    # Upload artifact with valid JWT (admin has full access)
    put_resp = requests.put(artifact_url, data=payload, headers=admin_headers)
    assert put_resp.status_code == 200, f"Upload failed: {put_resp.status_code} {put_resp.text}"

    # Download artifact with valid JWT
    get_resp = requests.get(artifact_url, headers=admin_headers)
    assert get_resp.status_code == 200, f"Download failed: {get_resp.status_code} {get_resp.text}"
    assert get_resp.content == payload

    # Verify non-admin user with valid JWT can also access (admins grant global read)
    username, _ = _mlflow_create_user_rest(client.tracking_uri, admin_headers)
    user_token = jwt.encode({"username": username}, "secret", algorithm="HS256")
    user_headers = {"Authorization": f"Bearer {user_token}"}

    get_resp = requests.get(artifact_url, headers=user_headers)
    assert get_resp.status_code in (200, 403)  # depends on default_permission


def test_custom_auth_artifact_rejects_invalid_token_fastapi(jwt_fastapi_artifact_client):
    client = jwt_fastapi_artifact_client
    base = f"{client.tracking_uri}/api/2.0/mlflow-artifacts/artifacts"
    artifact_url = f"{base}/1/run-id/artifacts/model.pkl"

    # No auth header → 401
    resp = requests.get(artifact_url)
    assert resp.status_code == 401

    # Invalid JWT secret → 401
    bad_token = jwt.encode({"username": "admin"}, "wrong-secret", algorithm="HS256")
    resp = requests.get(artifact_url, headers={"Authorization": f"Bearer {bad_token}"})
    assert resp.status_code == 401

    # Malformed header → 401
    resp = requests.get(artifact_url, headers={"Authorization": "NotBearer xyz"})
    assert resp.status_code == 401


def test_custom_auth_artifact_denies_unauthorized_user_fastapi(jwt_fastapi_artifact_client):
    client = jwt_fastapi_artifact_client
    admin_token = jwt.encode({"username": ADMIN_USERNAME}, "secret", algorithm="HS256")
    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    # Create two users
    user1, _ = _mlflow_create_user_rest(client.tracking_uri, admin_headers)
    user2, _ = _mlflow_create_user_rest(client.tracking_uri, admin_headers)
    user1_token = jwt.encode({"username": user1}, "secret", algorithm="HS256")
    user2_token = jwt.encode({"username": user2}, "secret", algorithm="HS256")
    user1_headers = {"Authorization": f"Bearer {user1_token}"}
    user2_headers = {"Authorization": f"Bearer {user2_token}"}

    # Create experiment as admin, grant EDIT to user1 only via roles API with JWT
    response = requests.post(
        f"{client.tracking_uri}/api/2.0/mlflow/experiments/create",
        headers=admin_headers,
        json={"name": "jwt-artifact-authz-e2e"},
    )
    response.raise_for_status()
    experiment_id = response.json()["experiment_id"]

    role_name = f"_test_jwt_{random_str()}"
    resp = requests.post(
        f"{client.tracking_uri}/api/3.0/mlflow/roles/create",
        headers=admin_headers,
        json={"name": role_name, "workspace": "default"},
    )
    resp.raise_for_status()
    role_id = resp.json()["role"]["id"]

    resp = requests.post(
        f"{client.tracking_uri}/api/3.0/mlflow/roles/permissions/add",
        headers=admin_headers,
        json={
            "role_id": role_id,
            "resource_type": "experiment",
            "resource_pattern": experiment_id,
            "permission": "EDIT",
        },
    )
    resp.raise_for_status()

    resp = requests.post(
        f"{client.tracking_uri}/api/3.0/mlflow/roles/assign",
        headers=admin_headers,
        json={"username": user1, "role_id": role_id},
    )
    resp.raise_for_status()

    artifact_path = f"{experiment_id}/run-id/artifacts/secret.bin"
    artifact_url = f"{client.tracking_uri}/api/2.0/mlflow-artifacts/artifacts/{artifact_path}"

    # user1 can upload
    put_resp = requests.put(artifact_url, data=b"secret data", headers=user1_headers)
    assert put_resp.status_code == 200

    # user2 cannot upload (no permission on this experiment)
    put_resp = requests.put(artifact_url, data=b"hacked", headers=user2_headers)
    assert put_resp.status_code == 403


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
def test_graphql_search_model_versions_judges_prompts_by_prompt_grants(client, monkeypatch):
    owner, owner_password = create_user(client.tracking_uri)
    prompt_reader, prompt_reader_password = create_user(client.tracking_uri)
    model_reader, model_reader_password = create_user(client.tracking_uri)
    prompt_name = f"gql_prompt_{random_str()}"

    with User(owner, owner_password, monkeypatch):
        client.register_prompt(prompt_name, "Hello, {{name}}!")

    grant_role_permission(client.tracking_uri, prompt_reader, "prompt", prompt_name, "READ")
    grant_role_permission(client.tracking_uri, model_reader, "registered_model", "*", "READ")

    query = """
    query SearchModelVersions($input: MlflowSearchModelVersionsInput){
      mlflowSearchModelVersions(input: $input){
        modelVersions { name version }
      }
    }
    """
    variables = {
        "input": {"filter": f"tags.`mlflow.prompt.is_prompt` = 'true' AND name = '{prompt_name}'"}
    }

    def visible_names(auth):
        response = _graphql_query(client.tracking_uri, query, variables=variables, auth=auth)
        response.raise_for_status()
        payload = response.json()
        assert payload.get("errors") in (None, [])
        return [mv["name"] for mv in payload["data"]["mlflowSearchModelVersions"]["modelVersions"]]

    # A prompt grant is what makes a prompt version readable, as on the REST search path.
    assert visible_names((prompt_reader, prompt_reader_password)) == [prompt_name]
    # A registered-model grant, even a wildcard, does not reach into the prompt namespace.
    assert visible_names((model_reader, model_reader_password)) == []


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_graphql_nested_run_model_versions_filtered_by_model_permission(client, monkeypatch):
    """
    Regression test for GHSA-f253-vggg-rwh8: the nested run.modelVersions field must
    apply the same per-model READ filter as the top-level mlflowSearchModelVersions query.
    """
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)

    readable = [0, 2, 4]

    with User(username1, password1, monkeypatch):
        experiment_id = client.create_experiment("gql_nested_mv_test_exp")
        run_id = client.create_run(experiment_id).info.run_id
        for i in range(5):
            rm = client.create_registered_model(f"gql_nested_mv_model{i}")
            client.create_model_version(rm.name, f"runs:/{run_id}/model", run_id=run_id)
            if i in readable:
                grant_role_permission(
                    client.tracking_uri, username2, "registered_model", rm.name, "READ"
                )

    # user2 can read the experiment (and thus the run) but only some of the models
    grant_role_permission(client.tracking_uri, username2, "experiment", experiment_id, "READ")

    get_run_query = """
    query($runId: String!) {
      mlflowGetRun(input: {runId: $runId}) {
        run { modelVersions { name version } }
      }
    }
    """
    search_runs_query = """
    query($experimentIds: [String]!) {
      mlflowSearchRuns(input: {experimentIds: $experimentIds}) {
        runs { modelVersions { name version } }
      }
    }
    """

    def nested_names(query, variables, auth):
        response = _graphql_query(client.tracking_uri, query, variables=variables, auth=auth)
        response.raise_for_status()
        payload = response.json()
        assert payload.get("errors") in (None, [])
        if "mlflowGetRun" in payload["data"]:
            model_versions = payload["data"]["mlflowGetRun"]["run"]["modelVersions"]
        else:
            runs = payload["data"]["mlflowSearchRuns"]["runs"]
            assert len(runs) == 1, runs
            model_versions = runs[0]["modelVersions"]
        return sorted(mv["name"] for mv in model_versions)

    get_run_vars = {"runId": run_id}
    search_runs_vars = {"experimentIds": [experiment_id]}

    # user1 (owner) sees every version through both nested paths
    assert nested_names(get_run_query, get_run_vars, (username1, password1)) == [
        f"gql_nested_mv_model{i}" for i in range(5)
    ]
    assert nested_names(search_runs_query, search_runs_vars, (username1, password1)) == [
        f"gql_nested_mv_model{i}" for i in range(5)
    ]

    # user2 only sees versions of models they can read, matching the top-level search
    assert nested_names(get_run_query, get_run_vars, (username2, password2)) == [
        f"gql_nested_mv_model{i}" for i in readable
    ]
    assert nested_names(search_runs_query, search_runs_vars, (username2, password2)) == [
        f"gql_nested_mv_model{i}" for i in readable
    ]


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
@pytest.mark.parametrize("source_id_key", ["run_id", "runId", "model_id", "modelId"])
def test_create_model_version_empty_source_id_does_not_bypass(
    client: MlflowClient, monkeypatch: pytest.MonkeyPatch, source_id_key: str
):
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)

    with User(username1, password1, monkeypatch):
        exp_id = client.create_experiment("empty-id-authz-exp")
        run = client.create_run(exp_id)
        source = run.info.artifact_uri

    with User(username2, password2, monkeypatch):
        rm = client.create_registered_model("empty-id-authz-model")

    # An explicitly-supplied empty source id must not skip the source-read guard: the
    # request is denied rather than slipping past as if the id were absent.
    response = _send_rest_tracking_post_request(
        client.tracking_uri,
        "/api/2.0/mlflow/model-versions/create",
        json_payload={"name": rm.name, "source": source, source_id_key: ""},
        auth=(username2, password2),
    )
    assert response.status_code == 403
    assert "Permission denied" in response.text


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
@pytest.mark.parametrize("source_id_key", ["run_id", "runId", "model_id", "modelId"])
def test_create_model_version_nonexistent_source_id_is_denied(
    client: MlflowClient, monkeypatch: pytest.MonkeyPatch, source_id_key: str
):
    username, password = create_user(client.tracking_uri)

    with User(username, password, monkeypatch):
        rm = client.create_registered_model("missing-source-authz-model")

    # A nonexistent source id is denied with 403 rather than surfacing the store's 404, so
    # the response cannot be used to probe which run/model ids exist.
    response = _send_rest_tracking_post_request(
        client.tracking_uri,
        "/api/2.0/mlflow/model-versions/create",
        json_payload={"name": rm.name, "source": "s3://bucket/x", source_id_key: "missing"},
        auth=(username, password),
    )
    assert response.status_code == 403
    assert "Permission denied" in response.text


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
@pytest.mark.parametrize("source_kind", ["run", "model"])
def test_create_model_version_camelcase_alias_requires_read_on_source(
    client: MlflowClient, monkeypatch: pytest.MonkeyPatch, source_kind: str
):
    # The handler parses the body through the proto, which also accepts the camelCase
    # `runId` / `modelId` aliases. The validator must authorize those aliases against the
    # same source the handler anchors the version to; otherwise a caller without READ on
    # the source could bind a version to it and read its artifacts via their own model.
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)

    with User(username1, password1, monkeypatch):
        exp_id = client.create_experiment(f"alias-{source_kind}-authz-exp")
        if source_kind == "run":
            run = client.create_run(exp_id)
            source = run.info.artifact_uri
            alias_field = {"runId": run.info.run_id}
        else:
            model = client.create_logged_model(experiment_id=exp_id)
            source = model.artifact_location
            alias_field = {"modelId": model.model_id}

    with User(username2, password2, monkeypatch):
        rm = client.create_registered_model(f"alias-{source_kind}-authz-model")

    payload = {"name": rm.name, "source": source, **alias_field}
    response = _send_rest_tracking_post_request(
        client.tracking_uri,
        "/api/2.0/mlflow/model-versions/create",
        json_payload=payload,
        auth=(username2, password2),
    )
    assert response.status_code == 403
    assert "Permission denied" in response.text

    grant_role_permission(client.tracking_uri, username2, "experiment", exp_id, "READ")

    response = _send_rest_tracking_post_request(
        client.tracking_uri,
        "/api/2.0/mlflow/model-versions/create",
        json_payload=payload,
        auth=(username2, password2),
    )
    assert response.status_code == 200


@pytest.fixture
def metric_model_authz(client: MlflowClient, monkeypatch: pytest.MonkeyPatch):
    # user2 (the actor) owns the run but has no permission on either model's experiment:
    # user1 owns model_id1's experiment and user3 owns model_id3's.
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)
    username3, password3 = create_user(client.tracking_uri)

    with User(username1, password1, monkeypatch):
        exp_id1 = client.create_experiment("metric-authz-model-exp-1")
        model_id1 = client.create_logged_model(experiment_id=exp_id1).model_id

    with User(username3, password3, monkeypatch):
        exp_id3 = client.create_experiment("metric-authz-model-exp-3")
        model_id3 = client.create_logged_model(experiment_id=exp_id3).model_id

    with User(username2, password2, monkeypatch):
        exp_id2 = client.create_experiment("metric-authz-run-exp")
        run_id = client.create_run(exp_id2).info.run_id

    return SimpleNamespace(
        tracking_uri=client.tracking_uri,
        user2=(username2, password2),
        exp_id1=exp_id1,
        model_id1=model_id1,
        model_id3=model_id3,
        run_id=run_id,
        timestamp=int(time.time() * 1000),
    )


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_log_metric_model_id_requires_update(metric_model_authz):
    # LogMetric can route a metric to a logged model via model_id; a user with UPDATE on
    # the run but not on the model_id's experiment must be denied.
    a = metric_model_authz

    # Unauthorized top-level model_id is denied.
    response = _send_rest_tracking_post_request(
        a.tracking_uri,
        "/api/2.0/mlflow/runs/log-metric",
        json_payload={
            "run_id": a.run_id,
            "key": "metric_with_model",
            "value": 1.0,
            "timestamp": a.timestamp,
            "model_id": a.model_id1,
        },
        auth=a.user2,
    )
    assert response.status_code == 403
    assert "Permission denied" in response.text

    # The camelCase `modelId` alias is covered too.
    response = _send_rest_tracking_post_request(
        a.tracking_uri,
        "/api/2.0/mlflow/runs/log-metric",
        json_payload={
            "run_id": a.run_id,
            "key": "metric_with_camel_model",
            "value": 2.0,
            "timestamp": a.timestamp,
            "modelId": a.model_id1,
        },
        auth=a.user2,
    )
    assert response.status_code == 403
    assert "Permission denied" in response.text

    # A nonexistent model_id returns a uniform 403, not a 404 existence oracle.
    response = _send_rest_tracking_post_request(
        a.tracking_uri,
        "/api/2.0/mlflow/runs/log-metric",
        json_payload={
            "run_id": a.run_id,
            "key": "metric_bogus_model",
            "value": 13.0,
            "timestamp": a.timestamp,
            "model_id": "m-bogus-nonexistent-model-id",
        },
        auth=a.user2,
    )
    assert response.status_code == 403
    assert "Permission denied" in response.text

    # No model_id: only run UPDATE is required, so it succeeds.
    response = _send_rest_tracking_post_request(
        a.tracking_uri,
        "/api/2.0/mlflow/runs/log-metric",
        json_payload={
            "run_id": a.run_id,
            "key": "metric_no_model",
            "value": 7.0,
            "timestamp": a.timestamp,
        },
        auth=a.user2,
    )
    assert response.status_code == 200


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_log_batch_model_id_requires_update(metric_model_authz):
    # LogBatch can route per-metric metrics to logged models via nested model_id; any
    # referenced model the caller lacks UPDATE on must deny the whole batch.
    a = metric_model_authz

    # Unauthorized nested model_id is denied.
    response = _send_rest_tracking_post_request(
        a.tracking_uri,
        "/api/2.0/mlflow/runs/log-batch",
        json_payload={
            "run_id": a.run_id,
            "metrics": [
                {
                    "key": "batch_metric_with_model",
                    "value": 3.0,
                    "timestamp": a.timestamp,
                    "model_id": a.model_id1,
                }
            ],
        },
        auth=a.user2,
    )
    assert response.status_code == 403
    assert "Permission denied" in response.text

    # The camelCase `modelId` alias on a nested metric is covered too.
    response = _send_rest_tracking_post_request(
        a.tracking_uri,
        "/api/2.0/mlflow/runs/log-batch",
        json_payload={
            "run_id": a.run_id,
            "metrics": [
                {
                    "key": "batch_metric_with_camel_model",
                    "value": 4.0,
                    "timestamp": a.timestamp,
                    "modelId": a.model_id1,
                }
            ],
        },
        auth=a.user2,
    )
    assert response.status_code == 403
    assert "Permission denied" in response.text

    # A batch referencing a distinct unauthorized model is denied.
    response = _send_rest_tracking_post_request(
        a.tracking_uri,
        "/api/2.0/mlflow/runs/log-batch",
        json_payload={
            "run_id": a.run_id,
            "metrics": [
                {
                    "key": "metric1",
                    "value": 5.0,
                    "timestamp": a.timestamp,
                    "model_id": a.model_id1,
                },
                {
                    "key": "metric2",
                    "value": 6.0,
                    "timestamp": a.timestamp,
                    "model_id": a.model_id3,
                },
            ],
        },
        auth=a.user2,
    )
    assert response.status_code == 403
    assert "Permission denied" in response.text

    # No model_id on any metric: only run UPDATE is required, so it succeeds.
    response = _send_rest_tracking_post_request(
        a.tracking_uri,
        "/api/2.0/mlflow/runs/log-batch",
        json_payload={
            "run_id": a.run_id,
            "metrics": [
                {"key": "batch_metric_1", "value": 8.0, "timestamp": a.timestamp},
                {"key": "batch_metric_2", "value": 9.0, "timestamp": a.timestamp},
            ],
        },
        auth=a.user2,
    )
    assert response.status_code == 200


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_log_metric_and_batch_model_id_allowed_after_grant(metric_model_authz):
    # After user2 is granted UPDATE on model_id1's experiment (but not model_id3's),
    # metrics targeting model_id1 succeed while model_id3 stays denied.
    a = metric_model_authz
    grant_role_permission(a.tracking_uri, a.user2[0], "experiment", a.exp_id1, "EDIT")

    # Authorized model_id via LogMetric succeeds.
    response = _send_rest_tracking_post_request(
        a.tracking_uri,
        "/api/2.0/mlflow/runs/log-metric",
        json_payload={
            "run_id": a.run_id,
            "key": "metric_with_perm",
            "value": 10.0,
            "timestamp": a.timestamp,
            "model_id": a.model_id1,
        },
        auth=a.user2,
    )
    assert response.status_code == 200

    # Authorized model_id via LogBatch succeeds.
    response = _send_rest_tracking_post_request(
        a.tracking_uri,
        "/api/2.0/mlflow/runs/log-batch",
        json_payload={
            "run_id": a.run_id,
            "metrics": [
                {
                    "key": "batch_metric_with_perm",
                    "value": 11.0,
                    "timestamp": a.timestamp,
                    "model_id": a.model_id1,
                }
            ],
        },
        auth=a.user2,
    )
    assert response.status_code == 200

    # Mixed-permission batch: the authorized model is listed first, so the 403 proves
    # every distinct model_id is checked, not just the first one that passes.
    response = _send_rest_tracking_post_request(
        a.tracking_uri,
        "/api/2.0/mlflow/runs/log-batch",
        json_payload={
            "run_id": a.run_id,
            "metrics": [
                {
                    "key": "batch_authorized_model",
                    "value": 11.5,
                    "timestamp": a.timestamp,
                    "model_id": a.model_id1,
                },
                {
                    "key": "batch_unauthorized_model",
                    "value": 11.6,
                    "timestamp": a.timestamp,
                    "model_id": a.model_id3,
                },
            ],
        },
        auth=a.user2,
    )
    assert response.status_code == 403
    assert "Permission denied" in response.text

    # A model in a still-unauthorized experiment remains denied.
    response = _send_rest_tracking_post_request(
        a.tracking_uri,
        "/api/2.0/mlflow/runs/log-metric",
        json_payload={
            "run_id": a.run_id,
            "key": "metric_without_perm",
            "value": 12.0,
            "timestamp": a.timestamp,
            "model_id": a.model_id3,
        },
        auth=a.user2,
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
def test_create_logged_model_child_permission_outcomes(client: MlflowClient, monkeypatch):
    owner, owner_password = create_user(client.tracking_uri)
    no_grant, no_grant_password = create_user(client.tracking_uri)
    parent_writer, parent_writer_password = create_user(client.tracking_uri)
    child_writer, child_writer_password = create_user(client.tracking_uri)
    denied_writer, denied_writer_password = create_user(client.tracking_uri)

    with User(owner, owner_password, monkeypatch):
        experiment_id = client.create_experiment("logged-model-child-permission-outcomes")

    grant_role_permission(client.tracking_uri, parent_writer, "experiment", experiment_id, "EDIT")
    grant_role_permission(client.tracking_uri, child_writer, "experiment", experiment_id, "READ")
    grant_role_permission(client.tracking_uri, child_writer, "logged_model", "*", "EDIT")
    grant_role_permission(client.tracking_uri, denied_writer, "experiment", experiment_id, "EDIT")
    grant_role_permission(client.tracking_uri, denied_writer, "logged_model", "*", "DENY")

    for username, password in (
        (no_grant, no_grant_password),
        (denied_writer, denied_writer_password),
    ):
        with User(username, password, monkeypatch):
            with pytest.raises(MlflowException, match="Permission denied"):
                client.create_logged_model(experiment_id=experiment_id)

    for username, password in (
        (parent_writer, parent_writer_password),
        (child_writer, child_writer_password),
    ):
        with User(username, password, monkeypatch):
            model = client.create_logged_model(experiment_id=experiment_id)

        assert model.experiment_id == experiment_id


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
def test_run_child_permission_outcomes(client: MlflowClient, monkeypatch):
    owner, owner_password = create_user(client.tracking_uri)
    no_grant, no_grant_password = create_user(client.tracking_uri)
    parent_writer, parent_writer_password = create_user(client.tracking_uri)
    child_writer, child_writer_password = create_user(client.tracking_uri)
    denied_writer, denied_writer_password = create_user(client.tracking_uri)

    with User(owner, owner_password, monkeypatch):
        experiment_id = client.create_experiment("run-child-permission-outcomes")
        anchor_run = client.create_run(experiment_id)

    grant_role_permission(client.tracking_uri, parent_writer, "experiment", experiment_id, "EDIT")
    grant_role_permission(client.tracking_uri, child_writer, "experiment", experiment_id, "READ")
    grant_role_permission(client.tracking_uri, child_writer, "run", "*", "EDIT")
    grant_role_permission(client.tracking_uri, denied_writer, "experiment", experiment_id, "EDIT")
    grant_role_permission(client.tracking_uri, denied_writer, "run", "*", "DENY")

    for username, password in (
        (no_grant, no_grant_password),
        (denied_writer, denied_writer_password),
    ):
        with User(username, password, monkeypatch):
            with pytest.raises(MlflowException, match="Permission denied"):
                client.create_run(experiment_id)
            with pytest.raises(MlflowException, match="Permission denied"):
                client.log_metric(anchor_run.info.run_id, "denied_metric", 1.0)

    for username, password in (
        (parent_writer, parent_writer_password),
        (child_writer, child_writer_password),
    ):
        with User(username, password, monkeypatch):
            run = client.create_run(experiment_id)
            client.log_metric(anchor_run.info.run_id, f"metric_{username}", 1.0)
        assert run.info.experiment_id == experiment_id


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_delete_and_restore_run_honor_run_deny(client: MlflowClient, monkeypatch):
    # §8.A delete tier: DeleteRun/RestoreRun resolve run .can_delete, so (run, *, DENY)
    # blocks them even for an experiment-EDIT caller. (The run outcomes test covers
    # create/update; this covers the delete capability specifically.)
    owner, owner_pw = create_user(client.tracking_uri)
    denied, denied_pw = create_user(client.tracking_uri)
    with User(owner, owner_pw, monkeypatch):
        experiment_id = client.create_experiment("delete-run-deny")
        run = client.create_run(experiment_id)
    grant_role_permission(client.tracking_uri, denied, "experiment", experiment_id, "EDIT")
    grant_role_permission(client.tracking_uri, denied, "run", "*", "DENY")
    with User(denied, denied_pw, monkeypatch):
        with pytest.raises(MlflowException, match="Permission denied"):
            client.delete_run(run.info.run_id)
        with pytest.raises(MlflowException, match="Permission denied"):
            client.restore_run(run.info.run_id)


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_log_batch_honors_logged_model_deny(client: MlflowClient, monkeypatch):
    # §8.A footnote: LogBatch/LogMetric target the run tier AND the per-metric model_id's
    # logged_model tier (secondary touch). A (logged_model, *, DENY) must block a metric
    # that names a model_id even when the caller can write the run.
    owner, owner_pw = create_user(client.tracking_uri)
    denied, denied_pw = create_user(client.tracking_uri)
    with User(owner, owner_pw, monkeypatch):
        experiment_id = client.create_experiment("log-batch-lm-deny")
        run = client.create_run(experiment_id)
        model = client.create_logged_model(experiment_id)
    grant_role_permission(client.tracking_uri, denied, "experiment", experiment_id, "EDIT")
    grant_role_permission(client.tracking_uri, denied, "logged_model", "*", "DENY")
    with User(denied, denied_pw, monkeypatch):
        # A metric bound to the denied model_id must be rejected (logged_model DENY),
        # even though the run itself is writable.
        with pytest.raises(MlflowException, match="Permission denied"):
            client.log_metric(run.info.run_id, "m", 1.0, model_id=model.model_id)


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_delete_traces_honors_trace_deny(client: MlflowClient, monkeypatch):
    # §8.A trace delete tier: DeleteTraces resolves trace .can_delete, so (trace, *, DENY)
    # blocks it for an experiment-EDIT caller.
    import requests

    owner, owner_pw = create_user(client.tracking_uri)
    denied, denied_pw = create_user(client.tracking_uri)
    with User(owner, owner_pw, monkeypatch):
        experiment_id = client.create_experiment("delete-traces-deny")
    grant_role_permission(client.tracking_uri, denied, "experiment", experiment_id, "EDIT")
    grant_role_permission(client.tracking_uri, denied, "trace", "*", "DENY")
    resp = requests.post(
        f"{client.tracking_uri}/api/2.0/mlflow/traces/delete-traces",
        json={"experiment_id": experiment_id, "trace_ids": ["tr-1"]},
        auth=(denied, denied_pw),
    )
    assert resp.status_code == 403


# §7 Class A — static transitive sink-trace. Maps each composite/job route to its submitted
# job entrypoint and the in-scope child types its gate covers, then walks the job's transitive
# closure (within mlflow.genai) for child-mutation sink calls and asserts every reached sink's
# child type is covered. Catches a future *delegated* child write buried N hops inside a job
# (the class that produced INVOKE_ISSUE_DETECTION's assessment write and INVOKE_GENAI_EVALUATE's
# set_trace_tag) without executing the job. A new uncovered sink fails the test, forcing either
# a gate (positive/veto) or an explicit allow-list update.
_SINK_TO_CHILD = {
    "create_run": "run",
    "delete_run": "run",
    "restore_run": "run",
    "log_batch": "run",
    "log_metric": "run",
    "set_trace_tag": "trace",
    "delete_trace_tag": "trace",
    "delete_traces": "trace",
    "log_assessment": "assessment",
    "_log_assessments": "assessment",
    "create_assessment": "assessment",
    "delete_assessment": "assessment",
    "log_feedback": "assessment",
    "log_expectation": "assessment",
    "log_issue": "assessment",  # issue detection writes Issue assessments (LLM_JUDGE)
    "create_logged_model": "logged_model",
    "delete_logged_model": "logged_model",
}

# route -> (job sink modules, child types the route's gate covers). The module list is the
# curated transitive closure of child-mutation code each job reaches (handler -> job ->
# harness/pipeline). It is intentionally explicit rather than auto-walked: a static call-graph
# walk silently under-detects cross-module/aliased calls (e.g. the eval job calls
# mlflow.genai.evaluate, the discovery job calls discover_issues), which would make the guard
# pass on an empty set and give false confidence. Explicit modules fail loudly if a job grows a
# new sink-bearing module the maintainer hasn't classified here.
_COMPOSITE_JOB_COVERAGE = {
    "INVOKE_SCORER": (
        ["mlflow/genai/scorers/job.py", "mlflow/genai/evaluation/harness.py"],
        {"trace", "assessment"},
    ),
    "INVOKE_GENAI_EVALUATE": (
        ["mlflow/genai/evaluation/job.py", "mlflow/genai/evaluation/harness.py"],
        {"run", "trace", "assessment"},
    ),
    "INVOKE_ISSUE_DETECTION": (
        ["mlflow/genai/discovery/job.py", "mlflow/genai/discovery/pipeline.py"],
        {"run", "trace", "assessment"},
    ),
}


def _sink_children_in_modules(module_paths: list[str]) -> set[str]:
    """Scan the given source files for child-mutation sink calls (attribute or bare name) and
    return the set of in-scope child types written. Uses an AST call walk per file so a bare
    substring in a comment/string doesn't count.
    """
    import ast
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    children: set[str] = set()
    for rel in module_paths:
        path = repo_root / rel
        try:
            tree = ast.parse(path.read_text())
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called = None
            if isinstance(node.func, ast.Name):
                called = node.func.id
            elif isinstance(node.func, ast.Attribute):
                called = node.func.attr
            if called in _SINK_TO_CHILD:
                children.add(_SINK_TO_CHILD[called])
    return children


@pytest.mark.parametrize("route", sorted(_COMPOSITE_JOB_COVERAGE))
def test_composite_job_child_writes_are_gate_covered(route):
    module_paths, covered = _COMPOSITE_JOB_COVERAGE[route]
    written = _sink_children_in_modules(module_paths)
    # Sanity: the curated module list must actually contain sinks; an empty scan means the
    # module paths drifted (renamed/moved) and the guard has silently stopped protecting.
    assert written, (
        f"{route}: no child-write sinks found in {module_paths} — the sink-module list is "
        f"stale (files moved/renamed?). Update _COMPOSITE_JOB_COVERAGE."
    )
    uncovered = written - covered
    assert not uncovered, (
        f"{route}: job sink modules {module_paths} write in-scope child types "
        f"{sorted(uncovered)} that its gate does not cover (covers {sorted(covered)}). Add a "
        f"gate (positive or DENY veto) for the new child type, or update the coverage entry if "
        f"intentionally allowed."
    )


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

    # Non-admin reads the config (UI needs secrets_available) but the using_default_passphrase
    # server-posture signal is redacted for them.
    with User(user1, password1, monkeypatch):
        response = requests.get(
            url=client.tracking_uri + "/ajax-api/3.0/mlflow/gateway/secrets/config",
            auth=(user1, password1),
        )
        response.raise_for_status()
        body = response.json()
        assert "secrets_available" in body
        assert "using_default_passphrase" not in body

    # Admin sees the full config, including using_default_passphrase.
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        response = requests.get(
            url=client.tracking_uri + "/ajax-api/3.0/mlflow/gateway/secrets/config",
            auth=(ADMIN_USERNAME, ADMIN_PASSWORD),
        )
        response.raise_for_status()
        assert "using_default_passphrase" in response.json()

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


def _create_gateway_endpoint(tracking_uri, name, auth, workspace=None):
    headers = {"X-MLFLOW-WORKSPACE": workspace} if workspace else {}
    response = requests.post(
        url=tracking_uri + "/api/3.0/mlflow/gateway/secrets/create",
        json={
            "secret_name": f"{name}-key",
            "secret_value": {"api_key": "test-key"},
            "provider": "openai",
        },
        auth=auth,
        headers=headers,
    )
    response.raise_for_status()
    secret_id = response.json()["secret"]["secret_id"]

    response = requests.post(
        url=tracking_uri + "/api/3.0/mlflow/gateway/model-definitions/create",
        json={
            "name": f"{name}-model",
            "secret_id": secret_id,
            "provider": "openai",
            "model_name": "gpt-4o",
        },
        auth=auth,
        headers=headers,
    )
    response.raise_for_status()
    model_id = response.json()["model_definition"]["model_definition_id"]

    response = requests.post(
        url=tracking_uri + "/api/3.0/mlflow/gateway/endpoints/create",
        json={
            "name": name,
            "model_configs": [{"model_definition_id": model_id, "linkage_type": "PRIMARY"}],
            "usage_tracking": False,
        },
        auth=auth,
        headers=headers,
    )
    response.raise_for_status()
    return response.json()["endpoint"]


def test_list_models_filters_by_use_permission(fastapi_client):
    tracking_uri = fastapi_client.tracking_uri
    owner_auth = create_user(tracking_uri)
    username, password = create_user(tracking_uri)
    read_only = _create_gateway_endpoint(tracking_uri, "read-only", owner_auth)
    usable = _create_gateway_endpoint(tracking_uri, "usable", owner_auth)
    url = tracking_uri + "/gateway/mlflow/v1/models"

    assert requests.get(url).status_code == 401

    response = requests.get(url, auth=(username, password))
    assert response.status_code == 200
    assert response.json()["data"] == []

    grant_role_permission(
        tracking_uri, username, "gateway_endpoint", read_only["endpoint_id"], "READ"
    )
    grant_role_permission(tracking_uri, username, "gateway_endpoint", usable["endpoint_id"], "USE")

    response = requests.get(url, auth=(username, password))
    assert response.status_code == 200
    assert [model["id"] for model in response.json()["data"]] == ["usable"]

    # The admin owns neither endpoint and has no endpoint role grants.
    response = requests.get(url, auth=(ADMIN_USERNAME, ADMIN_PASSWORD))
    assert response.status_code == 200
    assert [model["id"] for model in response.json()["data"]] == ["read-only", "usable"]


def test_list_models_isolates_workspaces(fastapi_workspace_client):
    tracking_uri = fastapi_workspace_client.tracking_uri
    admin_auth = (ADMIN_USERNAME, ADMIN_PASSWORD)
    username, password = create_user(tracking_uri)
    requests.post(
        url=tracking_uri + "/api/3.0/mlflow/workspaces",
        json={"name": "other-team"},
        auth=admin_auth,
    ).raise_for_status()
    _create_gateway_endpoint(tracking_uri, "shared", admin_auth, DEFAULT_WORKSPACE_NAME)
    other_endpoint = _create_gateway_endpoint(tracking_uri, "shared", admin_auth, "other-team")
    grant_role_permission(
        tracking_uri,
        username,
        "gateway_endpoint",
        other_endpoint["endpoint_id"],
        "USE",
        workspace="other-team",
    )
    url = tracking_uri + "/gateway/mlflow/v1/models"

    response = requests.get(
        url,
        auth=(username, password),
        headers={"X-MLFLOW-WORKSPACE": DEFAULT_WORKSPACE_NAME},
    )
    assert response.status_code == 200
    assert response.json()["data"] == []

    for credentials in ((username, password), admin_auth):
        response = requests.get(url, auth=credentials, headers={"X-MLFLOW-WORKSPACE": "other-team"})
        assert response.status_code == 200
        assert [model["id"] for model in response.json()["data"]] == ["shared"]


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


def test_gateway_alias_spellings_cannot_bypass_secret_authorization(client, monkeypatch):
    # Protobuf JSON accepts both `secret_id` and `secretId`, last key wins. Authorization
    # must not read one spelling while the handler acts on the other (GHSA-3g8m-hm3x-gh2r).
    user1, password1 = create_user(client.tracking_uri)
    user2, password2 = create_user(client.tracking_uri)

    def create_secret(name, user, password):
        response = requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/secrets/create",
            json={
                "secret_name": name,
                "secret_value": {"api_key": f"{name}-key"},
                "provider": "openai",
                "auth_config": {"api_base": "https://api.openai.com/v1"},
            },
            auth=(user, password),
        )
        response.raise_for_status()
        return response.json()["secret"]["secret_id"]

    with User(user1, password1, monkeypatch):
        victim_secret_id = create_secret("victim_secret", user1, password1)
        response = requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/model-definitions/create",
            json={
                "name": "victim_model",
                "secret_id": victim_secret_id,
                "provider": "openai",
                "model_name": "gpt-4",
            },
            auth=(user1, password1),
        )
        response.raise_for_status()
        victim_model_def_id = response.json()["model_definition"]["model_definition_id"]

    with User(user2, password2, monkeypatch):
        attacker_secret_id = create_secret("attacker_secret", user2, password2)

        # Both spellings in one body are ambiguous and refused, in either key order.
        for both in (
            {"secret_id": attacker_secret_id, "secretId": victim_secret_id},
            {"secretId": victim_secret_id, "secret_id": attacker_secret_id},
        ):
            response = requests.post(
                url=client.tracking_uri + "/api/3.0/mlflow/gateway/secrets/update",
                json={**both, "auth_config": {"api_base": "https://attacker.example/v1"}},
                auth=(user2, password2),
            )
            assert response.status_code == 400
            assert "both 'secret_id' and 'secretId'" in response.json()["message"]

            response = requests.post(
                url=client.tracking_uri + "/api/3.0/mlflow/gateway/model-definitions/create",
                json={"name": "m", **both, "provider": "openai", "model_name": "gpt-4"},
                auth=(user2, password2),
            )
            assert response.status_code == 400
            assert "both 'secret_id' and 'secretId'" in response.json()["message"]

        # The camelCase spelling alone is authorized against the secret it names.
        response = requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/model-definitions/create",
            json={
                "name": "m",
                "secretId": victim_secret_id,
                "provider": "openai",
                "model_name": "gpt-4",
            },
            auth=(user2, password2),
        )
        assert response.status_code == 403

        response = requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/model-definitions/create",
            json={
                "name": "attacker_model",
                "secret_id": attacker_secret_id,
                "provider": "openai",
                "model_name": "gpt-4",
            },
            auth=(user2, password2),
        )
        response.raise_for_status()
        attacker_model_def_id = response.json()["model_definition"]["model_definition_id"]

        response = requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/model-definitions/update",
            json={"model_definition_id": attacker_model_def_id, "secretId": victim_secret_id},
            auth=(user2, password2),
        )
        assert response.status_code == 403

        # Nested model configs follow the same rules on endpoint creation.
        response = requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/endpoints/create",
            json={
                "name": "e",
                "modelConfigs": [
                    {"modelDefinitionId": victim_model_def_id, "linkageType": "PRIMARY"}
                ],
            },
            auth=(user2, password2),
        )
        assert response.status_code == 403

        # A conflicting inner spelling is refused under either spelling of the outer key.
        for model_configs_key in ("model_configs", "modelConfigs"):
            response = requests.post(
                url=client.tracking_uri + "/api/3.0/mlflow/gateway/endpoints/create",
                json={
                    "name": "e",
                    model_configs_key: [
                        {
                            "model_definition_id": attacker_model_def_id,
                            "modelDefinitionId": victim_model_def_id,
                            "linkage_type": "PRIMARY",
                        }
                    ],
                },
                auth=(user2, password2),
            )
            assert response.status_code == 400
            assert (
                "both 'model_definition_id' and 'modelDefinitionId'" in response.json()["message"]
            )

        response = requests.post(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/endpoints/create",
            json={
                "name": "attacker_endpoint",
                "model_configs": [
                    {"model_definition_id": attacker_model_def_id, "linkage_type": "PRIMARY"}
                ],
            },
            auth=(user2, password2),
        )
        response.raise_for_status()
        attacker_endpoint_id = response.json()["endpoint"]["endpoint_id"]

        # Attaching a model requires USE on the model definition, whichever spelling names it.
        for model_config_key in ("model_config", "modelConfig"):
            response = requests.post(
                url=client.tracking_uri + "/api/3.0/mlflow/gateway/endpoints/models/attach",
                json={
                    "endpoint_id": attacker_endpoint_id,
                    model_config_key: {
                        "model_definition_id": victim_model_def_id,
                        "linkage_type": "FALLBACK",
                    },
                },
                auth=(user2, password2),
            )
            assert response.status_code == 403

        for model_config_key in ("model_config", "modelConfig"):
            response = requests.post(
                url=client.tracking_uri + "/api/3.0/mlflow/gateway/endpoints/models/attach",
                json={
                    "endpoint_id": attacker_endpoint_id,
                    model_config_key: {
                        "model_definition_id": attacker_model_def_id,
                        "modelDefinitionId": victim_model_def_id,
                        "linkage_type": "FALLBACK",
                    },
                },
                auth=(user2, password2),
            )
            assert response.status_code == 400
            assert (
                "both 'model_definition_id' and 'modelDefinitionId'" in response.json()["message"]
            )

    # The victim secret still points at its original provider.
    with User(user1, password1, monkeypatch):
        response = requests.get(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/secrets/get",
            params={"secret_id": victim_secret_id},
            auth=(user1, password1),
        )
        response.raise_for_status()
        assert response.json()["secret"]["auth_config"] == {"api_base": "https://api.openai.com/v1"}

    # Cleanup
    with User(user2, password2, monkeypatch):
        requests.delete(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/endpoints/delete",
            json={"endpoint_id": attacker_endpoint_id},
            auth=(user2, password2),
        ).raise_for_status()
        requests.delete(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/model-definitions/delete",
            json={"model_definition_id": attacker_model_def_id},
            auth=(user2, password2),
        ).raise_for_status()
        requests.delete(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/secrets/delete",
            json={"secret_id": attacker_secret_id},
            auth=(user2, password2),
        ).raise_for_status()
    with User(user1, password1, monkeypatch):
        requests.delete(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/model-definitions/delete",
            json={"model_definition_id": victim_model_def_id},
            auth=(user1, password1),
        ).raise_for_status()
        requests.delete(
            url=client.tracking_uri + "/api/3.0/mlflow/gateway/secrets/delete",
            json={"secret_id": victim_secret_id},
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

    job_payload = {
        "experiment_id": experiment_id,
        "source_prompt_uri": "prompts:/test/1",
        "config": {
            "optimizer_type": 1,  # GEPA
            "scorers": ["Correctness"],
        },
    }

    # Experiment EDIT alone is still not enough: the job WRITES a new version of the
    # source prompt, and that prompt does not even exist yet -- fail closed.
    response = requests.post(
        url=client.tracking_uri + "/api/3.0/mlflow/prompt-optimization/jobs",
        json=job_payload,
        auth=(user2, password2),
    )
    assert response.status_code == 403

    # Register the source prompt; without a prompt grant the version tier still resolves
    # below can_update for user2 (default floor is READ), so the job stays denied.
    with User(user1, password1, monkeypatch):
        client.register_prompt(name="test", template="Say hello to {{name}}")
    response = requests.post(
        url=client.tracking_uri + "/api/3.0/mlflow/prompt-optimization/jobs",
        json=job_payload,
        auth=(user2, password2),
    )
    assert response.status_code == 403

    # With prompt EDIT (version tier resolves through the prompt-parent fallback), the
    # permission gate passes. The request may still fail for other reasons, but not
    # with 403.
    grant_role_permission(client.tracking_uri, user2, "prompt", "test", "EDIT")
    response = requests.post(
        url=client.tracking_uri + "/api/3.0/mlflow/prompt-optimization/jobs",
        json=job_payload,
        auth=(user2, password2),
    )
    # Should not be 403 (permission denied)
    assert response.status_code != 403

    # A dataset_id the caller can't read -- here a NONEXISTENT one, which fails closed --
    # is denied at the auth gate before the handler loads it (review finding: the job
    # reads the dataset, so the direct dataset routes' READ check applies).
    response = requests.post(
        url=client.tracking_uri + "/api/3.0/mlflow/prompt-optimization/jobs",
        json={**job_payload, "config": {**job_payload["config"], "dataset_id": "no-such-ds"}},
        auth=(user2, password2),
    )
    assert response.status_code == 403


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


def test_job_search_only_returns_callers_jobs(fastapi_client, tmp_path):
    user1, password1 = create_user(fastapi_client.tracking_uri)
    user2, password2 = create_user(fastapi_client.tracking_uri)

    # Seed jobs straight into the server's SQLite backend (same file the fixture points the
    # server at). Submitting over HTTP would need job execution enabled plus an allowlisted job
    # function, and could never produce the creator-less legacy row this test covers. Each
    # create_job commits on session exit, and the job system already relies on the server and
    # the huey worker subprocess sharing this file, so committed rows are visible server-side.
    db_path = tmp_path.joinpath("sqlalchemy.db").as_uri()
    backend_uri = ("sqlite://" if is_windows() else "sqlite:////") + db_path[len("file://") :]
    job_store = SqlAlchemyJobStore(backend_uri)
    job1 = job_store.create_job("fn", json.dumps({"owner": user1}), creator=user1)
    job2 = job_store.create_job("fn", json.dumps({"owner": user2}), creator=user2)
    legacy_job = job_store.create_job("fn", json.dumps({"owner": "legacy"}), creator=None)

    def search(auth):
        response = requests.post(
            url=fastapi_client.tracking_uri + "/ajax-api/3.0/jobs/search",
            json={},
            auth=auth,
        )
        response.raise_for_status()
        return {job["job_id"]: job for job in response.json()["jobs"]}

    user1_jobs = search((user1, password1))
    assert set(user1_jobs) == {job1.job_id}
    assert user1_jobs[job1.job_id]["params"] == {"owner": user1}
    assert user1_jobs[job1.job_id]["creator"] == user1

    assert set(search((user2, password2))) == {job2.job_id}

    # Admins keep the unfiltered listing, including jobs with no recorded creator.
    admin_jobs = search((ADMIN_USERNAME, ADMIN_PASSWORD))
    assert {job1.job_id, job2.job_id, legacy_job.job_id} <= set(admin_jobs)


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


def test_gateway_auth_header_honored_under_static_prefix(
    mock_auth_store, mock_auth_config, monkeypatch
):
    monkeypatch.delenv(_MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN.name, raising=False)
    monkeypatch.setenv(STATIC_PREFIX_ENV_VAR, "/myprefix")
    credentials = base64.b64encode(b"alice:password123").decode("ascii")
    request = _make_request(
        "/myprefix/gateway/proxy/my-endpoint/v1/responses",
        authorization="Bearer sk-provider-key",
        mlflow_authorization=f"Basic {credentials}",
    )

    user = _authenticate_fastapi_request(request)

    assert user.username == "alice"
    mock_auth_store.authenticate_user.assert_called_once_with("alice", "password123")


def test_gateway_internal_token_not_honored_on_non_gateway_prefixed_route(
    mock_auth_store, mock_auth_config, monkeypatch
):
    # The internal token must not become a master password for non-gateway routes.
    monkeypatch.setenv(_MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN.name, "internal-secret")
    monkeypatch.setenv(STATIC_PREFIX_ENV_VAR, "/myprefix")
    mock_auth_store.authenticate_user.return_value = False
    credentials = base64.b64encode(b"alice:internal-secret").decode("ascii")
    request = _make_request(
        "/myprefix/ajax-api/3.0/jobs/search",
        authorization=f"Basic {credentials}",
    )

    user = _authenticate_fastapi_request(request)

    assert user is None
    mock_auth_store.get_user.assert_not_called()
    mock_auth_store.authenticate_user.assert_called_once_with("alice", "internal-secret")


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


@pytest.mark.parametrize("is_admin", [True, False])
def test_flask_basic_auth_rejects_legacy_default_password_for_admins(
    mock_auth_store, mock_auth_config, monkeypatch, caplog, is_admin
):
    monkeypatch.delenv(_MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN.name, raising=False)
    mock_auth_store.get_user.side_effect = lambda username: mock.Mock(
        username=username, is_admin=is_admin
    )
    fake_flask_request = mock.Mock()
    fake_flask_request.authorization.username = "admin"
    fake_flask_request.authorization.password = "password1234"
    challenge = object()

    with (
        mock.patch("mlflow.server.auth._USER_AUTH_CACHE", None),
        mock.patch("mlflow.server.auth.request", fake_flask_request),
        mock.patch("mlflow.server.auth.make_basic_auth_response", return_value=challenge),
        caplog.at_level(logging.WARNING, logger=auth_module.__name__),
    ):
        result = auth_module.authenticate_request_basic_auth()

    mock_auth_store.authenticate_user.assert_called_once_with("admin", "password1234")
    mock_auth_store.get_user.assert_called_once_with("admin")
    rejected = [r for r in caplog.records if "Rejected a login by admin user 'admin'" in r.message]
    if is_admin:
        assert result is challenge
        assert len(rejected) == 1
    else:
        assert result is fake_flask_request.authorization
        assert not rejected


@pytest.mark.parametrize("cache_enabled", [True, False])
def test_fastapi_basic_auth_rejects_legacy_default_password_for_admins(
    mock_auth_store, mock_auth_config, monkeypatch, caplog, cache_enabled
):
    monkeypatch.delenv(_MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN.name, raising=False)
    mock_auth_store.get_user.side_effect = lambda username: mock.Mock(
        username=username, is_admin=True
    )
    credentials = base64.b64encode(b"admin:password1234").decode("ascii")
    cache = TTLCache(maxsize=10, ttl=60) if cache_enabled else None

    with (
        mock.patch("mlflow.server.auth._USER_AUTH_CACHE", cache),
        caplog.at_level(logging.WARNING, logger=auth_module.__name__),
    ):
        assert _authenticate_fastapi_request(_make_request("/x", f"Basic {credentials}")) is None

    mock_auth_store.authenticate_user.assert_called_once_with("admin", "password1234")
    assert any("Rejected a login by admin user 'admin'" in r.message for r in caplog.records)
    if cache_enabled:
        # The rejected credential must not be cached as valid.
        assert auth_module._auth_cache_key("admin", "password1234") not in cache


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
    validator = _find_fastapi_validator(path, "GET")
    assert validator is not None


@pytest.mark.parametrize(
    "path",
    [f"{prefix}{sub}" for prefix in (_MCP_AJAX_PREFIX, _MCP_REST_PREFIX) for sub in _MCP_SUBPATHS],
)
def test_mcp_server_routes_return_validator_with_custom_auth(path):
    with mock.patch("mlflow.server.auth.auth_config") as cfg:
        cfg.authorization_function = "custom_auth:authorize"
        validator = _find_fastapi_validator(path, "GET")
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
    validator = _find_fastapi_validator(prefix, "POST")
    assert validator is not None

    auth_store.engine.dispose()


def test_validate_can_create_mcp_server_delegates_to_shared_helper():
    # The validator gates on the workspace-create helper AND a (mcp_server, *, DENY) veto;
    # with no DENY the veto is a no-op and the workspace helper decides.
    with (
        mock.patch.object(
            auth_module, "_can_create_in_workspace", return_value=True
        ) as mock_helper,
        mock.patch.object(auth_module, "_top_level_create_denied", return_value=False),
    ):
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

        def list_role_grants_for_user_in_workspace(
            self, user_id, workspace, resource_type, parent_type=None
        ):
            return []

    monkeypatch.setattr(auth_module, "store", DummyStore(), raising=False)

    with workspace_context.WorkspaceContext(default_workspace):
        predicate = auth_module._role_based_read_predicate("alice", resource_type)
        assert predicate(resource_id) is True


def test_child_grant_read_write_agree(monkeypatch, tmp_path):
    """Child EDIT grants enable matching read/write decisions while a child DENY
    blocks both paths without changing the parent experiment permission.
    """
    from mlflow.server.auth.permissions import DENY, EDIT

    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")
    monkeypatch.setattr(
        auth_module,
        "auth_config",
        auth_module.auth_config._replace(default_permission=NO_PERMISSIONS.name),
    )
    store = SqlAlchemyStore()
    store.init_db(f"sqlite:///{tmp_path / 'agree.db'}")
    monkeypatch.setattr(auth_module, "store", store, raising=False)

    ws = "default"  # DEFAULT_WORKSPACE_NAME when workspaces are disabled
    user = store.create_user("scientist", "supersecurepassword", is_admin=False)
    role = store.create_role(name="r", workspace=ws)
    store.add_role_permission(role.id, "experiment", "*", READ.name)
    for child_type in ("run", "assessment", "logged_model", "review_queue"):
        store.add_role_permission(role.id, child_type, "*", EDIT.name)
    store.add_role_permission(role.id, "trace", "*", DENY.name)
    store.assign_role_to_user(user.id, role.id)

    def write(child_type, child_id):
        return store.get_role_permission_for_resource(
            user.id, child_type, child_id, ws, parent_type="experiment", parent_id="e1"
        )

    def can_read(child_type, child_id):
        return auth_module._role_based_read_predicate(
            "scientist", child_type, parent_type="experiment"
        )(child_id)

    for child_type, child_id in (
        ("run", "r1"),
        ("assessment", "a1"),
        ("logged_model", "m1"),
        ("review_queue", "q1"),
    ):
        assert write(child_type, child_id).can_update
        assert write(child_type, child_id).can_read
        assert can_read(child_type, child_id)

    assert write("trace", "t1").name == DENY.name
    assert not write("trace", "t1").can_read
    assert not can_read("trace", "t1")
    assert not write("run", "r1").can_manage
    assert can_read("experiment", "e1")


def test_version_grant_read_write_agree(monkeypatch, tmp_path):
    from mlflow.server.auth.permissions import EDIT

    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")
    monkeypatch.setattr(
        auth_module,
        "auth_config",
        auth_module.auth_config._replace(default_permission=NO_PERMISSIONS.name),
    )
    store = SqlAlchemyStore()
    store.init_db(f"sqlite:///{tmp_path / 'version-agree.db'}")
    monkeypatch.setattr(auth_module, "store", store, raising=False)

    user = store.create_user("version-writer", "supersecurepassword", is_admin=False)
    role = store.create_role(name="versions", workspace="default")
    grants = (
        ("registered_model", "model-a", "registered_model_version"),
        ("prompt", "prompt-a", "prompt_version"),
        ("mcp_server", "namespace/server-a", "mcp_server_version"),
    )
    for parent_type, parent_id, child_type in grants:
        store.add_role_permission(role.id, parent_type, parent_id, READ.name)
        store.add_role_permission(role.id, child_type, "*", EDIT.name)
    store.assign_role_to_user(user.id, role.id)

    for parent_type, parent_id, child_type in grants:
        write = store.get_role_permission_for_resource(
            user.id,
            child_type,
            "v1",
            "default",
            parent_type=parent_type,
            parent_id=parent_id,
        )
        read = auth_module._role_based_read_predicate(
            "version-writer", child_type, parent_type=parent_type
        )(parent_id)
        assert write.can_update
        assert write.can_read
        assert read


@pytest.mark.parametrize("resource_type", ["registered_model", "prompt", "scorer", "mcp_server"])
def test_top_level_deny_blocks_rfc_parent_types(monkeypatch, tmp_path, resource_type):
    # DENY is grantable on the RFC's top-level parent types (the sub-resource parents), not
    # only the child tiers. A (parent_type, *, DENY) must block read/update/delete/manage and
    # empty search for that type -- verified via the store fold (all .can_* False) and the
    # read predicate that backs the list/search filters. Workspace-admin still bypasses.
    from mlflow.server.auth.permissions import DENY, MANAGE

    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")
    monkeypatch.setattr(
        auth_module,
        "auth_config",
        auth_module.auth_config._replace(default_permission=NO_PERMISSIONS.name),
    )
    store = SqlAlchemyStore()
    store.init_db(f"sqlite:///{tmp_path / f'top-deny-{resource_type}.db'}")
    monkeypatch.setattr(auth_module, "store", store, raising=False)

    ws = "default"  # DEFAULT_WORKSPACE_NAME when workspaces are disabled
    user = store.create_user("denied-user", "supersecurepassword", is_admin=False)
    role = store.create_role(name="deny-role", workspace=ws)
    store.add_role_permission(role.id, resource_type, "*", DENY.name)
    store.assign_role_to_user(user.id, role.id)

    # Single-tier fold resolves to DENY -> every capability is False, so read/update/delete/
    # manage validators (which call .can_read/.can_update/.can_delete/.can_manage) all block.
    perm = store.get_role_permission_for_resource(user.id, resource_type, "res-1", ws)
    assert perm is not None
    assert perm.name == DENY.name
    assert not perm.can_read
    assert not perm.can_update
    assert not perm.can_delete
    assert not perm.can_manage

    # The read predicate backing the list/search filters drops the DENY'd row (wildcard DENY
    # empties the list of that type).
    assert auth_module._role_based_read_predicate("denied-user", resource_type)("res-1") is False

    # Workspace-admin bypass beats a top-level DENY (matches child semantics).
    admin = store.create_user("ws-admin", "supersecurepassword", is_admin=False)
    admin_role = store.create_role(name="ws-admin-role", workspace=ws)
    store.add_role_permission(admin_role.id, "workspace", "*", MANAGE.name)
    store.add_role_permission(admin_role.id, resource_type, "*", DENY.name)
    store.assign_role_to_user(admin.id, admin_role.id)
    admin_perm = store.get_role_permission_for_resource(admin.id, resource_type, "res-1", ws)
    assert admin_perm is not None
    assert admin_perm.can_manage  # admin bypass wins over DENY


@pytest.mark.parametrize(
    "resource_type", ["experiment", "registered_model", "prompt", "mcp_server"]
)
def test_top_level_create_denied_by_self_type_deny(monkeypatch, tmp_path, resource_type):
    # Mirror of the sub-resource rule for parents: workspace USE/EDIT allows creating a
    # top-level resource, but a (type, *, DENY) prevents its creation. With workspaces
    # disabled, create rights are implicit, so the veto is what blocks. Verified via the
    # shared _top_level_create_denied helper (the create validators call it after the
    # workspace-create gate).
    from mlflow.server.auth.permissions import DENY

    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")
    store = SqlAlchemyStore()
    store.init_db(f"sqlite:///{tmp_path / f'create-deny-{resource_type}.db'}")
    monkeypatch.setattr(auth_module, "store", store, raising=False)

    ws = "default"
    user = store.create_user("creator", "supersecurepassword", is_admin=False)
    monkeypatch.setattr(
        auth_module, "authenticate_request", lambda: SimpleNamespace(username="creator")
    )

    # No DENY -> workspaces disabled means create is allowed (baseline, unchanged behavior).
    assert auth_module._top_level_create_denied(resource_type, "creator") is False

    # Add (type, *, DENY) -> creation is now vetoed.
    role = store.create_role(name=f"deny-create-{resource_type}", workspace=ws)
    store.add_role_permission(role.id, resource_type, "*", DENY.name)
    store.assign_role_to_user(user.id, role.id)
    assert auth_module._top_level_create_denied(resource_type, "creator") is True


@pytest.mark.parametrize(
    "endpoint_fn",
    [search_mcp_servers, search_all_access_endpoints],
)
def test_response_filter_matches_endpoint_functions(endpoint_fn):
    request = SimpleNamespace(scope={"endpoint": endpoint_fn})
    assert _find_fastapi_response_filter(request) is not None


def test_response_filter_stamps_allowed_actions_on_single_server_get(monkeypatch):
    request = SimpleNamespace(scope={"endpoint": get_mcp_server})
    handler = _find_fastapi_response_filter(request)
    assert handler is not None
    monkeypatch.setattr(
        auth_module,
        "_get_mcp_server_permission",
        lambda name, username: READ,
    )
    monkeypatch.setattr(
        auth_module,
        "_get_mcp_server_version_permission",
        lambda name, username: READ,
    )
    request = SimpleNamespace()
    body = json.dumps({"name": "com.test/server"}).encode()
    result = json.loads(handler("testuser", body, request))
    assert result["name"] == "com.test/server"
    assert result["allowed_actions"] == []


def test_response_filter_skips_sub_resource_endpoints():
    for endpoint_fn in (get_mcp_server_version, search_mcp_server_versions):
        request = SimpleNamespace(scope={"endpoint": endpoint_fn})
        assert _find_fastapi_response_filter(request) is None


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


@pytest.mark.parametrize(
    ("parent_can_delete", "version_can_delete", "expected"),
    [(True, True, True), (True, False, False), (False, True, False)],
)
def test_whole_parent_delete_requires_version_tier(
    monkeypatch, parent_can_delete, version_can_delete, expected
):
    # Deleting a scorer (no version in the body) or a registered model/prompt cascades
    # every version, so the version child tier's can_delete is required atop the parent
    # delete -- a child DENY or lower child grant that blocks deleting one version must
    # not be bypassed by deleting the parent (review finding).
    parent_perm = SimpleNamespace(can_delete=parent_can_delete)
    version_perm = SimpleNamespace(can_delete=version_can_delete)

    # Scorer whole-parent form (no "version" in the body).
    monkeypatch.setattr(auth_module, "_get_permission_from_scorer_name", lambda: parent_perm)
    monkeypatch.setattr(auth_module, "_get_scorer_version_permission", lambda _e, _n: version_perm)
    monkeypatch.setattr(
        auth_module,
        "_get_request_param",
        lambda name: {"experiment_id": "e1", "name": "s"}[name],
    )
    with auth_module.app.test_request_context("/x", method="POST", json={}):
        assert auth_module.validate_can_delete_scorer_version() is expected

    # Registered model / prompt form: exercise the LIVE shared validator that
    # BEFORE_REQUEST_HANDLERS actually maps for DeleteRegisteredModel (review finding:
    # the composite check was first added to an unmapped helper, so no route ran it).
    from mlflow.protos.model_registry_pb2 import DeleteRegisteredModel

    assert (
        auth_module.BEFORE_REQUEST_HANDLERS[DeleteRegisteredModel]
        is auth_module._validate_can_delete_registered_model_or_prompt
    )
    monkeypatch.setattr(
        auth_module, "_get_permission_from_registered_model_or_prompt_name", lambda: parent_perm
    )
    monkeypatch.setattr(
        auth_module,
        "_get_model_version_permission_from_registered_model_or_prompt_name",
        lambda: version_perm,
    )
    assert auth_module._validate_can_delete_registered_model_or_prompt() is expected
    # The type-specific spelling delegates to the same live validator (no drift).
    assert auth_module.validate_can_delete_registered_model() is expected


@pytest.mark.parametrize("prefix", [_MCP_AJAX_PREFIX, _MCP_REST_PREFIX])
def test_mcp_server_parent_delete_requires_version_tier(monkeypatch, prefix):
    # DELETE on the parent server cascades remaining version rows (delete-orphan), so the
    # version child tier must also allow delete (review finding).
    validator = _find_fastapi_validator(f"{prefix}/com.test/cascade-server", "DELETE")
    assert validator is not None
    monkeypatch.setattr(
        auth_module,
        "_get_mcp_server_permission",
        lambda _n, _u: SimpleNamespace(can_delete=True),
    )
    version_perm = SimpleNamespace(can_delete=False)
    monkeypatch.setattr(
        auth_module, "_get_mcp_server_version_permission", lambda _n, _u: version_perm
    )
    request = SimpleNamespace(method="DELETE", state=SimpleNamespace())
    assert asyncio.run(validator("alice", request)) is False
    version_perm.can_delete = True
    assert asyncio.run(validator("alice", request)) is True


@pytest.mark.parametrize("prefix", [_MCP_AJAX_PREFIX, _MCP_REST_PREFIX])
def test_version_create_validator_stores_live_update_recheck(monkeypatch, prefix):
    validator = _find_fastapi_validator(f"{prefix}/com.test/race-server/versions", "POST")
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
            name="EDIT",
            can_read=False,
            can_update=store.exists,
            can_delete=False,
        )
    )
    child_grant = mock.Mock(return_value=None)
    monkeypatch.setattr(auth_module, "_get_tracking_store", lambda: store)
    monkeypatch.setattr(auth_module, "_get_mcp_server_version_permission", permission_helper)
    monkeypatch.setattr(auth_module, "validate_can_create_mcp_server", lambda username: True)
    monkeypatch.setattr(auth_module, "_wildcard_grant_in_request_workspace", child_grant)

    request = SimpleNamespace(method="POST", state=SimpleNamespace())
    assert asyncio.run(validator("alice", request)) is True
    assert request.state.mcp_server_parent_auto_created is True
    # On the auto-create path the version child tier resolves via the workspace wildcard
    # grant (the parent doesn't exist, so the per-server permission helper can't resolve
    # its workspace and is NOT consulted); with no child grant the create gate governs.
    child_grant.assert_called_once_with("mcp_server_version", "alice")
    assert permission_helper.call_count == 0

    store.exists = True
    assert request.state.mcp_server_can_update_existing_recheck() is True
    assert permission_helper.call_count == 1
    permission_helper.assert_called_with("com.test/race-server", "alice")


@pytest.mark.parametrize("prefix", [_MCP_AJAX_PREFIX, _MCP_REST_PREFIX])
@pytest.mark.parametrize(
    ("_case", "child_grant", "expected"),
    [
        ("no_child_grant", None, True),
        ("child_deny", "DENY", False),
        ("child_read", "READ", False),
        ("child_use", "USE", False),
        ("child_edit", "EDIT", True),
    ],
)
def test_version_create_parent_missing_child_grant_outcomes(
    monkeypatch, prefix, _case, child_grant, expected
):
    # Auto-creating the server on first version write must resolve the version child tier
    # exactly like the existing-parent path (review finding): a matching wildcard
    # (mcp_server_version, *) grant is AUTHORITATIVE -- DENY vetoes, a lower positive grant
    # (READ/USE, floored by default READ) lacks can_update and vetoes, EDIT allows -- and
    # only with NO child grant does the workspace create gate alone govern.
    from mlflow.server.auth.permissions import get_permission

    validator = _find_fastapi_validator(f"{prefix}/com.test/deny-server/versions", "POST")
    assert validator is not None

    def _get_mcp_server(_name):
        raise MlflowException("not found", error_code=RESOURCE_DOES_NOT_EXIST)

    monkeypatch.setattr(
        auth_module, "_get_tracking_store", lambda: SimpleNamespace(get_mcp_server=_get_mcp_server)
    )
    monkeypatch.setattr(
        auth_module,
        "auth_config",
        auth_module.auth_config._replace(default_permission="READ"),
    )
    grant = mock.Mock(return_value=None if child_grant is None else get_permission(child_grant))
    monkeypatch.setattr(auth_module, "_wildcard_grant_in_request_workspace", grant)
    # Workspace create gate would otherwise allow the implicit parent.
    monkeypatch.setattr(auth_module, "validate_can_create_mcp_server", lambda username: True)

    request = SimpleNamespace(method="POST", state=SimpleNamespace())
    assert asyncio.run(validator("alice", request)) is expected
    grant.assert_called_once_with("mcp_server_version", "alice")


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
def test_mcp_server_patch_connect_options_allowed_for_edit(fastapi_client, monkeypatch, prefix):
    owner, owner_pw = create_user(fastapi_client.tracking_uri)
    editor, editor_pw = create_user(fastapi_client.tracking_uri)
    reader, reader_pw = create_user(fastapi_client.tracking_uri)
    server_name = "com.test/connect-opts"

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
    grant_role_permission(
        fastapi_client.tracking_uri,
        reader,
        "mcp_server",
        server_name,
        "READ",
    )

    # EDIT user can PATCH connect_options
    with User(editor, editor_pw, monkeypatch):
        resp = requests.patch(
            url=f"{fastapi_client.tracking_uri}{prefix}/{server_name}/versions/1.0.0",
            json={"connect_options": {"npm:foo": {"hidden": True}}},
            auth=(editor, editor_pw),
        )
        assert resp.status_code == 200

    # READ user cannot PATCH connect_options
    with User(reader, reader_pw, monkeypatch):
        resp = requests.patch(
            url=f"{fastapi_client.tracking_uri}{prefix}/{server_name}/versions/1.0.0",
            json={"connect_options": {"npm:bar": {"hidden": True}}},
            auth=(reader, reader_pw),
        )
        assert resp.status_code == 403

    with User(owner, owner_pw, monkeypatch):
        resp = requests.get(
            url=f"{fastapi_client.tracking_uri}{prefix}/{server_name}/versions/1.0.0",
            auth=(owner, owner_pw),
        )
        assert resp.status_code == 200
        assert resp.json()["connect_options"] == {"npm:foo": {"hidden": True}}


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
        ver_resp = requests.post(
            url=f"{fastapi_client.tracking_uri}{prefix}/{name}/versions",
            json={**_version_create_body(name), "status": "active"},
            auth=admin_auth,
        )
        ver_resp.raise_for_status()
        version = ver_resp.json()["version"]
        requests.post(
            url=f"{fastapi_client.tracking_uri}{prefix}/{name}/endpoints",
            json={"server_version": version, "url": f"https://example.com/{name}"},
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

    # Reader sees only servers with an explicit READ grant (all are available
    # because they have active versions with endpoints).
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
        ver_resp = requests.post(
            url=f"{fastapi_client.tracking_uri}{prefix}/{name}/versions",
            json={**_version_create_body(name), "status": "active"},
            auth=admin_auth,
        )
        ver_resp.raise_for_status()
        version = ver_resp.json()["version"]
        requests.post(
            url=f"{fastapi_client.tracking_uri}{prefix}/{name}/endpoints",
            json={"server_version": version, "url": f"https://example.com/{name}"},
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


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_evaluation_dataset_and_issue_apis_require_experiment_permission(client):
    # A NO_PERMISSIONS user must not read/tamper/enumerate/delete another user's
    # datasets or issues; both are gated on the associated experiment's permission.
    base = client.tracking_uri
    owner, owner_pw = create_user(base)
    attacker, attacker_pw = create_user(base)
    owner_auth = (owner, owner_pw)
    attacker_auth = (attacker, attacker_pw)

    def post(path, auth, body):
        return requests.post(f"{base}{path}", json=body, auth=auth)

    # Owner creates an experiment (gaining MANAGE), a dataset with a record, and an issue.
    exp_id = post("/api/2.0/mlflow/experiments/create", owner_auth, {"name": "owner-exp"}).json()[
        "experiment_id"
    ]
    dataset_id = post(
        "/api/3.0/mlflow/datasets/create",
        owner_auth,
        {"name": "owner-ds", "experiment_ids": [exp_id]},
    ).json()["dataset"]["dataset_id"]
    seed = post(
        f"/api/3.0/mlflow/datasets/{dataset_id}/records",
        owner_auth,
        {"records": json.dumps([{"inputs": {"q": "secret"}, "expectations": {"a": "truth"}}])},
    )
    assert seed.status_code == 200
    issue_id = post(
        "/api/3.0/mlflow/issues",
        owner_auth,
        {"experiment_id": exp_id, "name": "sec", "description": "confidential"},
    ).json()["issue"]["issue_id"]

    # Control: the attacker genuinely lacks access to the experiment.
    assert (
        requests.get(
            f"{base}/api/2.0/mlflow/experiments/get",
            params={"experiment_id": exp_id},
            auth=attacker_auth,
        ).status_code
        == 403
    )

    # Dataset reads and unscoped enumeration are denied.
    assert (
        requests.get(f"{base}/api/3.0/mlflow/datasets/{dataset_id}", auth=attacker_auth).status_code
        == 403
    )
    assert (
        requests.get(
            f"{base}/api/3.0/mlflow/datasets/{dataset_id}/records", auth=attacker_auth
        ).status_code
        == 403
    )
    assert post("/api/3.0/mlflow/datasets/search", attacker_auth, {}).status_code == 403

    # Dataset writes and deletes are denied.
    assert (
        post(
            f"/api/3.0/mlflow/datasets/{dataset_id}/records",
            attacker_auth,
            {"records": json.dumps([{"inputs": {"q": "x"}, "expectations": {"a": "poison"}}])},
        ).status_code
        == 403
    )
    assert (
        requests.delete(
            f"{base}/api/3.0/mlflow/datasets/{dataset_id}", auth=attacker_auth
        ).status_code
        == 403
    )

    # Issue create, read, update, and search are denied.
    assert (
        post(
            "/api/3.0/mlflow/issues",
            attacker_auth,
            {"experiment_id": exp_id, "name": "injected", "description": "injected"},
        ).status_code
        == 403
    )
    assert (
        requests.get(f"{base}/api/3.0/mlflow/issues/{issue_id}", auth=attacker_auth).status_code
        == 403
    )
    assert (
        requests.patch(
            f"{base}/api/3.0/mlflow/issues/{issue_id}",
            json={"issue_id": issue_id, "description": "tampered"},
            auth=attacker_auth,
        ).status_code
        == 403
    )
    assert (
        post("/api/3.0/mlflow/issues/search", attacker_auth, {"experiment_id": exp_id}).status_code
        == 403
    )

    # The legitimate owner still has full access — the record survived the attempted delete.
    assert (
        requests.get(f"{base}/api/3.0/mlflow/datasets/{dataset_id}", auth=owner_auth).status_code
        == 200
    )
    assert (
        requests.get(f"{base}/api/3.0/mlflow/issues/{issue_id}", auth=owner_auth).status_code == 200
    )


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_invoke_endpoints_require_experiment_update_permission(client):
    # invoke routes create runs / read traces / write assessments in an experiment -> a user
    # with no access to it is denied. Child-tier DENY enforcement is covered separately in
    # test_invoke_validators_honor_child_deny.
    base = client.tracking_uri
    owner, owner_pw = create_user(base)
    attacker, attacker_pw = create_user(base)

    exp_id = requests.post(
        f"{base}/api/2.0/mlflow/experiments/create",
        json={"name": "invoke-owner-exp"},
        auth=(owner, owner_pw),
    ).json()["experiment_id"]

    for path in (
        "/ajax-api/3.0/mlflow/issues/invoke",
        "/ajax-api/3.0/mlflow/genai/evaluate/invoke",
    ):
        resp = requests.post(
            f"{base}{path}",
            json={
                "experiment_id": exp_id,
                "trace_ids": ["tr-1"],
                "categories": ["x"],
                "provider": "p",
                "serialized_scorers": ["s"],
            },
            auth=(attacker, attacker_pw),
        )
        assert resp.status_code == 403, f"{path} -> {resp.status_code}"

        # The owner (MANAGE on the experiment) passes the auth gate — non-403 confirms the
        # validator is keyed to experiment update, not blanket-denying. The handler may still
        # error (e.g. the invoke backend isn't wired in this test env).
        resp = requests.post(
            f"{base}{path}",
            json={
                "experiment_id": exp_id,
                "trace_ids": ["tr-1"],
                "categories": ["x"],
                "provider": "p",
                "serialized_scorers": ["s"],
            },
            auth=(owner, owner_pw),
        )
        assert resp.status_code != 403, f"{path} -> {resp.status_code}"


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_invoke_validators_honor_child_deny(client):
    # The composite run-creating routes (issues/invoke, genai evaluate/invoke,
    # create-promptlab-run, prompt-optimization jobs) create a run in the target
    # experiment, so they gate on the run child tier: experiment EDIT alone must not let a
    # caller holding (run, *, DENY) create runs through them. Per RFC sub-resource
    # permissions, a child DENY must hold on every path that reaches the child, not only on
    # the direct CreateRun RPC.
    base = client.tracking_uri
    owner, owner_pw = create_user(base)
    attacker, attacker_pw = create_user(base)

    exp_id = requests.post(
        f"{base}/api/2.0/mlflow/experiments/create",
        json={"name": "invoke-child-deny-exp"},
        auth=(owner, owner_pw),
    ).json()["experiment_id"]

    # Attacker gets experiment EDIT -- enough to clear the experiment-tier gate.
    grant_role_permission(base, attacker, "experiment", exp_id, "EDIT")

    run_creating_routes = [
        (
            "/ajax-api/3.0/mlflow/issues/invoke",
            {"experiment_id": exp_id, "trace_ids": ["tr-1"], "categories": ["x"], "provider": "p"},
        ),
        (
            "/ajax-api/3.0/mlflow/genai/evaluate/invoke",
            {"experiment_id": exp_id, "trace_ids": ["tr-1"], "serialized_scorers": ["s"]},
        ),
        ("/ajax-api/2.0/mlflow/runs/create-promptlab-run", {"experiment_id": exp_id}),
        (
            "/api/3.0/mlflow/prompt-optimization/jobs",
            {
                "experiment_id": exp_id,
                "source_prompt_uri": "prompts:/test/1",
                "config": {"optimizer_type": 1, "dataset_id": "d", "scorers": ["Correctness"]},
            },
        ),
    ]

    # Baseline: experiment EDIT with no run grant -> the run tier falls back to experiment
    # update, so the auth gate passes. The handler may still error for unrelated reasons
    # (backend not wired in this env), so assert only that it is not a 403.
    for path, payload in run_creating_routes:
        resp = requests.post(f"{base}{path}", json=payload, auth=(attacker, attacker_pw))
        assert resp.status_code != 403, f"baseline {path} -> {resp.status_code}"

    # Add a run-tier DENY. The run child tier is consulted ahead of the experiment
    # fallback, so every run-creating route must now reject the caller with 403.
    grant_role_permission(base, attacker, "run", "*", "DENY")
    for path, payload in run_creating_routes:
        resp = requests.post(f"{base}{path}", json=payload, auth=(attacker, attacker_pw))
        assert resp.status_code == 403, f"run DENY {path} -> {resp.status_code}"


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_invoke_scorer_honors_child_deny(client):
    # INVOKE_SCORER scores traces and (optionally) writes assessments in an experiment. It
    # must honor a (trace, *, DENY) grant, and a (assessment, *, DENY) grant when
    # log_assessments is set -- an experiment-EDIT caller must not bypass those child DENYs.
    base = client.tracking_uri
    owner, owner_pw = create_user(base)
    exp_id = requests.post(
        f"{base}/api/2.0/mlflow/experiments/create",
        json={"name": "invoke-scorer-deny-exp"},
        auth=(owner, owner_pw),
    ).json()["experiment_id"]
    url = f"{base}/ajax-api/3.0/mlflow/scorer/invoke"
    # An inline (serialized) scorer references no registered scorer, so scorer_version is
    # not consulted; this isolates the trace/assessment tiers. The handler may 400 on the
    # dummy payload, so the baseline asserts only that the auth gate is not a 403.
    payload = {"experiment_id": exp_id, "serialized_scorer": "{}", "trace_ids": ["tr-1"]}

    def fresh_editor():
        user, pw = create_user(base)
        grant_role_permission(base, user, "experiment", exp_id, "EDIT")
        return user, pw

    # Baseline: experiment EDIT alone clears the gate (trace READ falls back to experiment).
    user, pw = fresh_editor()
    assert requests.post(url, json=payload, auth=(user, pw)).status_code != 403

    # (trace, *, DENY): the traces being scored are unreadable -> denied.
    user, pw = fresh_editor()
    grant_role_permission(base, user, "trace", "*", "DENY")
    assert requests.post(url, json=payload, auth=(user, pw)).status_code == 403

    # (assessment, *, DENY): consulted only when log_assessments is set.
    user, pw = fresh_editor()
    grant_role_permission(base, user, "assessment", "*", "DENY")
    # Without log_assessments the assessment tier is not consulted -> still passes.
    assert requests.post(url, json=payload, auth=(user, pw)).status_code != 403
    # With log_assessments the job writes assessments, so assessment UPDATE is required.
    assert (
        requests.post(url, json={**payload, "log_assessments": True}, auth=(user, pw)).status_code
        == 403
    )


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_invoke_genai_evaluate_and_issue_detection_honor_trace_assessment_deny(client):
    # INVOKE_GENAI_EVALUATE reads, tags (set_trace_tag), and writes assessments onto the
    # supplied traces, so it needs trace UPDATE + assessment UPDATE. INVOKE_ISSUE_DETECTION
    # reads the traces and writes Issue assessments (_annotate_issue_traces), so it needs
    # trace READ + assessment UPDATE. Both must honor (trace, *, DENY) and (assessment, *,
    # DENY) -- an experiment-EDIT caller must not bypass those child DENYs. (Trace/assessment
    # are gated experiment-scoped; whether the traces belong to the experiment is business
    # logic, out of the auth model's scope.)
    base = client.tracking_uri
    owner, owner_pw = create_user(base)
    exp_id = requests.post(
        f"{base}/api/2.0/mlflow/experiments/create",
        json={"name": "invoke-eval-deny-exp"},
        auth=(owner, owner_pw),
    ).json()["experiment_id"]
    evaluate_url = f"{base}/ajax-api/3.0/mlflow/genai/evaluate/invoke"
    issues_url = f"{base}/ajax-api/3.0/mlflow/issues/invoke"
    evaluate_payload = {"experiment_id": exp_id, "trace_ids": ["tr-1"], "serialized_scorers": ["s"]}
    issues_payload = {
        "experiment_id": exp_id,
        "trace_ids": ["tr-1"],
        "categories": ["x"],
        "provider": "p",
    }

    def fresh_editor():
        user, pw = create_user(base)
        grant_role_permission(base, user, "experiment", exp_id, "EDIT")
        return user, pw

    # Baseline: experiment EDIT clears every tier via fallback (handler may error for
    # unrelated reasons, so assert only that the auth gate is not a 403).
    user, pw = fresh_editor()
    assert requests.post(evaluate_url, json=evaluate_payload, auth=(user, pw)).status_code != 403
    assert requests.post(issues_url, json=issues_payload, auth=(user, pw)).status_code != 403

    # (trace, *, DENY): the traces being read are unreadable -> both routes denied.
    user, pw = fresh_editor()
    grant_role_permission(base, user, "trace", "*", "DENY")
    assert requests.post(evaluate_url, json=evaluate_payload, auth=(user, pw)).status_code == 403
    assert requests.post(issues_url, json=issues_payload, auth=(user, pw)).status_code == 403

    # (assessment, *, DENY): both routes write assessments back onto the traces
    # (genai-evaluate via the eval harness, issue detection via _annotate_issue_traces),
    # so both are denied.
    user, pw = fresh_editor()
    grant_role_permission(base, user, "assessment", "*", "DENY")
    assert requests.post(evaluate_url, json=evaluate_payload, auth=(user, pw)).status_code == 403
    assert requests.post(issues_url, json=issues_payload, auth=(user, pw)).status_code == 403

    # (trace, *, READ) caps traces at read: genai-evaluate tags the evaluated traces
    # (set_trace_tag), so it requires trace UPDATE -> denied; issue detection only reads the
    # traces, so trace READ is enough -> still allowed. This distinguishes the two tiers.
    user, pw = fresh_editor()
    grant_role_permission(base, user, "trace", "*", "READ")
    assert requests.post(evaluate_url, json=evaluate_payload, auth=(user, pw)).status_code == 403
    assert requests.post(issues_url, json=issues_payload, auth=(user, pw)).status_code != 403


def test_delete_prompt_optimization_job_honors_run_deny(monkeypatch):
    # DeletePromptOptimizationJob deletes the job's associated MLflow run, so it must honor a
    # (run, *, DENY) grant an experiment-MANAGE caller would otherwise bypass. Keep the
    # experiment/job-tier gate; add run-tier DELETE on the job's run.
    from mlflow.server import auth
    from mlflow.server.auth.permissions import DENY, MANAGE

    # Experiment/job-tier gate passes (caller has MANAGE on the experiment).
    monkeypatch.setattr(auth, "_get_permission_from_prompt_optimization_job_id", lambda: MANAGE)
    # The job created a run.
    monkeypatch.setattr(auth, "_prompt_optimization_job_run_id", lambda: "run-1")

    # (run, *, DENY) on that run -> delete denied despite experiment MANAGE.
    monkeypatch.setattr(auth, "_get_run_permission", lambda _rid: DENY)
    assert auth.validate_can_delete_prompt_optimization_job() is False

    # Run deletable -> allowed.
    monkeypatch.setattr(auth, "_get_run_permission", lambda _rid: MANAGE)
    assert auth.validate_can_delete_prompt_optimization_job() is True

    # No associated run -> the experiment/job-tier gate alone governs (allowed).
    monkeypatch.setattr(auth, "_prompt_optimization_job_run_id", lambda: None)
    assert auth.validate_can_delete_prompt_optimization_job() is True


def test_cancel_prompt_optimization_job_honors_run_deny(monkeypatch):
    # CancelPromptOptimizationJob terminates the job's associated MLflow run
    # (update_run_info -> KILLED), so it must honor a (run, *, DENY) grant an experiment
    # editor would otherwise bypass. Keep the experiment/job-tier gate; add run-tier UPDATE
    # on the job's run (Copilot).
    from mlflow.server import auth
    from mlflow.server.auth.permissions import DENY, MANAGE

    monkeypatch.setattr(auth, "_get_permission_from_prompt_optimization_job_id", lambda: MANAGE)
    monkeypatch.setattr(auth, "_prompt_optimization_job_run_id", lambda: "run-1")

    monkeypatch.setattr(auth, "_get_run_permission", lambda _rid: DENY)
    assert auth.validate_can_update_prompt_optimization_job() is False

    monkeypatch.setattr(auth, "_get_run_permission", lambda _rid: MANAGE)
    assert auth.validate_can_update_prompt_optimization_job() is True

    # No associated run -> the experiment/job-tier gate alone governs (allowed).
    monkeypatch.setattr(auth, "_prompt_optimization_job_run_id", lambda: None)
    assert auth.validate_can_update_prompt_optimization_job() is True

    # A run the store no longer knows resolves like the tolerant cancel handler: allowed.
    def _gone(_rid):
        raise MlflowException("gone", error_code=RESOURCE_DOES_NOT_EXIST)

    monkeypatch.setattr(auth, "_prompt_optimization_job_run_id", lambda: "run-1")
    monkeypatch.setattr(auth, "_get_run_permission", _gone)
    assert auth.validate_can_update_prompt_optimization_job() is True


@pytest.mark.parametrize("uri_field", ["source_prompt_uri", "sourcePromptUri"])
@pytest.mark.parametrize(
    ("prompt_version_permission", "expected"),
    [
        ("NO_PERMISSIONS", False),
        ("READ", False),
        ("USE", False),
        ("DENY", False),
        ("EDIT", True),
        ("MANAGE", True),
    ],
)
def test_create_prompt_optimization_job_requires_prompt_version_update(
    monkeypatch, uri_field, prompt_version_permission, expected
):
    # The optimize job WRITES a new version of the source prompt (register_prompt on
    # completion), so the prompt's version tier must positively allow can_update -- the
    # absence of a DENY is not enough (review finding). The body is parsed through the
    # handler's proto message, so the JSON alias spelling must hit the same check. All
    # other checks are stubbed to pass.
    from mlflow.server.auth.permissions import get_permission

    monkeypatch.setattr(auth_module, "validate_can_create_run", lambda: True)
    monkeypatch.setattr(auth_module, "_scorer_version_deny_active", lambda _e: False)
    captured = {}

    def fake_prompt_version_permission(name):
        captured["name"] = name
        return get_permission(prompt_version_permission)

    monkeypatch.setattr(auth_module, "_prompt_version_permission", fake_prompt_version_permission)
    with auth_module.app.test_request_context(
        "/x", method="POST", json={"experiment_id": "e1", uri_field: "prompts:/my-prompt/3"}
    ):
        assert auth_module.validate_can_create_prompt_optimization_job() is expected
    assert captured["name"] == "my-prompt"


def test_create_prompt_optimization_job_fails_closed_on_unresolvable_prompt(monkeypatch):
    # An unparsable URI or a prompt the registry does not know fails CLOSED (review
    # finding): the job would otherwise load/write an unresolvable prompt at runtime.
    monkeypatch.setattr(auth_module, "validate_can_create_run", lambda: True)
    monkeypatch.setattr(auth_module, "_scorer_version_deny_active", lambda _e: False)

    with auth_module.app.test_request_context(
        "/x", method="POST", json={"experiment_id": "e1", "source_prompt_uri": "not-a-uri"}
    ):
        assert auth_module.validate_can_create_prompt_optimization_job() is False

    def _missing(_name):
        raise MlflowException("no prompt", error_code=RESOURCE_DOES_NOT_EXIST)

    monkeypatch.setattr(auth_module, "_prompt_version_permission", _missing)
    with auth_module.app.test_request_context(
        "/x", method="POST", json={"experiment_id": "e1", "source_prompt_uri": "prompts:/p/1"}
    ):
        assert auth_module.validate_can_create_prompt_optimization_job() is False


@pytest.mark.parametrize("dataset_field", ["dataset_id", "datasetId"])
def test_create_prompt_optimization_job_requires_dataset_read(monkeypatch, dataset_field):
    # The handler immediately loads config.dataset_id, links it to the new run, and the
    # worker reads its records -- so the composite route must apply the direct dataset
    # APIs' check: READ on EVERY associated experiment, fail-closed on a missing dataset
    # or empty association (review finding: run/prompt permissions could otherwise read a
    # dataset the direct routes deny). Parsed through the handler's proto message, so the
    # JSON alias spelling hits the same check.
    from mlflow.server.auth.permissions import get_permission

    monkeypatch.setattr(auth_module, "validate_can_create_run", lambda: True)
    monkeypatch.setattr(auth_module, "_scorer_version_deny_active", lambda _e: False)
    monkeypatch.setattr(auth_module, "authenticate_request", lambda: SimpleNamespace(username="u"))
    experiment_perms = {"e1": "READ", "e2": "NO_PERMISSIONS"}
    monkeypatch.setattr(
        auth_module,
        "_get_experiment_permission",
        lambda eid, _u: get_permission(experiment_perms[eid]),
    )
    associations = {"ds-ok": ["e1"], "ds-blocked": ["e1", "e2"], "ds-orphan": []}

    def fake_get_dataset_experiment_ids(dataset_id):
        if dataset_id not in associations:
            raise MlflowException("no dataset", error_code=RESOURCE_DOES_NOT_EXIST)
        return associations[dataset_id]

    monkeypatch.setattr(
        auth_module,
        "_get_tracking_store",
        lambda: SimpleNamespace(get_dataset_experiment_ids=fake_get_dataset_experiment_ids),
    )

    def result(body):
        with auth_module.app.test_request_context("/x", method="POST", json=body):
            return auth_module.validate_can_create_prompt_optimization_job()

    assert result({"experiment_id": "e1", "config": {dataset_field: "ds-ok"}}) is True
    # READ missing on ANY associated experiment blocks the composite route.
    assert result({"experiment_id": "e1", "config": {dataset_field: "ds-blocked"}}) is False
    # A nonexistent dataset and an empty association both fail closed.
    assert result({"experiment_id": "e1", "config": {dataset_field: "ds-missing"}}) is False
    assert result({"experiment_id": "e1", "config": {dataset_field: "ds-orphan"}}) is False
    # No dataset supplied: no dataset check.
    assert result({"experiment_id": "e1"}) is True


def test_invoke_issue_detection_requires_endpoint_use(monkeypatch):
    # A non-empty endpoint_name makes the handler submit discovery through
    # gateway:/<name>, so the validator must require the SAME gateway-endpoint USE as
    # direct invocation -- the worker doesn't propagate the caller identity, so denial
    # must happen BEFORE the handler (review finding). The handler reads the raw JSON
    # body for this route, so the validator's raw-body read matches it exactly.
    from mlflow.server.auth.permissions import MANAGE

    monkeypatch.setattr(auth_module, "validate_can_create_run", lambda: True)
    monkeypatch.setattr(auth_module, "_get_request_param", lambda _n: "e1")
    monkeypatch.setattr(auth_module, "_get_trace_permission_for_experiment", lambda _e: MANAGE)
    monkeypatch.setattr(auth_module, "_experiment_child_permission", lambda *a, **k: MANAGE)
    monkeypatch.setattr(auth_module, "authenticate_request", lambda: SimpleNamespace(username="u"))
    checked = []

    def fake_gateway_use(endpoint_name, username):
        checked.append((endpoint_name, username))
        return endpoint_name == "ep-allowed"

    monkeypatch.setattr(auth_module, "_validate_gateway_use_permission", fake_gateway_use)

    def result(body):
        with auth_module.app.test_request_context("/x", method="POST", json=body):
            return auth_module.validate_can_invoke_issue_detection()

    base = {"experiment_id": "e1", "trace_ids": ["tr-1"]}
    assert result({**base, "endpoint_name": "ep-denied"}) is False
    assert result({**base, "endpoint_name": "ep-allowed"}) is True
    # The provider/model + secret path is unaffected when endpoint_name is absent.
    assert result(base) is True
    assert checked == [("ep-denied", "u"), ("ep-allowed", "u")]


def test_direct_job_submission_denies_protected_builtin_jobs():
    # POST /ajax-api/3.0/jobs/ runs allowlisted job functions with caller-controlled
    # params; every product path submits these jobs IN-PROCESS from already-validated
    # higher-level routes, so direct HTTP submission of the built-in names would bypass
    # every run/trace/assessment/prompt/scorer/gateway validator. It is denied outright,
    # fail closed on a malformed body (review finding, Critical). Operator-extended names
    # keep the authenticated-only contract.
    validator = auth_module._get_job_route_validator("/ajax-api/3.0/jobs")

    class FakeRequest:
        def __init__(self, body=None, raises=False):
            self._body, self._raises = body, raises

        async def json(self):
            if self._raises:
                raise ValueError("bad json")
            return self._body

    def run(req):
        return asyncio.run(validator("u", req))

    for name in sorted(auth_module._PROTECTED_BUILTIN_JOB_NAMES):
        assert run(FakeRequest({"job_name": name, "params": {}})) is False, name
    assert run(FakeRequest({"job_name": "operator_custom_job", "params": {}})) is True
    assert run(FakeRequest(raises=True)) is False
    # /jobs/search stays authentication-only (results are creator-filtered downstream).
    search_validator = auth_module._get_job_route_validator("/ajax-api/3.0/jobs/search")
    assert asyncio.run(search_validator("u", FakeRequest({}))) is True


def test_scorer_payload_gateway_ref():
    # Same extraction helpers as the store's registration path; malformed payloads
    # report parse failure so callers fail closed (review finding).
    payload = json.dumps({"instructions_judge_pydantic_data": {"model": "gateway:/ep-1"}})
    assert auth_module._scorer_payload_gateway_ref(payload) == (True, "ep-1")
    non_gateway = json.dumps({"instructions_judge_pydantic_data": {"model": "openai:/gpt-4o"}})
    assert auth_module._scorer_payload_gateway_ref(non_gateway) == (True, None)
    assert auth_module._scorer_payload_gateway_ref(json.dumps({"other": 1})) == (True, None)
    assert auth_module._scorer_payload_gateway_ref("not json") == (False, None)


def test_register_scorer_requires_gateway_endpoint_use(monkeypatch):
    # Registration resolves and BINDS the scorer's gateway endpoint (the store rewrites
    # the name to an id and creates/replaces a SqlGatewayEndpointBinding), so an
    # experiment editor without endpoint USE must be denied before the handler; malformed
    # payloads fail closed (review finding).
    from mlflow.server.auth.permissions import MANAGE

    monkeypatch.setattr(auth_module, "authenticate_request", lambda: SimpleNamespace(username="u"))
    monkeypatch.setattr(auth_module, "_get_experiment_permission", lambda _e, _u: MANAGE)
    monkeypatch.setattr(auth_module, "_register_scorer_version_permission", lambda _e, _n: MANAGE)
    monkeypatch.setattr(auth_module, "_scorer_version_deny_active", lambda _e: False)
    monkeypatch.setattr(
        auth_module,
        "_get_tracking_store",
        lambda: SimpleNamespace(get_scorer=lambda _e, _n: SimpleNamespace()),
    )
    gateway_allowed = {"value": False}
    checked = []

    def fake_gateway_use(endpoint_name, username):
        checked.append(endpoint_name)
        return gateway_allowed["value"]

    monkeypatch.setattr(auth_module, "_validate_gateway_use_permission", fake_gateway_use)

    def result(serialized_scorer):
        body = {"experiment_id": "e1", "name": "s1", "serialized_scorer": serialized_scorer}
        with auth_module.app.test_request_context("/x", method="POST", json=body):
            return auth_module.validate_can_register_scorer()

    gateway_payload = json.dumps({"builtin_scorer_pydantic_data": {"model": "gateway:/ep-1"}})
    assert result(gateway_payload) is False  # no endpoint USE
    gateway_allowed["value"] = True
    assert result(gateway_payload) is True
    assert checked == ["ep-1", "ep-1"]
    assert result("not json") is False  # malformed payload fails closed
    non_gateway = json.dumps({"builtin_scorer_pydantic_data": {"model": "openai:/gpt-4o"}})
    checked.clear()
    assert result(non_gateway) is True
    assert checked == []  # no gateway check for non-gateway scorers


def test_update_online_scoring_config_requires_enabled_work_tiers(monkeypatch):
    # A positive sample rate enables background jobs that read traces, execute the stored
    # scorer (with its gateway endpoint), and write assessments -- each denied tier must
    # block enabling; disabling needs only experiment UPDATE (review finding).
    from mlflow.server.auth.permissions import MANAGE, NO_PERMISSIONS

    monkeypatch.setattr(auth_module, "authenticate_request", lambda: SimpleNamespace(username="u"))
    monkeypatch.setattr(auth_module, "_get_experiment_permission", lambda _e, _u: MANAGE)
    granted = {"trace": True, "assessment": True, "scorer_version": True, "gateway": True}
    monkeypatch.setattr(
        auth_module,
        "_get_trace_permission_for_experiment",
        lambda _e: MANAGE if granted["trace"] else NO_PERMISSIONS,
    )
    monkeypatch.setattr(
        auth_module,
        "_experiment_child_permission",
        lambda _t, _k, _e: MANAGE if granted["assessment"] else NO_PERMISSIONS,
    )
    monkeypatch.setattr(
        auth_module,
        "_get_scorer_version_permission",
        lambda _e, _n: MANAGE if granted["scorer_version"] else NO_PERMISSIONS,
    )
    monkeypatch.setattr(
        auth_module,
        "_gateway_endpoint_use_permission_by_id",
        lambda _i, _u: granted["gateway"],
    )
    stored = {"scorer": SimpleNamespace(serialized_scorer="stored-payload")}

    def fake_get_scorer(_experiment_id, _name):
        if stored["scorer"] is None:
            raise MlflowException("no scorer", error_code=RESOURCE_DOES_NOT_EXIST)
        return stored["scorer"]

    monkeypatch.setattr(
        auth_module, "_get_tracking_store", lambda: SimpleNamespace(get_scorer=fake_get_scorer)
    )
    monkeypatch.setattr(auth_module, "_scorer_payload_gateway_ref", lambda _p: (True, "ep-id-1"))

    def result(body):
        with auth_module.app.test_request_context("/x", method="POST", json=body):
            return auth_module.validate_can_update_online_scoring_config()

    base = {"experiment_id": "e1", "name": "s1", "sample_rate": 0.5}
    assert result({**base, "sample_rate": 0}) is True  # disable: experiment UPDATE only
    assert result(base) is True  # enable with every tier granted
    for tier in ("trace", "assessment", "scorer_version", "gateway"):
        granted[tier] = False
        assert result(base) is False, tier
        granted[tier] = True
    assert result({"experiment_id": "e1", "name": "s1"}) is False  # missing sample_rate
    stored["scorer"] = None
    assert result(base) is False  # missing scorer fails closed


def test_read_online_scoring_configs_honors_scorer_version_tier(monkeypatch):
    # Configs carry scorer_id; the scorer_version tier parents to the SCORER
    # (<experiment_id>/<name>), so the read gate resolves ids to scorer identities in ONE
    # bulk listing across the distinct experiments (query-count bound: many experiments,
    # one listing call) -- a per-scorer parent DENY applies, and an unresolved id fails
    # closed (review findings).
    from mlflow.server.auth.permissions import MANAGE, NO_PERMISSIONS

    monkeypatch.setattr(auth_module, "authenticate_request", lambda: SimpleNamespace(username="u"))
    monkeypatch.setattr(auth_module, "_get_experiment_permission", lambda _e, _u: MANAGE)
    n_experiments = 40
    configs = [
        SimpleNamespace(experiment_id=f"e{i}", scorer_id=f"sc-{i}") for i in range(n_experiments)
    ]
    list_calls = []

    def fake_list_across(experiment_ids):
        list_calls.append(list(experiment_ids))
        return [
            SimpleNamespace(experiment_id=f"e{i}", scorer_id=f"sc-{i}", scorer_name=f"scorer-{i}")
            for i in range(n_experiments)
        ]

    monkeypatch.setattr(
        auth_module,
        "_get_tracking_store",
        lambda: SimpleNamespace(
            get_online_scoring_configs=lambda _ids: configs,
            list_scorers_across_experiments=fake_list_across,
        ),
    )
    version_perm = {"value": MANAGE}
    resolved = []

    def fake_version_perm(experiment_id, name):
        resolved.append((experiment_id, name))
        return version_perm["value"]

    monkeypatch.setattr(auth_module, "_get_scorer_version_permission", fake_version_perm)

    def result():
        with auth_module.app.test_request_context(
            "/x", method="POST", json={"scorer_ids": [c.scorer_id for c in configs]}
        ):
            return auth_module.validate_can_read_online_scoring_configs()

    assert result() is True
    assert len(list_calls) == 1  # one bulk listing for 40 experiments, never per-experiment
    assert sorted(list_calls[0]) == sorted(f"e{i}" for i in range(n_experiments))
    assert ("e0", "scorer-0") in resolved  # resolved to the scorer identity
    version_perm["value"] = NO_PERMISSIONS
    assert result() is False
    # An id absent from the bulk listing fails closed.
    configs[:] = [SimpleNamespace(experiment_id="e0", scorer_id="sc-unknown")]
    version_perm["value"] = MANAGE
    assert result() is False


def test_global_mcp_endpoint_search_honors_wildcard_version_deny(monkeypatch):
    # The global /mcp-servers/endpoints route never reaches the per-server gate; a
    # version-bearing selection there must honor the wildcard (mcp_server_version, *,
    # DENY) as a global veto (review finding).
    from starlette.datastructures import QueryParams

    from mlflow.server.auth.permissions import DENY, READ

    grant = {"value": None}
    monkeypatch.setattr(
        auth_module, "_wildcard_grant_in_request_workspace", lambda _t, _u: grant["value"]
    )
    validator = auth_module._get_mcp_server_validator("/api/3.0/mlflow/mcp-servers/endpoints")

    def run(query):
        req = SimpleNamespace(method="GET", query_params=QueryParams(query))
        return asyncio.run(validator("u", req))

    assert run("server_version=3") is True  # no grant: default posture
    grant["value"] = DENY
    assert run("server_version=3") is False
    assert run("server_alias=prod") is False
    assert run("filter_string=server_version%3D'3'") is False
    assert run("max_results=5") is True  # non-version-bearing search unaffected
    grant["value"] = READ
    assert run("server_version=3") is True


def test_legacy_trace_responses_filter_linked_prompts_tag(monkeypatch):
    # Legacy (V2) GetTraceInfo / SearchTraces return TraceInfo with repeated tags that
    # carry the reserved linked-prompts tag; both are filtered like the V3 spelling
    # (review finding).
    from mlflow.protos.service_pb2 import GetTraceInfo, SearchTraces

    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    monkeypatch.setattr(auth_module, "authenticate_request", lambda: SimpleNamespace(username="u"))
    monkeypatch.setattr(auth_module, "_role_based_read_predicate", _linked_prompts_predicate_fake)
    value = json.dumps([{"name": "p-ok", "version": "1"}, {"name": "p-denied", "version": "2"}])

    get_msg = GetTraceInfo.Response()
    tag = get_msg.trace_info.tags.add()
    tag.key = auth_module._LINKED_PROMPTS_TAG_KEY
    tag.value = value
    resp = _fake_resp(get_msg)
    auth_module.redact_get_trace_info_linked_prompts(resp)
    out = GetTraceInfo.Response()
    auth_module.parse_dict(json.loads(resp.data), out)
    (out_tag,) = out.trace_info.tags
    assert json.loads(out_tag.value) == [{"name": "p-ok", "version": "1"}]

    search_msg = SearchTraces.Response()
    ti = search_msg.traces.add()
    tag = ti.tags.add()
    tag.key = auth_module._LINKED_PROMPTS_TAG_KEY
    tag.value = json.dumps([{"name": "p-denied", "version": "2"}])
    resp = _fake_resp(search_msg)
    auth_module.redact_search_traces_linked_prompts(resp)
    out = SearchTraces.Response()
    auth_module.parse_dict(json.loads(resp.data), out)
    assert not [t for t in out.traces[0].tags if t.key == auth_module._LINKED_PROMPTS_TAG_KEY]


def test_trace_responses_filter_sibling_run_and_model_ids(monkeypatch):
    # Trace READ plus (run, *, DENY) / (logged_model, *, DENY) must not learn
    # mlflow.sourceRun / mlflow.modelId through trace responses. Each reference is judged
    # by its OWN resource's experiment via bounded bulk resolution; unresolvable
    # references are hidden fail-closed (review finding). Covers the V3 map and the
    # legacy repeated metadata shapes through the shared filter.
    from mlflow.protos import service_pb2 as pb
    from mlflow.store.entities.paged_list import PagedList
    from mlflow.tracing.constant import TraceMetadataKey

    def fake_search_runs(experiment_ids, filter_string, run_view_type, max_results):
        assert experiment_ids == ["9"]
        return PagedList(
            [SimpleNamespace(info=SimpleNamespace(run_id="run-ok", experiment_id="9"))]
            if "run-ok" in filter_string
            else [],
            None,
        )

    def fake_search_logged_models(experiment_ids, filter_string, max_results):
        return PagedList(
            [SimpleNamespace(model_id="m-denied", experiment_id="13")]
            if "m-denied" in filter_string
            else [],
            None,
        )

    def fake_predicate(_u, resource_type, parent_type=None):
        if resource_type == "run":
            return lambda eid: eid == "9"
        if resource_type == "logged_model":
            return lambda eid: eid == "9"  # m-denied resolves to 13: unreadable
        return lambda *_a: True

    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    monkeypatch.setattr(auth_module, "authenticate_request", lambda: SimpleNamespace(username="u"))
    monkeypatch.setattr(
        auth_module,
        "_get_tracking_store",
        lambda: SimpleNamespace(
            search_runs=fake_search_runs, search_logged_models=fake_search_logged_models
        ),
    )
    monkeypatch.setattr(auth_module, "_role_based_read_predicate", fake_predicate)

    # V3 spelling (map metadata).
    v3 = pb.GetTraceInfoV3.Response()
    ti = v3.trace.trace_info
    ti.trace_location.mlflow_experiment.experiment_id = "9"
    ti.trace_metadata[TraceMetadataKey.SOURCE_RUN] = "run-ok"
    ti.trace_metadata[TraceMetadataKey.MODEL_ID] = "m-denied"
    resp = _fake_resp(v3)
    auth_module.redact_get_trace_info_v3_assessments(resp)
    out = pb.GetTraceInfoV3.Response()
    auth_module.parse_dict(json.loads(resp.data), out)
    md = out.trace.trace_info.trace_metadata
    assert md[TraceMetadataKey.SOURCE_RUN] == "run-ok"  # resolves to exp 9: readable
    assert TraceMetadataKey.MODEL_ID not in md  # resolves to exp 13: hidden

    # Legacy spelling (repeated request_metadata); an unresolvable run is hidden.
    v2 = pb.GetTraceInfo.Response()
    v2.trace_info.experiment_id = "9"
    entry = v2.trace_info.request_metadata.add()
    entry.key = TraceMetadataKey.SOURCE_RUN
    entry.value = "run-missing"
    resp = _fake_resp(v2)
    auth_module.redact_get_trace_info_linked_prompts(resp)
    out = pb.GetTraceInfo.Response()
    auth_module.parse_dict(json.loads(resp.data), out)
    assert not [e for e in out.trace_info.request_metadata if e.key == TraceMetadataKey.SOURCE_RUN]


def test_search_traces_resource_filter_gate(monkeypatch):
    # run_id / metadata.`mlflow.sourceRun` / metadata.`mlflow.modelId` trace filters
    # execute against protected sibling references, so they require the corresponding
    # tier's READ on every requested experiment (review finding).
    denied = {"run": False, "logged_model": False}

    def fake_predicate(_u, resource_type, parent_type=None):
        return lambda _eid: not denied[resource_type]

    monkeypatch.setattr(auth_module, "authenticate_request", lambda: SimpleNamespace(username="u"))
    monkeypatch.setattr(auth_module, "_role_based_read_predicate", fake_predicate)

    allowed = auth_module._search_traces_resource_filter_allowed
    assert allowed(["e1"], "run_id = 'r1'") is True
    denied["run"] = True
    assert allowed(["e1"], "run_id = 'r1'") is False
    assert allowed(["e1"], "metadata.`mlflow.sourceRun` = 'r1'") is False
    assert allowed(["e1"], "metadata.`mlflow.modelId` = 'm1'") is True  # model tier ok
    denied["logged_model"] = True
    assert allowed(["e1"], "request_metadata.`mlflow.modelId` = 'm1'") is False
    # Non-resource filters and other metadata keys never consult the tiers.
    assert allowed(["e1"], "tags.foo = 'bar'") is True
    assert allowed(["e1"], "metadata.`custom.key` = 'x'") is True
    assert allowed(["e1"], "") is True


def test_permission_introspection_validates_namespace_and_scorer(monkeypatch):
    # get_user_permission must not report a permission for the WRONG registry namespace
    # (prompts and registered models share the table) or a nonexistent scorer parent --
    # both follow the RESOURCE_DOES_NOT_EXIST -> NO_PERMISSIONS path (review finding).
    registry = {
        "real-model": SimpleNamespace(_is_prompt=lambda: False),
        "real-prompt": SimpleNamespace(_is_prompt=lambda: True),
    }

    def fake_get_rm(name):
        if name not in registry:
            raise MlflowException("no rm", error_code=RESOURCE_DOES_NOT_EXIST)
        return registry[name]

    def fake_get_scorer(_experiment_id, name):
        if name != "real-scorer":
            raise MlflowException("no scorer", error_code=RESOURCE_DOES_NOT_EXIST)
        return SimpleNamespace()

    monkeypatch.setattr(
        auth_module,
        "_get_model_registry_store",
        lambda: SimpleNamespace(get_registered_model=fake_get_rm),
    )
    monkeypatch.setattr(
        auth_module,
        "_get_tracking_store",
        lambda: SimpleNamespace(
            get_scorer=fake_get_scorer, get_experiment=lambda _e: SimpleNamespace()
        ),
    )

    def expect_not_found(resource_type, resource_id):
        with pytest.raises(MlflowException, match="not found|no scorer|no rm") as exc:
            auth_module._resource_dispatch_keys(resource_type, resource_id)
        assert exc.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)

    # Correct namespaces dispatch.
    assert auth_module._resource_dispatch_keys("registered_model_version", "real-model")
    assert auth_module._resource_dispatch_keys("prompt_version", "real-prompt")
    assert auth_module._resource_dispatch_keys("scorer_version", "e1/real-scorer")
    # Namespace inversions and a missing scorer parent fail closed.
    expect_not_found("prompt_version", "real-model")
    expect_not_found("registered_model_version", "real-prompt")
    expect_not_found("scorer_version", "e1/ghost-scorer")


def test_mcp_access_endpoints_honor_server_version_tier(monkeypatch):
    # Endpoint create/update BINDS to a server version and search accepts version-bearing
    # selection; both must consult the version tier before the handler, not rely on
    # response redaction (review finding).
    from starlette.datastructures import QueryParams

    from mlflow.server.auth.permissions import MANAGE, NO_PERMISSIONS

    monkeypatch.setattr(auth_module, "_get_mcp_server_permission", lambda _n, _u: MANAGE)
    version = {"perm": NO_PERMISSIONS}
    monkeypatch.setattr(
        auth_module, "_get_mcp_server_version_permission", lambda _n, _u: version["perm"]
    )
    create_validator = auth_module._get_mcp_server_validator(
        "/api/3.0/mlflow/mcp-servers/ns/srv/endpoints"
    )

    class FakeRequest:
        def __init__(self, method, body=None, query=""):
            self.method = method
            self._body = body
            self.query_params = QueryParams(query)
            self.state = SimpleNamespace()

        async def json(self):
            if self._body is None:
                raise ValueError("no body")
            return self._body

    # A version-binding create is denied without version-tier UPDATE, allowed with it.
    bind_body = {"url": "http://x", "server_version": "3"}
    assert asyncio.run(create_validator("u", FakeRequest("POST", bind_body))) is False
    version["perm"] = MANAGE
    assert asyncio.run(create_validator("u", FakeRequest("POST", bind_body))) is True
    version["perm"] = NO_PERMISSIONS
    # An alias binding is gated the same way; a bindingless create needs only the parent.
    alias_body = {"url": "http://x", "server_alias": "prod"}
    assert asyncio.run(create_validator("u", FakeRequest("POST", alias_body))) is False
    assert asyncio.run(create_validator("u", FakeRequest("POST", {"url": "http://x"}))) is True
    # A malformed body fails closed.
    assert asyncio.run(create_validator("u", FakeRequest("POST"))) is False
    # Version-bearing search selection requires version-tier READ; plain search does not.
    assert asyncio.run(create_validator("u", FakeRequest("GET", query="server_version=3"))) is False
    assert (
        asyncio.run(
            create_validator("u", FakeRequest("GET", query="filter_string=server_alias%3D'x'"))
        )
        is False
    )
    assert asyncio.run(create_validator("u", FakeRequest("GET", query="max_results=5"))) is True


def test_visible_linked_prompts_value():
    readable = lambda name: name == "p-ok"  # noqa: E731
    value = json.dumps([{"name": "p-ok", "version": "1"}, {"name": "p-denied", "version": "2"}])
    assert json.loads(auth_module._visible_linked_prompts_value(value, readable)) == [
        {"name": "p-ok", "version": "1"}
    ]
    all_denied = json.dumps([{"name": "p-denied", "version": "2"}])
    assert auth_module._visible_linked_prompts_value(all_denied, readable) is None
    all_ok = json.dumps([{"name": "p-ok", "version": "1"}])
    assert auth_module._visible_linked_prompts_value(all_ok, readable) == all_ok
    assert auth_module._visible_linked_prompts_value("not json", readable) is None


def _linked_prompts_predicate_fake(_u, resource_type, parent_type=None):
    if resource_type == "prompt_version":
        return lambda name: name == "p-ok"
    return lambda _key: True


def test_run_response_filters_linked_prompts_tag(monkeypatch):
    # Runs serialize prompt names AND version numbers into the reserved linked-prompts
    # tag; filter it by the prompt_version tier like the trace and logged-model
    # spellings (review finding).
    from mlflow.protos.service_pb2 import GetRun

    msg = GetRun.Response()
    msg.run.info.experiment_id = "9"
    tag = msg.run.data.tags.add()
    tag.key = auth_module._LINKED_PROMPTS_TAG_KEY
    tag.value = json.dumps([{"name": "p-ok", "version": "1"}, {"name": "p-denied", "version": "2"}])

    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    monkeypatch.setattr(auth_module, "authenticate_request", lambda: SimpleNamespace(username="u"))
    monkeypatch.setattr(auth_module, "_role_based_read_predicate", _linked_prompts_predicate_fake)
    resp = _fake_resp(msg)
    auth_module.redact_get_run_model_io(resp)
    out = GetRun.Response()
    auth_module.parse_dict(json.loads(resp.data), out)
    (out_tag,) = out.run.data.tags
    assert json.loads(out_tag.value) == [{"name": "p-ok", "version": "1"}]


def test_trace_response_filters_linked_prompts_tag(monkeypatch):
    from mlflow.protos import service_pb2 as pb

    resp_msg = pb.GetTraceInfoV3.Response()
    ti = resp_msg.trace.trace_info
    ti.trace_id = "tr-1"
    ti.trace_location.mlflow_experiment.experiment_id = "9"
    ti.tags[auth_module._LINKED_PROMPTS_TAG_KEY] = json.dumps([
        {"name": "p-denied", "version": "2"}
    ])

    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    monkeypatch.setattr(auth_module, "authenticate_request", lambda: SimpleNamespace(username="u"))
    monkeypatch.setattr(auth_module, "_role_based_read_predicate", _linked_prompts_predicate_fake)
    resp = _fake_resp(resp_msg)
    auth_module.redact_get_trace_info_v3_assessments(resp)
    out = pb.GetTraceInfoV3.Response()
    auth_module.parse_dict(json.loads(resp.data), out)
    # Every reference was denied: the tag is dropped entirely.
    assert auth_module._LINKED_PROMPTS_TAG_KEY not in out.trace.trace_info.tags


def test_get_logged_model_filters_linked_prompts_tag(monkeypatch):
    from mlflow.protos.service_pb2 import GetLoggedModel

    msg = GetLoggedModel.Response()
    msg.model.info.experiment_id = "9"
    tag = msg.model.info.tags.add()
    tag.key = auth_module._LINKED_PROMPTS_TAG_KEY
    tag.value = json.dumps([{"name": "p-denied", "version": "2"}])

    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    monkeypatch.setattr(auth_module, "authenticate_request", lambda: SimpleNamespace(username="u"))
    monkeypatch.setattr(auth_module, "_role_based_read_predicate", _linked_prompts_predicate_fake)
    resp = _fake_resp(msg)
    auth_module.redact_get_logged_model_linked_prompts(resp)
    out = GetLoggedModel.Response()
    auth_module.parse_dict(json.loads(resp.data), out)
    assert not [t for t in out.model.info.tags if t.key == auth_module._LINKED_PROMPTS_TAG_KEY]


def test_search_traces_prompt_filter_gate(monkeypatch):
    # The bare `prompt` trace-filter key maps (via SearchTraceUtils, the store's own
    # parser) to the linked-prompts tag, so match presence/counts leak denied
    # prompt-version references. An equality comparison names its target
    # ('<name>/<version>'), so the exact prompt resolves through the prompt_version fold
    # WITH prompt-parent fallback; broad operators fail closed; a workspace-wide
    # (prompt_version, *, DENY) vetoes everything (review finding).
    from mlflow.server.auth.permissions import DENY, READ

    monkeypatch.setattr(auth_module, "authenticate_request", lambda: SimpleNamespace(username="u"))
    grant = {"value": None}
    monkeypatch.setattr(
        auth_module, "_wildcard_grant_in_request_workspace", lambda _t, _u: grant["value"]
    )
    resolved = []

    def fake_predicate(_username):
        def readable(name):
            resolved.append(name)
            return name == "ok-prompt"

        return readable

    monkeypatch.setattr(auth_module, "_prompt_version_read_predicate", fake_predicate)
    manager = {"value": False}
    monkeypatch.setattr(auth_module, "_request_workspace_manager", lambda _u: manager["value"])

    # Equality resolves the exact target name through the per-prompt fold.
    assert auth_module._search_traces_prompt_filter_allowed("prompt = 'ok-prompt/3'") is True
    assert auth_module._search_traces_prompt_filter_allowed("prompt = 'denied-prompt/1'") is False
    assert resolved == ["ok-prompt", "denied-prompt"]
    assert (
        auth_module._search_traces_prompt_filter_allowed("name = 'x' AND prompt = 'ok-prompt/1'")
        is True
    )
    # Broad operators cannot be bounded to specific parents without an authoritative
    # wildcard child grant: fail closed.
    assert auth_module._search_traces_prompt_filter_allowed("prompt != 'ok-prompt/3'") is False
    # A positive wildcard child grant is AUTHORITATIVE for every prompt: broad allowed.
    grant["value"] = READ
    assert auth_module._search_traces_prompt_filter_allowed("prompt != 'ok-prompt/3'") is True
    # The workspace-manager bypass wins over everything, including per-name denials.
    grant["value"] = None
    manager["value"] = True
    assert auth_module._search_traces_prompt_filter_allowed("prompt = 'denied-prompt/1'") is True
    assert auth_module._search_traces_prompt_filter_allowed("prompt != 'x/1'") is True
    manager["value"] = False
    # A workspace-wide version DENY vetoes even a readable target.
    grant["value"] = DENY
    assert auth_module._search_traces_prompt_filter_allowed("prompt = 'ok-prompt/3'") is False
    grant["value"] = READ
    assert auth_module._search_traces_prompt_filter_allowed("prompt = 'ok-prompt/3'") is True
    # An unparsable prompt-mentioning filter is unboundable: fail closed without an
    # authoritative wildcard grant; unrelated grammars other layers own (assessment
    # filters have their own gate) stay un-gated here.
    grant["value"] = None
    assert auth_module._search_traces_prompt_filter_allowed("prompt ~~~ garbage") is False
    assert auth_module._search_traces_prompt_filter_allowed("assessment.foo = 'x'") is True
    # Non-prompt filters never consult the grants.
    grant["value"] = None
    resolved.clear()
    assert auth_module._search_traces_prompt_filter_allowed("tags.foo = 'bar'") is True
    assert auth_module._search_traces_prompt_filter_allowed("") is True
    assert resolved == []


def test_end_trace_response_filters_linked_prompts_tag(monkeypatch):
    # EndTrace echoes the COMPLETE updated TraceInfo, including pre-existing
    # linked-prompt tags the caller didn't write (review finding).
    from mlflow.protos.service_pb2 import EndTrace

    msg = EndTrace.Response()
    tag = msg.trace_info.tags.add()
    tag.key = auth_module._LINKED_PROMPTS_TAG_KEY
    tag.value = json.dumps([{"name": "p-denied", "version": "2"}])
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    monkeypatch.setattr(auth_module, "authenticate_request", lambda: SimpleNamespace(username="u"))
    monkeypatch.setattr(auth_module, "_role_based_read_predicate", _linked_prompts_predicate_fake)
    resp = _fake_resp(msg)
    auth_module.redact_end_trace_linked_prompts(resp)
    out = EndTrace.Response()
    auth_module.parse_dict(json.loads(resp.data), out)
    assert not [t for t in out.trace_info.tags if t.key == auth_module._LINKED_PROMPTS_TAG_KEY]


def test_search_model_versions_backfills_after_filtering(monkeypatch):
    # The shared model/prompt-version search filter must fetch forward to refill
    # max_results after dropping unauthorized rows, advancing the token by consumed store
    # rows -- otherwise mixed grants produce empty/short pages while authorized versions
    # exist immediately afterward (review finding).
    from mlflow.protos.model_registry_pb2 import ModelVersion as ProtoModelVersion
    from mlflow.protos.model_registry_pb2 import SearchModelVersions
    from mlflow.store.entities.paged_list import PagedList
    from mlflow.utils.search_utils import SearchUtils

    def mv_proto(name, version):
        p = ProtoModelVersion()
        p.name = name
        p.version = version
        return p

    msg = SearchModelVersions.Response()
    msg.model_versions.extend([mv_proto("denied-m", "1"), mv_proto("denied-m", "2")])
    first_token = SearchUtils.create_page_token(2).decode()
    msg.next_page_token = first_token

    search_calls = []

    def fake_search(filter_string, max_results, order_by, page_token):
        search_calls.append(page_token)
        entities = [
            SimpleNamespace(to_proto=lambda v=v: mv_proto("ok-m", v)) for v in ("3", "4", "5")
        ]
        return PagedList(entities, None)  # last page

    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    monkeypatch.setattr(auth_module, "authenticate_request", lambda: SimpleNamespace(username="u"))
    monkeypatch.setattr(
        auth_module,
        "_rm_or_prompt_version_read_predicate",
        lambda _u: lambda mv: mv.name == "ok-m",
    )
    monkeypatch.setattr(
        auth_module,
        "_get_model_registry_store",
        lambda: SimpleNamespace(search_model_versions=fake_search),
    )
    with auth_module.app.test_request_context(
        "/x", method="GET", query_string={"max_results": "2"}
    ):
        resp = _fake_resp(msg)
        auth_module.filter_search_model_versions(resp)
    out = SearchModelVersions.Response()
    auth_module.parse_dict(json.loads(resp.data), out)
    # Both denied rows were replaced by the next authorized rows, up to max_results.
    assert [(mv.name, mv.version) for mv in out.model_versions] == [("ok-m", "3"), ("ok-m", "4")]
    assert search_calls == [first_token]
    # The token advances past the two consumed store rows (offset 2 + 2), not to the
    # store's own next page, so iteration resumes at the first unconsumed row.
    assert SearchUtils.parse_start_offset_from_page_token(out.next_page_token) == 4


def test_parameterized_after_request_routes_have_fallback():
    # Redactors registered on parameterized template paths (traces/<trace_id>,
    # logged-models/<model_id>) can never exact-match a real request path; the
    # generalized regex fallback must resolve them (review-round catch: the
    # GetTraceInfoV3 redactor was registered but never fired).
    def resolve(path, method):
        handler = auth_module.AFTER_REQUEST_HANDLERS.get((path, method))
        if handler is None:
            for (pattern, m), candidate in auth_module.PARAMETERIZED_AFTER_REQUEST_HANDLERS.items():
                if m == method and pattern.fullmatch(path):
                    return candidate
        return handler

    assert (
        resolve("/api/3.0/mlflow/traces/tr-abc123", "GET")
        is auth_module.redact_get_trace_info_v3_assessments
    )
    assert (
        resolve("/api/2.0/mlflow/logged-models/m-abc123", "GET")
        is auth_module.redact_get_logged_model_linked_prompts
    )


def test_create_prompt_optimization_job_builtin_fallback_checks_registered_deny(monkeypatch):
    # The job treats getattr(builtin_scorers, name) as a built-in only if it instantiates;
    # otherwise it falls back to the REGISTERED scorer of that name. A module attribute like
    # "json" (an import, not a scorer) must therefore still be checked against the
    # registered-scorer DENY (Copilot).
    monkeypatch.setattr(auth_module, "validate_can_create_run", lambda: True)
    monkeypatch.setattr(auth_module, "_scorer_version_deny_active", lambda _e: False)
    checked = []

    def fake_registered_deny(_experiment_id, name):
        checked.append(name)
        return name == "json"

    monkeypatch.setattr(auth_module, "_registered_scorer_deny_active", fake_registered_deny)
    from mlflow.genai.scorers import builtin_scorers

    assert getattr(builtin_scorers, "json", None) is not None  # the hazard under test
    with auth_module.app.test_request_context(
        "/x", method="POST", json={"experiment_id": "e1", "config": {"scorers": ["json"]}}
    ):
        assert auth_module.validate_can_create_prompt_optimization_job() is False
    assert checked == ["json"]

    # A real built-in (instantiable) is not a stored resource: no registered check, allowed.
    checked.clear()
    with auth_module.app.test_request_context(
        "/x", method="POST", json={"experiment_id": "e1", "config": {"scorers": ["Safety"]}}
    ):
        assert auth_module.validate_can_create_prompt_optimization_job() is True
    assert checked == []


def test_read_prompt_optimization_job_honors_run_deny(monkeypatch):
    # GetPromptOptimizationJob copies the associated run's metric values into the response,
    # so it must honor a (run, *, DENY) an experiment reader would otherwise bypass through
    # this composite route (review finding). Matches the Cancel/Delete treatment.
    from mlflow.server import auth
    from mlflow.server.auth.permissions import DENY, MANAGE

    monkeypatch.setattr(auth, "_get_permission_from_prompt_optimization_job_id", lambda: MANAGE)
    monkeypatch.setattr(auth, "_prompt_optimization_job_run_id", lambda: "run-1")

    monkeypatch.setattr(auth, "_get_run_permission", lambda _rid: DENY)
    assert auth.validate_can_read_prompt_optimization_job() is False

    monkeypatch.setattr(auth, "_get_run_permission", lambda _rid: MANAGE)
    assert auth.validate_can_read_prompt_optimization_job() is True

    # No associated run -> the experiment/job-tier gate alone governs (allowed).
    monkeypatch.setattr(auth, "_prompt_optimization_job_run_id", lambda: None)
    assert auth.validate_can_read_prompt_optimization_job() is True

    # A run the store no longer knows resolves like the tolerant handler: allowed.
    def _gone(_rid):
        raise MlflowException("gone", error_code=RESOURCE_DOES_NOT_EXIST)

    monkeypatch.setattr(auth, "_prompt_optimization_job_run_id", lambda: "run-1")
    monkeypatch.setattr(auth, "_get_run_permission", _gone)
    assert auth.validate_can_read_prompt_optimization_job() is True


def test_filter_references_assessments_matches_backtick_quoted_identifiers():
    # Trace filters may backtick-quote the entity identifier (SearchUtils._valid_entity_type
    # strips the backticks), so `feedback`.correctness must be detected as an assessment
    # reference just like the bare form -- otherwise it bypasses the assessment-read gate on
    # QueryTraceMetrics / CalculateTraceFilterCorrelation.
    from mlflow.server import auth

    assert auth._filter_references_assessments("feedback.correctness > 0.5")
    assert auth._filter_references_assessments("`feedback`.correctness > 0.5")
    assert auth._filter_references_assessments("`assessment`.foo = 'x'")
    assert auth._filter_references_assessments("`expectation`.bar < 1")
    # issue.id resolves from assessment rows (assessment_type == "issue"), so it is
    # assessment-derived too (review finding).
    assert auth._filter_references_assessments("issue.id = 'i-1'")
    assert auth._filter_references_assessments("`ISSUE`.id = 'i-1'")
    # Non-assessment fields and mere substring hits are not flagged (fail-safe, not
    # over-eager): a leading word char before the identifier must not match.
    assert not auth._filter_references_assessments("attributes.status = 'OK'")
    assert not auth._filter_references_assessments("myfeedback.value = 1")
    assert not auth._filter_references_assessments("myissue.id = 'x'")


@pytest.mark.parametrize(
    ("_case", "source_run_id", "run_permission", "expected"),
    [
        ("no_source_run", "", None, True),
        ("readable_source_run", "run-1", "READ", True),
        ("denied_source_run", "run-1", "DENY", False),
        ("missing_source_run", "run-1", "missing", False),
    ],
)
def test_create_logged_model_honors_source_run_deny(
    monkeypatch, _case, source_run_id, run_permission, expected
):
    # CreateLoggedModel persists source_run_id as lineage, so a caller with logged_model
    # EDIT and (run, *, DENY) must not record provenance from a denied run; a nonexistent
    # run fails closed rather than acting as an existence oracle (review finding).
    from mlflow.server.auth.permissions import get_permission

    monkeypatch.setattr(
        auth_module,
        "_get_logged_model_permission_for_experiment",
        lambda _e: SimpleNamespace(can_update=True),
    )

    def fake_run_permission(_run_id):
        if run_permission == "missing":
            raise MlflowException("no run", error_code=RESOURCE_DOES_NOT_EXIST)
        return get_permission(run_permission)

    monkeypatch.setattr(auth_module, "_get_run_permission", fake_run_permission)
    body = {"experiment_id": "e1", "name": "m"}
    if source_run_id:
        body["source_run_id"] = source_run_id
    with auth_module.app.test_request_context("/x", method="POST", json=body):
        assert auth_module.validate_can_create_logged_model() is expected


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_trace_metrics_and_correlation_honor_assessment_deny(client):
    # QueryTraceMetrics(view_type=ASSESSMENTS) and CalculateTraceFilterCorrelation with an
    # assessment-referencing filter return assessment-*derived* data (never emitting an
    # assessment object, so redaction can't apply), so they must gate on the assessment tier
    # in addition to trace READ. A caller with trace READ (via experiment READ) but
    # (assessment, *, DENY) must be blocked from the assessment-derived variants and allowed
    # the trace-only variants.
    base = client.tracking_uri
    owner, owner_pw = create_user(base)
    user, pw = create_user(base)
    exp_id = requests.post(
        f"{base}/api/2.0/mlflow/experiments/create",
        json={"name": "trace-metrics-deny-exp"},
        auth=(owner, owner_pw),
    ).json()["experiment_id"]
    # experiment READ grants trace READ and assessment READ via fallback ...
    grant_role_permission(base, user, "experiment", exp_id, "READ")
    # ... then an explicit assessment DENY overrides only the assessment tier.
    grant_role_permission(base, user, "assessment", "*", "DENY")

    metrics_url = f"{base}/api/3.0/mlflow/traces/metrics"
    corr_url = f"{base}/api/3.0/mlflow/traces/calculate-filter-correlation"

    # QueryTraceMetrics: the ASSESSMENTS view is assessment-derived -> denied.
    assert (
        requests.post(
            metrics_url,
            json={"experiment_ids": [exp_id], "view_type": "ASSESSMENTS", "metric_name": "count"},
            auth=(user, pw),
        ).status_code
        == 403
    )
    # A non-assessment (TRACES) view needs only trace READ -> not blocked by assessment DENY.
    assert (
        requests.post(
            metrics_url,
            json={"experiment_ids": [exp_id], "view_type": "TRACES", "metric_name": "count"},
            auth=(user, pw),
        ).status_code
        != 403
    )

    # CalculateTraceFilterCorrelation: an assessment-referencing filter is assessment-derived.
    assert (
        requests.post(
            corr_url,
            json={
                "experiment_ids": [exp_id],
                "filter_string1": "feedback.correctness > 0.5",
                "filter_string2": 'trace.status = "OK"',
            },
            auth=(user, pw),
        ).status_code
        == 403
    )
    # A trace-only correlation touches no assessment data -> allowed.
    assert (
        requests.post(
            corr_url,
            json={
                "experiment_ids": [exp_id],
                "filter_string1": 'trace.status = "OK"',
                "filter_string2": 'trace.status = "ERROR"',
            },
            auth=(user, pw),
        ).status_code
        != 403
    )


def test_trace_assessment_redactor_is_query_bounded(monkeypatch):
    # The trace-assessment redactor must build the assessment read predicate ONCE per
    # response, not once per experiment, so redacting a batch spanning many experiments stays
    # O(1) authorization queries. Guards against a caching-only regression (which would still
    # build/query per distinct experiment).
    from mlflow.protos.service_pb2 import SearchTracesV3
    from mlflow.server import auth
    from mlflow.utils.proto_json_utils import message_to_json

    builds = {"count": 0}

    def fake_predicate(username, resource_type, parent_type=None):
        builds["count"] += 1
        return lambda _exp_id: False  # deny -> clears assessments, exercises the path

    monkeypatch.setattr(auth, "_role_based_read_predicate", fake_predicate)
    monkeypatch.setattr(auth, "sender_is_admin", lambda: False)
    monkeypatch.setattr(auth, "authenticate_request", lambda: type("U", (), {"username": "u"})())

    response_message = SearchTracesV3.Response()
    for i in range(10):  # 10 DISTINCT experiments
        trace_info = response_message.traces.add()
        trace_info.trace_location.mlflow_experiment.experiment_id = f"exp-{i}"

    class _FakeResp:
        def __init__(self, msg):
            self.json = json.loads(message_to_json(msg))
            self.data = None

    resp = _FakeResp(response_message)
    auth._redact_trace_assessments_response(resp, SearchTracesV3, lambda m: list(m.traces))

    # Built once per resource type for the whole response (assessment + prompt_version),
    # never once per experiment/row.
    assert builds["count"] == 2


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_issue_detection_invoke_requires_use_permission_on_secret(client):
    # issues/invoke decrypts the referenced gateway secret into the job environment, so
    # UPDATE on the caller's own experiment must not be enough to consume someone else's
    # secret (GHSA-2m86-c5q7-rxgr).
    base = client.tracking_uri
    owner, owner_pw = create_user(base)
    attacker, attacker_pw = create_user(base)

    resp = requests.post(
        f"{base}/api/3.0/mlflow/gateway/secrets/create",
        json={
            "secret_name": "owner-openai-key",
            "secret_value": {"api_key": "sk-owner"},
            "provider": "openai",
        },
        auth=(owner, owner_pw),
    )
    resp.raise_for_status()
    secret_id = resp.json()["secret"]["secret_id"]

    attacker_exp_id = requests.post(
        f"{base}/api/2.0/mlflow/experiments/create",
        json={"name": "attacker-exp"},
        auth=(attacker, attacker_pw),
    ).json()["experiment_id"]

    payload = {
        "experiment_id": attacker_exp_id,
        "trace_ids": ["tr-1"],
        "categories": ["x"],
        "provider": "openai",
        "model": "gpt-4o",
        "secret_id": secret_id,
    }
    url = f"{base}/ajax-api/3.0/mlflow/issues/invoke"

    # UPDATE on the experiment alone: denied at the secret boundary.
    resp = requests.post(url, json=payload, auth=(attacker, attacker_pw))
    assert resp.status_code == 403

    # READ on the secret only exposes masked metadata; consuming it still requires USE.
    grant_role_permission(base, attacker, "gateway_secret", secret_id, "READ")
    resp = requests.post(url, json=payload, auth=(attacker, attacker_pw))
    assert resp.status_code == 403

    # Unknown secret id: fail closed rather than surface a permission oracle.
    resp = requests.post(
        url, json={**payload, "secret_id": "s-does-not-exist"}, auth=(attacker, attacker_pw)
    )
    assert resp.status_code == 403

    # With USE the auth gate passes. The handler may still fail later (e.g. the job
    # backend isn't wired in this test env), so only assert it is no longer a 403.
    grant_role_permission(base, attacker, "gateway_secret", secret_id, "USE")
    resp = requests.post(url, json=payload, auth=(attacker, attacker_pw))
    assert resp.status_code != 403

    # The secret owner (MANAGE) passes the gate against their own experiment.
    owner_exp_id = requests.post(
        f"{base}/api/2.0/mlflow/experiments/create",
        json={"name": "owner-exp"},
        auth=(owner, owner_pw),
    ).json()["experiment_id"]
    resp = requests.post(
        url, json={**payload, "experiment_id": owner_exp_id}, auth=(owner, owner_pw)
    )
    assert resp.status_code != 403


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
@pytest.mark.parametrize(
    ("path", "extra_payload"),
    [
        ("genai/evaluate/invoke", {"serialized_scorers": ['{"name": "judge"}']}),
        (
            "issues/invoke",
            {
                "categories": ["correctness"],
                # provider+model path: an endpoint_name would now be denied at the
                # VALIDATOR (gateway USE, fail-closed for a nonexistent endpoint), which
                # has its own test -- this test targets the HANDLER's trace binding.
                "provider": "openai",
                "model": "gpt-4o",
            },
        ),
    ],
)
def test_invoke_endpoints_reject_foreign_trace_ids(client, path, extra_payload):
    # UPDATE on the caller's own experiment must not be enough to process a trace from an
    # experiment the caller cannot read (GHSA-v7w2-x9m4-3743).
    base = client.tracking_uri
    victim, victim_pw = create_user(base)
    attacker, attacker_pw = create_user(base)

    victim_exp_id = requests.post(
        f"{base}/api/2.0/mlflow/experiments/create",
        json={"name": "victim-exp"},
        auth=(victim, victim_pw),
    ).json()["experiment_id"]
    victim_trace_id = _create_trace(base, victim_exp_id, (victim, victim_pw))
    attacker_exp_id = requests.post(
        f"{base}/api/2.0/mlflow/experiments/create",
        json={"name": "attacker-exp"},
        auth=(attacker, attacker_pw),
    ).json()["experiment_id"]

    # Control: the attacker cannot read the victim's trace directly.
    resp = requests.get(
        f"{base}/api/2.0/mlflow/traces/{victim_trace_id}/info", auth=(attacker, attacker_pw)
    )
    assert resp.status_code == 403

    resp = requests.post(
        f"{base}/ajax-api/3.0/mlflow/{path}",
        json={
            "experiment_id": attacker_exp_id,
            "trace_ids": [victim_trace_id],
            **extra_payload,
        },
        auth=(attacker, attacker_pw),
    )
    # The route validator passes (the attacker owns the experiment); the JSON error body
    # shows the rejection comes from the handler binding the trace to that experiment.
    assert resp.status_code == 403
    assert resp.json()["error_code"] == "PERMISSION_DENIED"

    runs = requests.post(
        f"{base}/api/2.0/mlflow/runs/search",
        json={"experiment_ids": [attacker_exp_id]},
        auth=(attacker, attacker_pw),
    )
    runs.raise_for_status()
    assert runs.json().get("runs", []) == []


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_presigned_upload_url_requires_run_update_permission(client):
    # Presigned upload URL grants direct artifact write -> denied without run update.
    base = client.tracking_uri
    owner, owner_pw = create_user(base)
    attacker, attacker_pw = create_user(base)

    exp_id = requests.post(
        f"{base}/api/2.0/mlflow/experiments/create",
        json={"name": "presigned-owner-exp"},
        auth=(owner, owner_pw),
    ).json()["experiment_id"]
    run_id = requests.post(
        f"{base}/api/2.0/mlflow/runs/create",
        json={"experiment_id": exp_id},
        auth=(owner, owner_pw),
    ).json()["run"]["info"]["run_id"]

    resp = requests.post(
        f"{base}/api/2.0/mlflow/artifacts/presigned-upload-url",
        json={"run_id": run_id, "path": "model.pkl"},
        auth=(attacker, attacker_pw),
    )
    assert resp.status_code == 403

    # The run owner (MANAGE) passes the auth gate; non-403 confirms the validator is keyed to
    # run update, not blanket-denying (the local artifact repo handler then errors, as in the
    # presigned-download twin).
    resp = requests.post(
        f"{base}/api/2.0/mlflow/artifacts/presigned-upload-url",
        json={"run_id": run_id, "path": "model.pkl"},
        auth=(owner, owner_pw),
    )
    assert resp.status_code != 403


_CHILD_VALIDATOR_OUTCOMES = [
    ("validate_can_create_run", "_get_permission_from_experiment_id_for_run", "can_update"),
    ("validate_can_read_run", "_get_permission_from_run_id", "can_read"),
    ("validate_can_update_run", "_get_permission_from_run_id", "can_update"),
    ("validate_can_read_run_artifact", "_get_permission_from_run_id_or_uuid", "can_read"),
    ("validate_can_update_run_artifact", "_get_permission_from_run_id_or_uuid", "can_update"),
    ("validate_can_start_trace", "_get_trace_permission_for_experiment", "can_update"),
    ("validate_can_read_trace_by_request_id", "_get_trace_permission", "can_read"),
    ("validate_can_read_trace_by_trace_id", "_get_trace_permission", "can_read"),
    ("validate_can_update_trace_by_request_id", "_get_trace_permission", "can_update"),
    ("validate_can_update_trace_by_trace_id", "_get_trace_permission", "can_update"),
    ("validate_can_read_assessment", "_get_assessment_permission_from_trace_id", "can_read"),
    ("validate_can_update_assessment", "_get_assessment_permission_from_trace_id", "can_update"),
    ("validate_can_read_logged_model", "_get_permission_from_model_id", "can_read"),
    ("validate_can_update_logged_model", "_get_permission_from_model_id", "can_update"),
    ("validate_can_delete_logged_model", "_get_permission_from_model_id", "can_delete"),
    (
        # validate_can_add_items_to_review_queue left this harness: it now gates on
        # queue EDIT AND trace READ (dedicated test below).
        "validate_can_remove_items_from_review_queue",
        "_get_permission_from_review_queue_id",
        "can_update",
    ),
    (
        "_validate_can_read_model_version_or_prompt_version",
        "_get_model_version_permission_from_registered_model_or_prompt_name",
        "can_read",
    ),
    (
        "_validate_can_update_model_version_or_prompt_version",
        "_get_model_version_permission_from_registered_model_or_prompt_name",
        "can_update",
    ),
    (
        "_validate_can_delete_model_version_or_prompt_version",
        "_get_model_version_permission_from_registered_model_or_prompt_name",
        "can_delete",
    ),
    (
        "validate_can_read_scorer_version",
        "_get_permission_from_scorer_version_name",
        "can_read",
    ),
    (
        "validate_can_update_scorer_version",
        "_get_permission_from_scorer_version_name",
        "can_update",
    ),
]


@pytest.mark.parametrize(
    ("validator_name", "resolver_name", "capability"), _CHILD_VALIDATOR_OUTCOMES
)
@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("parent_inherited", True),
        ("no_parent_or_child", False),
        ("child_override", True),
        ("child_deny", False),
    ],
)
def test_child_validator_outcomes(
    monkeypatch, validator_name, resolver_name, capability, case, expected
):
    from mlflow.server.auth.permissions import get_permission

    allow_permission = "MANAGE" if capability == "can_delete" else "EDIT"
    permission = allow_permission if expected else "NO_PERMISSIONS"
    if case == "child_deny":
        permission = "DENY"
    monkeypatch.setattr(auth_module, "_get_request_param", lambda _name: "resource-id")
    monkeypatch.setattr(auth_module, resolver_name, lambda *_args: get_permission(permission))

    assert getattr(auth_module, validator_name)() is expected


@pytest.mark.parametrize(
    ("_case", "permission", "expected"),
    [
        ("parent_inherited", "EDIT", True),
        ("no_parent_or_child", "NO_PERMISSIONS", False),
        ("child_override", "EDIT", True),
        ("child_deny", "DENY", False),
    ],
)
def test_create_review_queue_outcomes(monkeypatch, _case, permission, expected):
    from mlflow.server.auth.permissions import get_permission

    reject = mock.Mock()
    monkeypatch.setattr(auth_module, "_get_request_param", lambda _name: "experiment-id")
    monkeypatch.setattr(
        auth_module,
        "_get_review_queue_permission_for_experiment",
        lambda _experiment_id: get_permission(permission),
    )
    monkeypatch.setattr(auth_module, "_reject_create_review_queue_shadowing_user", reject)

    assert auth_module.validate_can_create_review_queue() is expected
    assert reject.called is expected


@pytest.mark.parametrize(
    ("_case", "permission", "expected"),
    [
        ("parent_inherited", "EDIT", True),
        ("no_parent_or_child", "NO_PERMISSIONS", False),
        ("child_override", "EDIT", True),
        ("child_deny", "DENY", False),
    ],
)
def test_create_model_version_outcomes(monkeypatch, _case, permission, expected):
    from mlflow.server.auth.permissions import get_permission

    monkeypatch.setattr(
        auth_module,
        "_get_model_version_permission_from_registered_model_or_prompt_name",
        lambda: get_permission(permission),
    )

    with auth_module.app.test_request_context("/model-versions/create", method="POST", json={}):
        assert auth_module.validate_can_create_model_version() is expected


@pytest.mark.parametrize(
    ("_case", "permission", "expected"),
    [
        ("parent_inherited", "EDIT", True),
        ("no_parent_or_child", "NO_PERMISSIONS", False),
        ("child_override", "EDIT", True),
        ("child_deny", "DENY", False),
    ],
)
def test_start_trace_v3_outcomes(monkeypatch, _case, permission, expected):
    from mlflow.server.auth.permissions import get_permission

    monkeypatch.setattr(
        auth_module,
        "_get_trace_permission_for_experiment",
        lambda _experiment_id: get_permission(permission),
    )
    payload = {
        "trace": {"trace_info": {"trace_location": {"mlflow_experiment": {"experiment_id": "e1"}}}}
    }
    with auth_module.app.test_request_context("/traces/start", method="POST", json=payload):
        assert auth_module.validate_can_start_trace_v3() is expected


@pytest.mark.parametrize(
    ("_case", "scorer_tier", "version_denied", "expected"),
    [
        ("scorer_tier_edit", "EDIT", False, True),
        ("scorer_tier_manage", "MANAGE", False, True),
        ("scorer_tier_read_blocks", "READ", False, False),
        ("scorer_tier_deny_blocks", "DENY", False, False),
        ("version_wildcard_deny", "MANAGE", True, False),
    ],
)
def test_register_existing_scorer_version_outcomes(
    monkeypatch, _case, scorer_tier, version_denied, expected
):
    # OWNER RULING: a version-add on an EXISTING scorer is a scorer-version create and
    # requires the SCORER tier's can_update (EDIT+), resolved through the normal fold --
    # a per-name scorer grant is authoritative (DENY or a below-EDIT grant blocks), and
    # with no scorer grant the tier falls back to the experiment. (scorer_version, *,
    # DENY) still vetoes. The create-veto is NOT consulted and the parent-created flag
    # is NOT set (no MANAGE upsert for version-adders).
    from mlflow.server.auth.permissions import get_permission

    monkeypatch.setattr(
        auth_module,
        "_get_request_param",
        lambda name: {"experiment_id": "e1", "name": "s"}[name],
    )
    monkeypatch.setattr(
        auth_module,
        "_get_tracking_store",
        lambda: SimpleNamespace(get_scorer=lambda _experiment_id, _name: SimpleNamespace()),
    )
    monkeypatch.setattr(auth_module, "authenticate_request", lambda: SimpleNamespace(username="u"))
    monkeypatch.setattr(
        auth_module,
        "_register_scorer_version_permission",
        lambda _e, _n: get_permission(scorer_tier),
    )
    monkeypatch.setattr(auth_module, "_scorer_version_deny_active", lambda _e: version_denied)
    exp_perm = mock.Mock()
    monkeypatch.setattr(auth_module, "_get_experiment_permission", exp_perm)
    create_deny = mock.Mock()
    monkeypatch.setattr(auth_module, "_top_level_create_denied", create_deny)

    with auth_module.app.test_request_context("/scorers", method="POST"):
        assert auth_module.validate_can_register_scorer() is expected
        assert getattr(auth_module.g, "mlflow_creates_scorer_parent", False) is False
    # The existing branch decides on the scorer fold (experiment fallback lives INSIDE
    # it); the direct experiment check and the create veto belong to the create branch.
    exp_perm.assert_not_called()
    create_deny.assert_not_called()


def test_deny_veto_rejects_iff_any_permission_is_deny():
    from mlflow.server.auth.permissions import DENY, MANAGE, NO_PERMISSIONS, READ

    assert auth_module._deny_veto(DENY) is True
    assert auth_module._deny_veto(READ, DENY) is True  # any DENY vetoes
    assert auth_module._deny_veto(READ, MANAGE, NO_PERMISSIONS) is False
    assert auth_module._deny_veto(None) is False  # unresolved is not a veto
    assert auth_module._deny_veto() is False


@pytest.mark.parametrize(
    ("has_assessments", "assessment_can_update", "expected"),
    [(False, None, True), (True, True, True), (True, False, False)],
)
def test_start_trace_v3_embedded_assessments_require_assessment_update(
    monkeypatch, has_assessments, assessment_can_update, expected
):
    # StartTraceV3 persists embedded TraceInfoV3.assessments (including on the
    # existing-trace merge path), so writing them requires the assessment child tier --
    # trace EDIT alone must not create assessments (review finding).
    monkeypatch.setattr(
        auth_module,
        "_get_trace_permission_for_experiment",
        lambda _e: SimpleNamespace(can_update=True),
    )
    calls = {}

    def fake_child_permission(child_type, child_key, experiment_id, username=None):
        calls["args"] = (child_type, child_key, experiment_id)
        return SimpleNamespace(can_update=assessment_can_update)

    monkeypatch.setattr(auth_module, "_experiment_child_permission", fake_child_permission)
    trace_info = {"trace_location": {"mlflow_experiment": {"experiment_id": "9"}}}
    if has_assessments:
        trace_info["assessments"] = [{"assessment_name": "a"}]
    with auth_module.app.test_request_context(
        "/x", method="POST", json={"trace": {"trace_info": trace_info}}
    ):
        assert auth_module.validate_can_start_trace_v3() is expected
    if has_assessments:
        assert calls["args"] == ("assessment", "*", "9")
    else:
        assert "args" not in calls  # no embedded assessments: tier not consulted


@pytest.mark.parametrize("route", ["v2", "v3"])
@pytest.mark.parametrize(
    ("filter_string", "assessment_readable", "expected"),
    [
        ("attributes.status = 'OK'", False, True),  # non-assessment filter: not gated
        ("feedback.correctness > 0.5", True, True),
        ("issue.id = 'i-1'", False, False),
        ("`expectation`.bar < 1", False, False),
        ("assessment.foo = 'x'", False, False),
    ],
)
def test_search_traces_assessment_filters_require_assessment_read(
    monkeypatch, route, filter_string, assessment_readable, expected
):
    # Assessment-backed trace filters execute against assessment data (which rows match
    # leaks denied values), so they require assessment READ on every requested experiment
    # -- the same gate as the metrics/correlation routes (review finding).
    monkeypatch.setattr(auth_module, "_trace_read_predicate", lambda: lambda _e: True)
    monkeypatch.setattr(auth_module, "authenticate_request", lambda: SimpleNamespace(username="u"))
    monkeypatch.setattr(
        auth_module,
        "_role_based_read_predicate",
        lambda _u, rt, parent_type=None: lambda _e: assessment_readable,
    )
    if route == "v2":
        with auth_module.app.test_request_context(
            "/x", query_string={"experiment_ids": "9", "filter": filter_string}
        ):
            assert auth_module.validate_can_search_traces() is expected
    else:
        with auth_module.app.test_request_context(
            "/x",
            method="POST",
            json={
                "locations": [{"mlflow_experiment": {"experiment_id": "9"}}],
                "filter": filter_string,
            },
        ):
            assert auth_module.validate_can_search_traces_v3() is expected


@pytest.mark.parametrize(
    ("run_can_delete", "assessment_can_update", "expected"),
    [(True, True, True), (True, False, False), (False, True, False)],
)
def test_delete_run_requires_assessment_tier(
    monkeypatch, run_can_delete, assessment_can_update, expected
):
    # DeleteRun permanently deletes the run's source-run assessments, so the assessment
    # child tier's mutation capability is required atop run delete; RestoreRun deletes no
    # assessments and keeps the plain run-tier check (review finding).
    monkeypatch.setattr(auth_module, "_get_request_param", lambda _n: "run-1")
    monkeypatch.setattr(
        auth_module,
        "_get_tracking_store",
        lambda: SimpleNamespace(
            get_run=lambda _rid: SimpleNamespace(info=SimpleNamespace(experiment_id="9"))
        ),
    )

    def fake_child_permission(child_type, child_key, experiment_id, username=None):
        assert experiment_id == "9"
        if child_type == "run":
            return SimpleNamespace(can_delete=run_can_delete)
        assert (child_type, child_key) == ("assessment", "*")
        return SimpleNamespace(can_update=assessment_can_update)

    monkeypatch.setattr(auth_module, "_experiment_child_permission", fake_child_permission)
    assert auth_module.validate_can_delete_run() is expected

    # RestoreRun: run tier only, and the route map keeps the two validators split.
    monkeypatch.setattr(
        auth_module,
        "_get_permission_from_run_id",
        lambda: SimpleNamespace(can_delete=run_can_delete),
    )
    assert auth_module.validate_can_restore_run() is run_can_delete
    from mlflow.protos.service_pb2 import DeleteRun, RestoreRun

    assert auth_module.BEFORE_REQUEST_HANDLERS[DeleteRun] is auth_module.validate_can_delete_run
    assert auth_module.BEFORE_REQUEST_HANDLERS[RestoreRun] is auth_module.validate_can_restore_run


@pytest.mark.parametrize(
    ("exp_can_delete", "run_can_delete", "assessment_can_update", "delete_ok", "restore_ok"),
    [
        (True, True, True, True, True),
        (True, True, False, False, True),  # assessment veto blocks delete, not restore
        (True, False, True, False, False),  # run tier gates both lifecycle directions
        (False, True, True, False, False),
    ],
)
def test_experiment_lifecycle_requires_child_tiers(
    monkeypatch, exp_can_delete, run_can_delete, assessment_can_update, delete_ok, restore_ok
):
    # DeleteExperiment marks every run deleted (and deletes their source-run assessments);
    # RestoreExperiment restores every run. Both must honor the authoritative run tier,
    # and delete must additionally honor the assessment tier (review finding).
    monkeypatch.setattr(auth_module, "_get_request_param", lambda _n: "9")
    monkeypatch.setattr(
        auth_module,
        "_get_permission_from_experiment_id",
        lambda: SimpleNamespace(can_delete=exp_can_delete),
    )

    def fake_child_permission(child_type, child_key, experiment_id, username=None):
        assert (child_key, experiment_id) == ("*", "9")
        if child_type == "run":
            return SimpleNamespace(can_delete=run_can_delete)
        assert child_type == "assessment"
        return SimpleNamespace(can_update=assessment_can_update)

    monkeypatch.setattr(auth_module, "_experiment_child_permission", fake_child_permission)
    assert auth_module.validate_can_delete_experiment_lifecycle() is delete_ok
    assert auth_module.validate_can_restore_experiment_lifecycle() is restore_ok
    from mlflow.protos.service_pb2 import DeleteExperiment, RestoreExperiment

    assert (
        auth_module.BEFORE_REQUEST_HANDLERS[DeleteExperiment]
        is auth_module.validate_can_delete_experiment_lifecycle
    )
    assert (
        auth_module.BEFORE_REQUEST_HANDLERS[RestoreExperiment]
        is auth_module.validate_can_restore_experiment_lifecycle
    )


@pytest.mark.parametrize(
    ("trace_can_delete", "assessment_can_update", "queue_can_update", "expected"),
    [
        (True, True, True, True),
        (True, False, True, False),
        (True, True, False, False),
        (False, True, True, False),
    ],
)
def test_delete_traces_requires_assessment_and_queue_tiers(
    monkeypatch, trace_can_delete, assessment_can_update, queue_can_update, expected
):
    # DeleteTraces cascades assessments and prunes review-queue items, so both child
    # tiers' mutation capability is required atop trace delete (review finding). All
    # affected rows live in the request's experiment and child grants are wildcard-only,
    # so the pre-request experiment-scoped checks govern every selected row.
    monkeypatch.setattr(auth_module, "_get_request_param", lambda _n: "9")
    monkeypatch.setattr(
        auth_module,
        "_get_trace_permission_for_experiment",
        lambda _e: SimpleNamespace(can_delete=trace_can_delete),
    )

    def fake_child_permission(child_type, child_key, experiment_id, username=None):
        assert (child_key, experiment_id) == ("*", "9")
        if child_type == "assessment":
            return SimpleNamespace(can_update=assessment_can_update)
        assert child_type == "review_queue"
        return SimpleNamespace(can_update=queue_can_update)

    monkeypatch.setattr(auth_module, "_experiment_child_permission", fake_child_permission)
    assert auth_module.validate_can_delete_traces() is expected


def test_redact_run_model_io_on_logged_model_deny(monkeypatch):
    # GetRun/SearchRuns serialize inputs.model_inputs / outputs.model_outputs; a run
    # reader with a logged_model DENY must not enumerate denied model ids through them.
    # EXPLICIT AUTH POLICY (findings F7/F8, flagged gap U2): strictly batched resolution
    # scoped to the response's experiments, NO per-id fallback -- links that don't resolve
    # (cross-experiment or missing) are omitted from the embedded view fail-closed, while
    # the models stay reachable via point routes.
    from mlflow.protos.service_pb2 import GetRun
    from mlflow.store.entities.paged_list import PagedList

    msg = GetRun.Response()
    msg.run.info.experiment_id = "9"
    msg.run.inputs.model_inputs.add().model_id = "m-local"  # exp 9: readable, kept
    msg.run.inputs.model_inputs.add().model_id = "m-foreign"  # exp 13: outside scope
    msg.run.outputs.model_outputs.add().model_id = "m-denied"  # unresolved: hidden
    msg.run.outputs.model_outputs.add().model_id = "m-missing"  # unresolved: hidden

    search_calls = []

    def fake_search_logged_models(experiment_ids, filter_string, max_results):
        search_calls.append(filter_string)
        assert experiment_ids == ["9"]  # scoped to the response's experiments
        return PagedList(
            [SimpleNamespace(model_id="m-local", experiment_id="9")]
            if "m-local" in filter_string
            else [],
            None,
        )

    def fail_point_lookup(_model_id):
        pytest.fail("no per-id fallback: the policy is strictly batched")

    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    monkeypatch.setattr(auth_module, "authenticate_request", lambda: SimpleNamespace(username="u"))
    monkeypatch.setattr(
        auth_module,
        "_get_tracking_store",
        lambda: SimpleNamespace(
            search_logged_models=fake_search_logged_models,
            get_logged_model=fail_point_lookup,
        ),
    )
    monkeypatch.setattr(
        auth_module,
        "_role_based_read_predicate",
        lambda _u, rt, parent_type=None: lambda exp_id: exp_id in ("9", "13"),
    )
    resp = _fake_resp(msg)
    auth_module.redact_get_run_model_io(resp)
    out = GetRun.Response()
    auth_module.parse_dict(json.loads(resp.data), out)
    assert [m.model_id for m in out.run.inputs.model_inputs] == ["m-local"]
    assert len(out.run.outputs.model_outputs) == 0
    assert len(search_calls) == 1


def test_run_model_io_filter_query_count_bounded_at_scale(monkeypatch):
    # 450 distinct model ids across a response resolve in exactly
    # ceil(450 / _LOGGED_MODEL_LOOKUP_CHUNK) bulk searches -- no per-id lookups at any
    # size (the chosen policy's 200-id call-count bound, review finding F7).
    import math

    from mlflow.protos.service_pb2 import SearchRuns
    from mlflow.store.entities.paged_list import PagedList

    n = 450
    msg = SearchRuns.Response()
    run = msg.runs.add()
    run.info.experiment_id = "9"
    for i in range(n):
        run.inputs.model_inputs.add().model_id = f"m-{i:04d}"

    search_calls = []

    def fake_search_logged_models(experiment_ids, filter_string, max_results):
        search_calls.append(filter_string)
        return PagedList([], None)  # nothing resolves: everything is hidden fail-closed

    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    monkeypatch.setattr(auth_module, "authenticate_request", lambda: SimpleNamespace(username="u"))
    monkeypatch.setattr(
        auth_module,
        "_get_tracking_store",
        lambda: SimpleNamespace(
            search_logged_models=fake_search_logged_models,
            get_logged_model=lambda _mid: pytest.fail("no per-id fallback"),
        ),
    )
    monkeypatch.setattr(
        auth_module,
        "_role_based_read_predicate",
        lambda _u, rt, parent_type=None: lambda _e: True,
    )
    resp = _fake_resp(msg)
    auth_module.redact_search_runs_model_io(resp)
    out = SearchRuns.Response()
    auth_module.parse_dict(json.loads(resp.data), out)
    assert len(out.runs[0].inputs.model_inputs) == 0
    assert len(search_calls) == math.ceil(n / auth_module._LOGGED_MODEL_LOOKUP_CHUNK)


def test_redact_prompt_optimization_jobs_response(monkeypatch):
    # Job responses expose run/prompt/scorer/dataset identifiers; each is dropped when the
    # caller's corresponding tier can't read it, while instantiable built-in scorers are
    # retained (review finding). BOUNDED DATASET POLICY (review finding F4): the Search
    # collection is unpaginated, so config.dataset_id is cleared OUTRIGHT for non-admins
    # with ZERO per-dataset authorization queries at any scale; the point check lives on
    # the Get spelling.
    from mlflow.protos.service_pb2 import SearchPromptOptimizationJobs

    msg = SearchPromptOptimizationJobs.Response()
    job = msg.jobs.add()
    job.experiment_id = "9"
    job.run_id = "run-1"
    job.source_prompt_uri = "prompts:/p-src/1"
    job.optimized_prompt_uri = "prompts:/p-opt/2"
    job.config.scorers.extend(["Safety", "custom-scorer"])
    job.config.dataset_id = "ds-1"
    # Many DISTINCT dataset ids: the scale case F4 requires to stay query-free.
    for i in range(300):
        extra = msg.jobs.add()
        extra.experiment_id = "9"
        extra.config.dataset_id = f"ds-{i + 2:04d}"

    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    monkeypatch.setattr(auth_module, "authenticate_request", lambda: SimpleNamespace(username="u"))
    monkeypatch.setattr(auth_module.store, "_scorer_pattern", lambda e, n: f"{e}/{n}")
    monkeypatch.setattr(
        auth_module,
        "_dataset_read_allowed",
        lambda _d, _u: pytest.fail("Search must not issue per-dataset authorization queries"),
    )

    def fake_predicate(_u, resource_type, parent_type=None):
        # run tier denied; prompt_version readable only for p-opt; scorer_version denied.
        if resource_type == "run":
            return lambda _e: False
        if resource_type == "prompt_version":
            return lambda name: name == "p-opt"
        assert resource_type == "scorer_version"
        return lambda _key: False

    monkeypatch.setattr(auth_module, "_role_based_read_predicate", fake_predicate)
    resp = _fake_resp(msg)
    auth_module.redact_search_prompt_optimization_jobs(resp)
    out = SearchPromptOptimizationJobs.Response()
    auth_module.parse_dict(json.loads(resp.data), out)
    redacted = out.jobs[0]
    assert redacted.run_id == ""
    assert redacted.source_prompt_uri == ""
    assert redacted.optimized_prompt_uri == "prompts:/p-opt/2"
    # The built-in survives; the unreadable registered scorer is dropped.
    assert list(redacted.config.scorers) == ["Safety"]
    # Every dataset id is cleared, with zero authorization queries (the pytest.fail stub).
    assert all(j.config.dataset_id == "" for j in out.jobs)


def test_redact_get_prompt_optimization_job_dataset_id(monkeypatch):
    # The Get spelling (exactly one job) keeps the POINT dataset check: cleared when the
    # all-associated-experiments READ check fails, retained when it passes (review
    # findings F3/F4).
    from mlflow.protos.service_pb2 import GetPromptOptimizationJob

    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    monkeypatch.setattr(auth_module, "authenticate_request", lambda: SimpleNamespace(username="u"))
    monkeypatch.setattr(
        auth_module, "_role_based_read_predicate", lambda *_a, **_k: lambda *_x: True
    )

    for readable, expected in ((False, ""), (True, "ds-1")):
        msg = GetPromptOptimizationJob.Response()
        msg.job.experiment_id = "9"
        msg.job.config.dataset_id = "ds-1"
        monkeypatch.setattr(auth_module, "_dataset_read_allowed", lambda _d, _u, r=readable: r)
        resp = _fake_resp(msg)
        auth_module.redact_get_prompt_optimization_job(resp)
        out = GetPromptOptimizationJob.Response()
        auth_module.parse_dict(json.loads(resp.data), out)
        assert out.job.config.dataset_id == expected


@pytest.mark.parametrize(("scorer_version_denied", "expected"), [(False, True), (True, False)])
def test_composite_routes_honor_scorer_version_deny(monkeypatch, scorer_version_denied, expected):
    # A (scorer_version, *, DENY) vetoes INVOKE_SCORER / INVOKE_GENAI_EVALUATE /
    # CreatePromptOptimizationJob even though no positive scorer_version grant is required
    # (Copilot #32 split verdict). All positive checks are stubbed to pass so the veto is
    # the only deciding factor.
    from mlflow.server.auth.permissions import MANAGE

    monkeypatch.setattr(auth_module, "_get_request_param", lambda _name: "e1")
    monkeypatch.setattr(auth_module, "validate_can_update_experiment", lambda: True)
    monkeypatch.setattr(auth_module, "validate_can_create_run", lambda: True)
    monkeypatch.setattr(auth_module, "_get_trace_permission_for_experiment", lambda _e: MANAGE)
    monkeypatch.setattr(auth_module, "_experiment_child_permission", lambda *a, **k: MANAGE)
    monkeypatch.setattr(
        auth_module, "_scorer_version_deny_active", lambda _e: scorer_version_denied
    )

    with auth_module.app.test_request_context("/x", method="POST", json={}):
        assert auth_module.validate_can_invoke_scorer() is expected
        assert auth_module.validate_can_invoke_genai_evaluate() is expected
        assert auth_module.validate_can_create_prompt_optimization_job() is expected


@pytest.mark.parametrize(("scorer_version_denied", "expected"), [(False, True), (True, False)])
def test_register_new_scorer_honors_scorer_version_deny(
    monkeypatch, scorer_version_denied, expected
):
    # Creating a brand-new scorer is gated on experiment.can_update (its pre-RFC contract),
    # but a (scorer_version, *, DENY) still vetoes writing version 1.
    from mlflow.server.auth.permissions import MANAGE

    def _raise_not_found(_experiment_id, _name):
        raise MlflowException("no scorer", error_code=RESOURCE_DOES_NOT_EXIST)

    monkeypatch.setattr(
        auth_module,
        "_get_request_param",
        lambda name: {"experiment_id": "e1", "name": "s"}[name],
    )
    monkeypatch.setattr(
        auth_module, "_get_tracking_store", lambda: SimpleNamespace(get_scorer=_raise_not_found)
    )
    monkeypatch.setattr(auth_module, "_get_experiment_permission", lambda _e, _u: MANAGE)
    monkeypatch.setattr(auth_module, "authenticate_request", lambda: SimpleNamespace(username="u"))
    monkeypatch.setattr(
        auth_module, "_scorer_version_deny_active", lambda _e: scorer_version_denied
    )
    # The new-scorer branch also checks the scorer PARENT create-veto; not under test here.
    monkeypatch.setattr(auth_module, "_top_level_create_denied", lambda _rt, _u: False)

    with auth_module.app.test_request_context("/scorers", method="POST"):
        assert auth_module.validate_can_register_scorer() is expected


@pytest.mark.parametrize(("scorer_parent_denied", "expected"), [(False, True), (True, False)])
def test_register_new_scorer_honors_scorer_parent_deny(monkeypatch, scorer_parent_denied, expected):
    # Creating a brand-new scorer creates the scorer PARENT, so (scorer, *, DENY) must veto it
    # -- a surface distinct from the scorer_version veto (Copilot r4052491497). Experiment
    # EDIT + no scorer_version DENY, so only the parent veto decides.
    from mlflow.server.auth.permissions import MANAGE

    def _raise_not_found(_experiment_id, _name):
        raise MlflowException("no scorer", error_code=RESOURCE_DOES_NOT_EXIST)

    monkeypatch.setattr(
        auth_module,
        "_get_request_param",
        lambda name: {"experiment_id": "e1", "name": "s"}[name],
    )
    monkeypatch.setattr(
        auth_module, "_get_tracking_store", lambda: SimpleNamespace(get_scorer=_raise_not_found)
    )
    monkeypatch.setattr(auth_module, "_get_experiment_permission", lambda _e, _u: MANAGE)
    monkeypatch.setattr(auth_module, "authenticate_request", lambda: SimpleNamespace(username="u"))
    monkeypatch.setattr(auth_module, "_scorer_version_deny_active", lambda _e: False)
    monkeypatch.setattr(
        auth_module,
        "_top_level_create_denied",
        lambda rt, _u: rt == "scorer" and scorer_parent_denied,
    )

    with auth_module.app.test_request_context("/scorers", method="POST"):
        assert auth_module.validate_can_register_scorer() is expected


@pytest.mark.parametrize(
    "gateway_type",
    ["gateway_secret", "gateway_endpoint", "gateway_model_definition"],
)
@pytest.mark.parametrize(("self_type_denied", "expected"), [(False, True), (True, False)])
def test_gateway_create_honors_self_type_deny(
    monkeypatch, gateway_type, self_type_denied, expected
):
    # Workspace USE allows creating a gateway resource, but (gateway_<t>, *, DENY) vetoes it
    # (parent mirror of the sub-resource rule; §0.3 decision). The positive gate is stubbed
    # to allow, so the self-type veto is what decides.
    monkeypatch.setattr(auth_module, "authenticate_request", lambda: SimpleNamespace(username="u"))
    # Positive gates: workspace-create (secret) and the referenced-resource USE checks
    # (endpoint/model-definition) all allow, isolating the veto.
    monkeypatch.setattr(auth_module, "_can_create_in_workspace", lambda _u: True)
    monkeypatch.setattr(auth_module, "_user_can_create_in_workspace", lambda: True)
    monkeypatch.setattr(
        auth_module, "_validate_can_use_model_definitions_for_create", lambda _c: True
    )
    monkeypatch.setattr(
        auth_module,
        "_top_level_create_denied",
        lambda rt, _u: rt == gateway_type and self_type_denied,
    )

    validators = {
        "gateway_secret": auth_module.validate_can_create_gateway_secret,
        "gateway_endpoint": auth_module.validate_can_create_gateway_endpoint,
        "gateway_model_definition": auth_module.validate_can_create_gateway_model_definition,
    }
    # model_definition with no secret_id short-circuits to True after the veto, which is the
    # path that isolates the self-type veto.
    with auth_module.app.test_request_context("/gateway", method="POST", json={}):
        assert validators[gateway_type]() is expected


@pytest.mark.parametrize(
    ("is_prompt_body", "denied_type", "expected"),
    [
        # Creating a registered model: only (registered_model, *, DENY) blocks it.
        (False, "registered_model", False),
        (False, "prompt", True),  # a prompt DENY must NOT block a plain model create
        # Creating a prompt: only (prompt, *, DENY) blocks it.
        (True, "prompt", False),
        (True, "registered_model", True),  # a model DENY must NOT block a prompt create
    ],
)
def test_create_registered_model_vetoes_only_created_type(
    monkeypatch, is_prompt_body, denied_type, expected
):
    # Copilot r4052491505: (prompt, *, DENY) must not block creating an ordinary registered
    # model, and (registered_model, *, DENY) must not block creating a prompt. The request's
    # is_prompt tag classifies which type is being created; only that type's DENY vetoes.
    monkeypatch.setattr(auth_module, "_user_can_create_in_workspace", lambda: True)
    monkeypatch.setattr(auth_module, "authenticate_request", lambda: SimpleNamespace(username="u"))
    monkeypatch.setattr(auth_module, "_top_level_create_denied", lambda rt, _u: rt == denied_type)

    tags = (
        [{"key": IS_PROMPT_TAG_KEY, "value": "true"}]
        if is_prompt_body
        else [{"key": "some.other.tag", "value": "x"}]
    )
    with auth_module.app.test_request_context(
        "/registered-models/create", method="POST", json={"name": "m", "tags": tags}
    ):
        assert auth_module.validate_can_create_registered_model() is expected


@pytest.mark.parametrize(
    ("_case", "permission", "expected"),
    [
        ("parent_inherited", "EDIT", True),
        ("no_parent_or_child", "NO_PERMISSIONS", False),
        ("child_override", "EDIT", True),
        ("child_deny", "DENY", False),
    ],
)
def test_mcp_server_version_create_outcomes(monkeypatch, _case, permission, expected):
    from mlflow.server.auth.permissions import get_permission

    monkeypatch.setattr(
        auth_module,
        "_get_tracking_store",
        lambda: SimpleNamespace(get_mcp_server=lambda _name: SimpleNamespace()),
    )
    monkeypatch.setattr(
        auth_module,
        "_get_mcp_server_version_permission",
        lambda _name, _username: get_permission(permission),
    )
    validator = auth_module._get_mcp_server_validator(
        "/api/3.0/mlflow/mcp-servers/namespace/server/versions"
    )

    request = SimpleNamespace(method="POST", state=SimpleNamespace())
    assert asyncio.run(validator("user", request)) is expected


@pytest.mark.parametrize(
    ("_case", "permission", "expected"),
    [
        ("parent_inherited", "EDIT", True),
        ("no_parent_or_child", "NO_PERMISSIONS", False),
        ("child_override", "EDIT", True),
        ("child_deny", "DENY", False),
    ],
)
def test_create_logged_model_outcomes(monkeypatch, _case, permission, expected):
    from mlflow.server.auth.permissions import get_permission

    monkeypatch.setattr(
        auth_module,
        "_get_logged_model_permission_for_experiment",
        lambda _experiment_id: get_permission(permission),
    )

    # No source_run_id: only the logged_model tier decides (the source-run rule has its
    # own dedicated test).
    with auth_module.app.test_request_context(
        "/x", method="POST", json={"experiment_id": "experiment-id", "name": "m"}
    ):
        assert auth_module.validate_can_create_logged_model() is expected


@pytest.mark.parametrize(
    ("_case", "queue_permission", "trace_permission", "expected"),
    [
        ("queue_edit_trace_readable", "EDIT", "READ", True),
        ("queue_edit_trace_denied", "EDIT", "DENY", False),
        ("no_queue_grant", "NO_PERMISSIONS", "READ", False),
        ("queue_denied", "DENY", "READ", False),
    ],
)
def test_add_items_to_review_queue_requires_trace_read(
    monkeypatch, _case, queue_permission, trace_permission, expected
):
    # Attaching items resolves and persists trace references, so queue EDIT alone is not
    # enough: a (trace, *, DENY) on the queue's experiment vetoes the add (review finding).
    from mlflow.server.auth.permissions import get_permission

    queue = SimpleNamespace(experiment_id=9)
    monkeypatch.setattr(auth_module, "_get_request_param", lambda _n: "q-1")
    monkeypatch.setattr(
        auth_module,
        "_get_tracking_store",
        lambda: SimpleNamespace(get_review_queue=lambda _q: queue),
    )
    monkeypatch.setattr(
        auth_module, "_get_review_queue_permission", lambda _q: get_permission(queue_permission)
    )
    captured = {}

    def fake_trace_permission(experiment_id):
        captured["experiment_id"] = experiment_id
        return get_permission(trace_permission)

    monkeypatch.setattr(auth_module, "_get_trace_permission_for_experiment", fake_trace_permission)
    assert auth_module.validate_can_add_items_to_review_queue() is expected
    if expected or trace_permission == "DENY":
        assert captured["experiment_id"] == "9"


@pytest.mark.parametrize(
    ("_case", "view_allowed", "trace_permission", "expected"),
    [
        ("visible_and_trace_readable", True, "READ", True),
        ("visible_but_trace_denied", True, "DENY", False),
        ("not_visible", False, "READ", False),
    ],
)
def test_list_review_queue_items_requires_trace_read(
    monkeypatch, _case, view_allowed, trace_permission, expected
):
    # Listing items returns the queue's trace references, so queue visibility alone is not
    # enough: a (trace, *, DENY) on the queue's experiment vetoes the list (review finding).
    from mlflow.server.auth.permissions import get_permission

    monkeypatch.setattr(auth_module, "validate_can_view_review_queue", lambda: view_allowed)
    monkeypatch.setattr(auth_module, "_get_request_param", lambda _n: "q-1")
    monkeypatch.setattr(
        auth_module,
        "_get_tracking_store",
        lambda: SimpleNamespace(get_review_queue=lambda _q: SimpleNamespace(experiment_id=9)),
    )
    monkeypatch.setattr(
        auth_module,
        "_get_trace_permission_for_experiment",
        lambda _e: get_permission(trace_permission),
    )
    assert auth_module.validate_can_list_review_queue_items() is expected


@pytest.mark.parametrize("grant_topology", ["parent_and_child", "child_only", "parent_only"])
@pytest.mark.parametrize("workspace_admin", [False, True])
def test_read_predicate_child_deny_and_workspace_admin_precedence(
    monkeypatch, tmp_path, grant_topology, workspace_admin
):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")
    monkeypatch.setattr(
        auth_module,
        "auth_config",
        auth_module.auth_config._replace(default_permission=NO_PERMISSIONS.name),
    )
    store = SqlAlchemyStore()
    store.init_db(f"sqlite:///{tmp_path / 'read-precedence.db'}")
    monkeypatch.setattr(auth_module, "store", store, raising=False)

    user = store.create_user("reader", "supersecurepassword", is_admin=False)
    role = store.create_role(name="role", workspace="default")
    store.assign_role_to_user(user.id, role.id)

    if grant_topology == "parent_and_child":
        store.add_role_permission(role.id, "experiment", "e1", "EDIT")
        store.add_role_permission(role.id, "run", "*", "DENY")
    elif grant_topology == "child_only":
        store.add_role_permission(role.id, "run", "*", "DENY")
    else:
        store.add_role_permission(role.id, "experiment", "e1", "DENY")

    if workspace_admin:
        store.add_role_permission(role.id, "workspace", "*", "MANAGE")

    can_read = auth_module._role_based_read_predicate("reader", "run", parent_type="experiment")
    assert can_read("e1") is workspace_admin


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_start_trace_child_permission_outcomes(client, monkeypatch):
    owner, owner_password = create_user(client.tracking_uri)
    no_grant, no_grant_password = create_user(client.tracking_uri)
    parent_writer, parent_writer_password = create_user(client.tracking_uri)
    child_writer, child_writer_password = create_user(client.tracking_uri)
    denied_writer, denied_writer_password = create_user(client.tracking_uri)

    with User(owner, owner_password, monkeypatch):
        experiment_id = client.create_experiment("trace-child-permission-outcomes")
        anchor_trace_id = _create_trace(client.tracking_uri, experiment_id, (owner, owner_password))

    grant_role_permission(client.tracking_uri, parent_writer, "experiment", experiment_id, "EDIT")
    grant_role_permission(client.tracking_uri, child_writer, "experiment", experiment_id, "READ")
    grant_role_permission(client.tracking_uri, child_writer, "trace", "*", "EDIT")
    grant_role_permission(client.tracking_uri, denied_writer, "experiment", experiment_id, "EDIT")
    grant_role_permission(client.tracking_uri, denied_writer, "trace", "*", "DENY")

    payload = {
        "experiment_id": experiment_id,
        "timestamp_ms": int(time.time() * 1000),
        "execution_time_ms": 10,
        "status": "OK",
        "request_metadata": [],
        "tags": [],
    }
    for username, password in (
        (no_grant, no_grant_password),
        (denied_writer, denied_writer_password),
    ):
        response = requests.post(
            url=client.tracking_uri + "/api/2.0/mlflow/traces",
            json=payload,
            auth=(username, password),
        )
        assert response.status_code == 403
        response = requests.patch(
            url=client.tracking_uri + f"/api/2.0/mlflow/traces/{anchor_trace_id}/tags",
            json={"key": "denied", "value": "true"},
            auth=(username, password),
        )
        assert response.status_code == 403

    for username, password in (
        (parent_writer, parent_writer_password),
        (child_writer, child_writer_password),
    ):
        response = requests.post(
            url=client.tracking_uri + "/api/2.0/mlflow/traces",
            json=payload,
            auth=(username, password),
        )
        assert response.status_code == 200
        response = requests.patch(
            url=client.tracking_uri + f"/api/2.0/mlflow/traces/{anchor_trace_id}/tags",
            json={"key": f"tag_{username}", "value": "true"},
            auth=(username, password),
        )
        assert response.status_code == 200


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_assessment_child_permission_outcomes(client, monkeypatch):
    owner, owner_password = create_user(client.tracking_uri)
    no_grant, no_grant_password = create_user(client.tracking_uri)
    parent_writer, parent_writer_password = create_user(client.tracking_uri)
    child_writer, child_writer_password = create_user(client.tracking_uri)
    denied_writer, denied_writer_password = create_user(client.tracking_uri)

    with User(owner, owner_password, monkeypatch):
        experiment_id = client.create_experiment("assessment-child-permission-outcomes")
        trace_id = _create_trace(client.tracking_uri, experiment_id, (owner, owner_password))

    grant_role_permission(client.tracking_uri, parent_writer, "experiment", experiment_id, "EDIT")
    grant_role_permission(client.tracking_uri, child_writer, "experiment", experiment_id, "READ")
    grant_role_permission(client.tracking_uri, child_writer, "assessment", "*", "EDIT")
    grant_role_permission(client.tracking_uri, denied_writer, "experiment", experiment_id, "EDIT")
    grant_role_permission(client.tracking_uri, denied_writer, "assessment", "*", "DENY")

    def create_assessment(auth, name):
        return requests.post(
            url=client.tracking_uri + f"/api/3.0/mlflow/traces/{trace_id}/assessments",
            json={
                "assessment": {
                    "assessment_name": name,
                    "feedback": {"value": {"rating": 4}},
                    "source": {"source_type": "HUMAN", "source_id": "tester"},
                }
            },
            auth=auth,
        )

    for auth in ((no_grant, no_grant_password), (denied_writer, denied_writer_password)):
        assert create_assessment(auth, f"denied_{auth[0]}").status_code == 403

    for auth, name in (
        ((parent_writer, parent_writer_password), "parent_assessment"),
        ((child_writer, child_writer_password), "child_assessment"),
    ):
        response = create_assessment(auth, name)
        assert response.status_code == 200
        assessment_id = response.json()["assessment"]["assessment_id"]
        response = requests.patch(
            url=client.tracking_uri
            + f"/api/3.0/mlflow/traces/{trace_id}/assessments/{assessment_id}",
            json={
                "assessment": {
                    "assessment_id": assessment_id,
                    "trace_id": trace_id,
                    "assessment_name": f"updated_{name}",
                },
                "update_mask": "assessmentName",
            },
            auth=auth,
        )
        assert response.status_code == 200
        response = requests.delete(
            url=client.tracking_uri
            + f"/api/3.0/mlflow/traces/{trace_id}/assessments/{assessment_id}",
            auth=auth,
        )
        assert response.status_code == 200


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_registered_model_version_child_permission_outcomes(client, monkeypatch):
    owner, owner_password = create_user(client.tracking_uri)
    no_grant, no_grant_password = create_user(client.tracking_uri)
    parent_writer, parent_writer_password = create_user(client.tracking_uri)
    child_writer, child_writer_password = create_user(client.tracking_uri)
    denied_writer, denied_writer_password = create_user(client.tracking_uri)

    with User(owner, owner_password, monkeypatch):
        experiment_id = client.create_experiment("model-version-child-permission-outcomes")
        run = client.create_run(experiment_id)
        model = client.create_registered_model("model-version-child-permission")

    source = f"runs:/{run.info.run_id}/model"
    for username in (parent_writer, child_writer, denied_writer):
        grant_role_permission(client.tracking_uri, username, "experiment", experiment_id, "READ")
    grant_role_permission(
        client.tracking_uri, parent_writer, "registered_model", model.name, "EDIT"
    )
    grant_role_permission(client.tracking_uri, child_writer, "registered_model", model.name, "READ")
    grant_role_permission(
        client.tracking_uri, child_writer, "registered_model_version", "*", "EDIT"
    )
    grant_role_permission(
        client.tracking_uri, denied_writer, "registered_model", model.name, "EDIT"
    )
    grant_role_permission(
        client.tracking_uri, denied_writer, "registered_model_version", "*", "DENY"
    )

    for username, password in (
        (no_grant, no_grant_password),
        (denied_writer, denied_writer_password),
    ):
        with User(username, password, monkeypatch):
            with pytest.raises(MlflowException, match="Permission denied"):
                client.create_model_version(model.name, source, run_id=run.info.run_id)

    for username, password in (
        (parent_writer, parent_writer_password),
        (child_writer, child_writer_password),
    ):
        with User(username, password, monkeypatch):
            version = client.create_model_version(model.name, source, run_id=run.info.run_id)

        assert version.name == model.name


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_prompt_version_child_permission_outcomes(client: MlflowClient, monkeypatch):
    owner, owner_password = create_user(client.tracking_uri)
    no_grant, no_grant_password = create_user(client.tracking_uri)
    parent_writer, parent_writer_password = create_user(client.tracking_uri)
    child_writer, child_writer_password = create_user(client.tracking_uri)
    denied_writer, denied_writer_password = create_user(client.tracking_uri)

    with User(owner, owner_password, monkeypatch):
        prompt = client.create_prompt("prompt-version-child-permission")

    grant_role_permission(client.tracking_uri, parent_writer, "prompt", prompt.name, "EDIT")
    grant_role_permission(client.tracking_uri, child_writer, "prompt", prompt.name, "READ")
    grant_role_permission(client.tracking_uri, child_writer, "prompt_version", "*", "EDIT")
    grant_role_permission(client.tracking_uri, denied_writer, "prompt", prompt.name, "EDIT")
    grant_role_permission(client.tracking_uri, denied_writer, "prompt_version", "*", "DENY")

    for username, password in (
        (no_grant, no_grant_password),
        (denied_writer, denied_writer_password),
    ):
        with User(username, password, monkeypatch):
            with pytest.raises(MlflowException, match="Permission denied"):
                client.create_prompt_version(prompt.name, "hello")

    for username, password in (
        (parent_writer, parent_writer_password),
        (child_writer, child_writer_password),
    ):
        with User(username, password, monkeypatch):
            version = client.create_prompt_version(prompt.name, f"hello {username}")

        assert version.name == prompt.name


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_scorer_version_child_permission_outcomes(client: MlflowClient, monkeypatch):
    # OWNER RULING: version-adds on an existing scorer require the SCORER tier's EDIT --
    # a per-name scorer grant is authoritative (EDIT+ authorizes even WITHOUT an
    # experiment grant; READ or DENY blocks), and with no scorer grant the tier falls
    # back to experiment EDIT. (scorer_version, *, DENY) and (scorer, *, DENY) both veto.
    owner, owner_password = create_user(client.tracking_uri)
    no_grant, no_grant_password = create_user(client.tracking_uri)
    scorer_only_writer, scorer_only_writer_password = create_user(client.tracking_uri)
    child_writer, child_writer_password = create_user(client.tracking_uri)
    exp_writer, exp_writer_password = create_user(client.tracking_uri)
    version_denied_writer, version_denied_writer_password = create_user(client.tracking_uri)
    parent_denied_writer, parent_denied_writer_password = create_user(client.tracking_uri)

    with User(owner, owner_password, monkeypatch):
        experiment_id = client.create_experiment("scorer-version-child-permission-outcomes")
        response = requests.post(
            client.tracking_uri + "/api/3.0/mlflow/scorers/register",
            json={
                "experiment_id": experiment_id,
                "name": "scorer_child_permission",
                "serialized_scorer": json.dumps({"v": 1}),
            },
            auth=(owner, owner_password),
        )
        response.raise_for_status()

    scorer_pattern = f"{experiment_id}/scorer_child_permission"
    # A per-name scorer EDIT with NO experiment grant: the scorer tier authorizes.
    grant_role_permission(client.tracking_uri, scorer_only_writer, "scorer", scorer_pattern, "EDIT")
    # A per-name scorer READ is authoritative and below EDIT: blocks, even with a
    # positive version-tier grant alongside.
    grant_role_permission(client.tracking_uri, child_writer, "scorer", scorer_pattern, "READ")
    grant_role_permission(client.tracking_uri, child_writer, "scorer_version", "*", "EDIT")
    # Experiment EDIT holders, with DENY overlays for two of them.
    for username in (exp_writer, version_denied_writer, parent_denied_writer):
        grant_role_permission(client.tracking_uri, username, "experiment", experiment_id, "EDIT")
    grant_role_permission(client.tracking_uri, version_denied_writer, "scorer_version", "*", "DENY")
    grant_role_permission(client.tracking_uri, parent_denied_writer, "scorer", "*", "DENY")

    payload = {
        "experiment_id": experiment_id,
        "name": "scorer_child_permission",
        "serialized_scorer": json.dumps({"v": 2}),
    }
    # Denied: no grant anywhere; a below-EDIT authoritative scorer grant; a version DENY;
    # a wildcard scorer DENY (authoritative over the experiment fallback).
    for auth in (
        (no_grant, no_grant_password),
        (child_writer, child_writer_password),
        (version_denied_writer, version_denied_writer_password),
        (parent_denied_writer, parent_denied_writer_password),
    ):
        response = requests.post(
            client.tracking_uri + "/api/3.0/mlflow/scorers/register",
            json=payload,
            auth=auth,
        )
        assert response.status_code == 403

    # Allowed: experiment EDIT via the fallback (no scorer grant), and per-name scorer
    # EDIT even without any experiment grant (the child-escalation the RFC advertises).
    for auth in (
        (exp_writer, exp_writer_password),
        (scorer_only_writer, scorer_only_writer_password),
    ):
        response = requests.post(
            client.tracking_uri + "/api/3.0/mlflow/scorers/register",
            json=payload,
            auth=auth,
        )
        assert response.status_code == 200

    from mlflow.server.auth.client import AuthServiceClient

    # A successful version-add must NOT upsert creator MANAGE on the existing scorer.
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        permission = AuthServiceClient(client.tracking_uri).get_user_permission(
            exp_writer, "scorer", scorer_pattern
        )
    assert permission.permission == "NO_PERMISSIONS"
    assert permission.allowed is False


@pytest.mark.parametrize("prefix", [_MCP_REST_PREFIX])
def test_mcp_server_version_child_permission_outcomes(fastapi_client, monkeypatch, prefix):
    owner, owner_password = create_user(fastapi_client.tracking_uri)
    no_grant, no_grant_password = create_user(fastapi_client.tracking_uri)
    parent_writer, parent_writer_password = create_user(fastapi_client.tracking_uri)
    child_writer, child_writer_password = create_user(fastapi_client.tracking_uri)
    denied_writer, denied_writer_password = create_user(fastapi_client.tracking_uri)
    server_name = "com.test/version-child-permission"

    with User(owner, owner_password, monkeypatch):
        requests.post(
            url=fastapi_client.tracking_uri + prefix,
            json={"name": server_name},
            auth=(owner, owner_password),
        ).raise_for_status()

    grant_role_permission(
        fastapi_client.tracking_uri, parent_writer, "mcp_server", server_name, "EDIT"
    )
    grant_role_permission(
        fastapi_client.tracking_uri, child_writer, "mcp_server", server_name, "READ"
    )
    grant_role_permission(
        fastapi_client.tracking_uri, child_writer, "mcp_server_version", "*", "EDIT"
    )
    grant_role_permission(
        fastapi_client.tracking_uri, denied_writer, "mcp_server", server_name, "EDIT"
    )
    grant_role_permission(
        fastapi_client.tracking_uri, denied_writer, "mcp_server_version", "*", "DENY"
    )

    def create_version(auth, version):
        return requests.post(
            url=fastapi_client.tracking_uri + f"{prefix}/{server_name}/versions",
            json={
                "server_json": {"name": server_name, "version": version},
                "source": "https://example.com/server.py",
            },
            auth=auth,
        )

    assert create_version((no_grant, no_grant_password), "1.0.0").status_code == 403
    assert create_version((denied_writer, denied_writer_password), "1.0.1").status_code == 403
    assert create_version((parent_writer, parent_writer_password), "1.0.2").status_code == 200
    assert create_version((child_writer, child_writer_password), "1.0.3").status_code == 200


@pytest.mark.parametrize(
    "client",
    [{"MLFLOW_AUTH_CONFIG_PATH": "fixtures/no_permission_auth.ini"}],
    indirect=True,
)
def test_review_queue_child_permission_outcomes(client: MlflowClient, monkeypatch):
    owner, owner_password = create_user(client.tracking_uri)
    no_grant, no_grant_password = create_user(client.tracking_uri)
    parent_writer, parent_writer_password = create_user(client.tracking_uri)
    child_writer, child_writer_password = create_user(client.tracking_uri)
    denied_writer, denied_writer_password = create_user(client.tracking_uri)

    with User(owner, owner_password, monkeypatch):
        experiment_id = client.create_experiment("review-queue-child-permission-outcomes")
        trace_id = _create_trace(client.tracking_uri, experiment_id, (owner, owner_password))
        response = requests.post(
            client.tracking_uri + "/api/3.0/mlflow/review-queues/create",
            json={"experiment_id": experiment_id, "name": "owner_queue", "queue_type": "CUSTOM"},
            auth=(owner, owner_password),
        )
        response.raise_for_status()
        owner_queue_id = response.json()["review_queue"]["queue_id"]

    grant_role_permission(client.tracking_uri, parent_writer, "experiment", experiment_id, "EDIT")
    grant_role_permission(client.tracking_uri, child_writer, "experiment", experiment_id, "READ")
    grant_role_permission(client.tracking_uri, child_writer, "review_queue", "*", "EDIT")
    grant_role_permission(client.tracking_uri, denied_writer, "experiment", experiment_id, "EDIT")
    grant_role_permission(client.tracking_uri, denied_writer, "review_queue", "*", "DENY")

    def create_queue(auth, name):
        return requests.post(
            client.tracking_uri + "/api/3.0/mlflow/review-queues/create",
            json={"experiment_id": experiment_id, "name": name, "queue_type": "CUSTOM"},
            auth=auth,
        )

    def add_item(auth, queue_id):
        return requests.post(
            client.tracking_uri + "/api/3.0/mlflow/review-queues/items/add",
            json={"queue_id": queue_id, "item_type": "TRACE", "item_ids": [trace_id]},
            auth=auth,
        )

    def remove_item(auth, queue_id):
        return requests.post(
            client.tracking_uri + "/api/3.0/mlflow/review-queues/items/remove",
            json={"queue_id": queue_id, "item_ids": [trace_id]},
            auth=auth,
        )

    for auth in ((no_grant, no_grant_password), (denied_writer, denied_writer_password)):
        assert create_queue(auth, f"denied_{auth[0]}").status_code == 403
        assert add_item(auth, owner_queue_id).status_code == 403
        assert remove_item(auth, owner_queue_id).status_code == 403

    for auth, name in (
        ((parent_writer, parent_writer_password), "parent_queue"),
        ((child_writer, child_writer_password), "child_queue"),
    ):
        response = create_queue(auth, name)
        assert response.status_code == 200
        queue_id = response.json()["review_queue"]["queue_id"]
        assert add_item(auth, queue_id).status_code == 200
        assert remove_item(auth, queue_id).status_code == 200

    # Items are TRACE references: attaching resolves them and listing returns them, so a
    # (trace, *, DENY) vetoes both even with full queue rights -- while removing (which
    # touches no trace data) stays queue-EDIT-only (review finding).
    trace_denied, trace_denied_password = create_user(client.tracking_uri)
    grant_role_permission(client.tracking_uri, trace_denied, "experiment", experiment_id, "EDIT")
    grant_role_permission(client.tracking_uri, trace_denied, "trace", "*", "DENY")
    t_auth = (trace_denied, trace_denied_password)
    response = create_queue(t_auth, "trace_denied_queue")
    assert response.status_code == 200
    trace_denied_queue_id = response.json()["review_queue"]["queue_id"]
    assert add_item(t_auth, trace_denied_queue_id).status_code == 403
    assert (
        requests.get(
            client.tracking_uri + "/api/3.0/mlflow/review-queues/items/list",
            params={"queue_id": trace_denied_queue_id},
            auth=t_auth,
        ).status_code
        == 403
    )
    assert remove_item(t_auth, trace_denied_queue_id).status_code == 200
    # Without the DENY the same operations succeed (trace tier falls back to experiment).
    assert add_item((owner, owner_password), owner_queue_id).status_code == 200
    assert (
        requests.get(
            client.tracking_uri + "/api/3.0/mlflow/review-queues/items/list",
            params={"queue_id": owner_queue_id},
            auth=(owner, owner_password),
        ).status_code
        == 200
    )


def test_read_predicate_scopes_parent_deny_to_matching_resource(monkeypatch, tmp_path):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")
    monkeypatch.setattr(
        auth_module,
        "auth_config",
        auth_module.auth_config._replace(default_permission=READ.name),
    )
    store = SqlAlchemyStore()
    store.init_db(f"sqlite:///{tmp_path / 'read-pattern-scope.db'}")
    monkeypatch.setattr(auth_module, "store", store, raising=False)

    user = store.create_user("reader", "supersecurepassword", is_admin=False)
    role = store.create_role(name="role", workspace="default")
    store.assign_role_to_user(user.id, role.id)
    store.add_role_permission(role.id, "experiment", "e1", "DENY")

    can_read = auth_module._role_based_read_predicate("reader", "experiment")
    assert not can_read("e1")
    assert can_read("e2")


@pytest.mark.parametrize(("permission", "expected"), [("MANAGE", True), ("DENY", False)])
def test_delete_scorer_version_uses_child_permission(monkeypatch, permission, expected):
    from mlflow.server.auth.permissions import get_permission

    monkeypatch.setattr(
        auth_module,
        "_get_permission_from_scorer_version_name",
        lambda: get_permission(permission),
    )
    monkeypatch.setattr(
        auth_module,
        "_get_permission_from_scorer_name",
        lambda: pytest.fail("whole-scorer permission must not resolve for a version delete"),
    )

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/scorers/delete",
        method="DELETE",
        json={"experiment_id": "1", "name": "scorer", "version": 1},
    ):
        assert auth_module.validate_can_delete_scorer_version() is expected


@pytest.mark.parametrize(("permission", "expected"), [("MANAGE", True), ("DENY", False)])
def test_delete_scorer_without_version_uses_parent_permission(monkeypatch, permission, expected):
    # Whole-scorer delete: the parent scorer tier decides first, and the version child
    # tier (wildcard, parent fallback) must also allow -- but the CONCRETE version helper
    # must not resolve (no version in the body).
    from mlflow.server.auth.permissions import get_permission

    monkeypatch.setattr(
        auth_module,
        "_get_permission_from_scorer_name",
        lambda: get_permission(permission),
    )
    monkeypatch.setattr(
        auth_module,
        "_get_scorer_version_permission",
        lambda _e, _n: get_permission(permission),
    )
    monkeypatch.setattr(
        auth_module,
        "_get_request_param",
        lambda name: {"experiment_id": "1", "name": "scorer"}[name],
    )
    monkeypatch.setattr(
        auth_module,
        "_get_permission_from_scorer_version_name",
        lambda: pytest.fail("concrete version permission must not resolve for whole-scorer delete"),
    )

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/scorers/delete",
        method="DELETE",
        json={"experiment_id": "1", "name": "scorer"},
    ):
        assert auth_module.validate_can_delete_scorer_version() is expected


def test_delete_scorer_version_tolerates_non_object_json(monkeypatch):
    # A non-object JSON body (e.g. a list) must not 500 the validator: it carries no
    # "version", so it resolves the whole-scorer form instead of raising AttributeError on
    # `.get`. Guards against the `(get_json() or {}).get(...)` non-dict crash.
    from mlflow.server.auth.permissions import MANAGE

    monkeypatch.setattr(auth_module, "_get_permission_from_scorer_name", lambda: MANAGE)
    monkeypatch.setattr(auth_module, "_get_scorer_version_permission", lambda _e, _n: MANAGE)
    monkeypatch.setattr(
        auth_module,
        "_get_request_param",
        lambda name: {"experiment_id": "1", "name": "scorer"}[name],
    )
    monkeypatch.setattr(
        auth_module,
        "_get_permission_from_scorer_version_name",
        lambda: pytest.fail("version tier must not resolve for a non-object body (no version)"),
    )
    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/scorers/delete", method="DELETE", json=[1, 2, 3]
    ):
        assert auth_module.validate_can_delete_scorer_version() is True


def test_delete_scorer_version_does_not_cascade_parent_grants(monkeypatch):
    auth_store = mock.Mock()
    auth_store._scorer_pattern.return_value = "1/scorer"
    monkeypatch.setattr(auth_module, "store", auth_store)

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/scorers/delete",
        method="DELETE",
        json={"experiment_id": "1", "name": "scorer", "version": 1},
    ):
        auth_module.delete_scorer_permissions_cascade(mock.Mock())
    auth_store.delete_grants_for_resource.assert_not_called()

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/scorers/delete",
        method="DELETE",
        json={"experiment_id": "1", "name": "scorer"},
    ):
        auth_module.delete_scorer_permissions_cascade(mock.Mock())
    auth_store.delete_grants_for_resource.assert_called_once_with("scorer", "1/scorer")


def test_graphql_run_permission_uses_run_child_tier(monkeypatch):
    run = SimpleNamespace(info=SimpleNamespace(experiment_id="7"))
    monkeypatch.setattr(
        auth_module, "_get_tracking_store", lambda: SimpleNamespace(get_run=lambda _rid: run)
    )
    captured = {}

    def fake_child_permission(child_type, child_key, experiment_id, username=None):
        captured.update(
            child_type=child_type,
            child_key=child_key,
            experiment_id=experiment_id,
            username=username,
        )
        return SimpleNamespace(can_read=False)

    monkeypatch.setattr(auth_module, "_experiment_child_permission", fake_child_permission)
    perm = auth_module._graphql_get_permission_for_run("run-1", "bob")
    assert perm.can_read is False
    assert captured == {
        "child_type": "run",
        "child_key": "run-1",
        "experiment_id": "7",
        "username": "bob",
    }


def test_graphql_search_runs_prefilter_keeps_run_only_grant(monkeypatch):
    # A run-tier READ grant with no experiment READ must not drop the experiment.
    def fake_child_permission(child_type, child_key, experiment_id, username=None):
        assert child_type == "run"
        return SimpleNamespace(can_read=experiment_id == "keep")

    monkeypatch.setattr(auth_module, "_experiment_child_permission", fake_child_permission)
    mw = auth_module.GraphQLAuthorizationMiddleware()
    input_obj = SimpleNamespace(experiment_ids=["keep", "drop"])
    allowed = mw._check_authorization("mlflowSearchRuns", {"input": input_obj}, "bob")
    assert allowed is True
    assert input_obj.experiment_ids == ["keep"]


def test_otlp_ingestion_validator_uses_trace_child_tier(monkeypatch):
    captured = {}

    def fake_child_permission(child_type, child_key, experiment_id, username=None):
        captured.update(child_type=child_type, experiment_id=experiment_id, username=username)
        return SimpleNamespace(can_update=False)

    monkeypatch.setattr(auth_module, "_experiment_child_permission", fake_child_permission)
    validator = auth_module._get_otel_validator("/otlp/v1/traces")
    request = SimpleNamespace(headers={"x-mlflow-experiment-id": "9"})
    allowed = asyncio.run(validator("carol", request))
    assert allowed is False
    assert captured == {"child_type": "trace", "experiment_id": "9", "username": "carol"}


def test_get_or_create_user_queue_uses_review_queue_child_tier(monkeypatch):
    monkeypatch.setattr(auth_module, "_get_request_param", lambda _p: "9")
    monkeypatch.setattr(
        auth_module,
        "_get_review_queue_permission_for_experiment",
        lambda eid: SimpleNamespace(can_update=eid == "9"),
    )
    # Experiment-tier resolver must not be consulted for this child operation.
    monkeypatch.setattr(
        auth_module,
        "_get_permission_from_experiment_id",
        lambda: (_ for _ in ()).throw(AssertionError("should use review_queue tier")),
    )
    assert auth_module.validate_can_get_or_create_user_queue() is True


def test_review_queue_item_uses_review_queue_child_tier(monkeypatch):
    queue = SimpleNamespace(experiment_id="9")
    monkeypatch.setattr(
        auth_module, "authenticate_request", lambda: SimpleNamespace(username="dan")
    )
    monkeypatch.setattr(auth_module, "_get_request_param", lambda _p: "queue-1")
    monkeypatch.setattr(
        auth_module,
        "_get_tracking_store",
        lambda: SimpleNamespace(get_review_queue=lambda _q: queue),
    )
    monkeypatch.setattr(
        auth_module, "_get_review_queue_permission", lambda q: SimpleNamespace(can_update=True)
    )
    monkeypatch.setattr(auth_module, "_review_queue_has_member", lambda q, u: True)
    monkeypatch.setattr(
        auth_module,
        "_get_experiment_permission",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should use review_queue tier")),
    )
    assert auth_module.validate_can_review_queue_item() is True


def test_canonicalize_artifact_proxy_path_collapses_dot_and_empty_segments():
    # The handler's validate_path_is_safe accepts '.' and empty (and percent-encoded '.')
    # segments that filesystem joining then collapses, so authorization must classify the
    # collapsed form (review finding).
    canon = auth_module._canonicalize_artifact_proxy_path
    assert canon("7/run-1/./artifacts/secret.bin") == "7/run-1/artifacts/secret.bin"
    assert canon("7//models/m-1/artifacts/model.pkl") == "7/models/m-1/artifacts/model.pkl"
    assert canon("./7/run-1/artifacts/x") == "7/run-1/artifacts/x"
    assert canon("7/%2E/run-1/artifacts/x") == "7/run-1/artifacts/x"
    assert canon("7/run-1/artifacts/x") == "7/run-1/artifacts/x"
    assert canon(None) is None
    # A path the handler's safety check rejects is passed through unchanged: the handler
    # rejects the request outright, so no data is served under any classification.
    assert auth_module._canonicalize_artifact_proxy_path("7/../8/artifacts/x") == (
        "7/../8/artifacts/x"
    )


def test_artifact_proxy_path_from_view_args_canonicalizes():
    # The Flask chokepoint (path param / ?path= query param) must hand authorization the
    # canonical path, for both the layout classifier and the experiment-id extraction.
    from flask import request as flask_request

    with auth_module.app.test_request_context("/x?path=7/run-1/./artifacts/secret.bin"):
        flask_request.view_args = {}
        assert auth_module._artifact_proxy_path_from_view_args() == ("7/run-1/artifacts/secret.bin")
        assert auth_module._get_experiment_id_from_view_args() == "7"
    with auth_module.app.test_request_context("/x"):
        flask_request.view_args = {"artifact_path": "./7/models/m-1/artifacts/MLmodel"}
        assert auth_module._artifact_proxy_path_from_view_args() == (
            "7/models/m-1/artifacts/MLmodel"
        )
        assert auth_module._get_experiment_id_from_view_args() == "7"


def test_artifact_proxy_child_from_path_dispatches_by_layout():
    assert auth_module._artifact_proxy_child_from_path("42/run-abc/artifacts/model.pkl") == (
        "run",
        "run-abc",
    )
    assert auth_module._artifact_proxy_child_from_path(
        "workspaces/team-a/7/run-xyz/artifacts/f"
    ) == ("run", "run-xyz")
    # ``traces``/``models`` folders dispatch to their own child tiers, not the run tier.
    assert auth_module._artifact_proxy_child_from_path("42/traces/tr-1/artifacts/data") == (
        "trace",
        "tr-1",
    )
    assert auth_module._artifact_proxy_child_from_path("42/models/m-1/artifacts/MLmodel") == (
        "logged_model",
        "m-1",
    )
    # Experiment-level path has no child.
    assert auth_module._artifact_proxy_child_from_path("42/artifacts/plot.png") is None
    # Experiment-ROOT objects (no fixed ``artifacts`` directory after the second segment)
    # are experiment-level too: a run-only grant must not read them, and a (run, *, DENY)
    # must not block them (Copilot).
    assert auth_module._artifact_proxy_child_from_path("42/test.txt") is None
    assert auth_module._artifact_proxy_child_from_path("42/dir/file.txt") is None
    # A bare two-segment path can't be distinguished from an experiment-root file, so it
    # resolves on the experiment tier; run CONTENT (under <run>/artifacts/) stays run-gated.
    assert auth_module._artifact_proxy_child_from_path("42/run-abc") is None
    # A child folder root (no concrete id) resolves on the wildcard child key so a
    # child DENY is still honored when listing the folder.
    assert auth_module._artifact_proxy_child_from_path("42/traces") == ("trace", "*")
    assert auth_module._artifact_proxy_child_from_path("42/models") == ("logged_model", "*")


@pytest.mark.parametrize(
    ("artifact_path", "expected_type", "expected_key"),
    [
        ("42/run-abc/artifacts/model.pkl", "run", "run-abc"),
        ("42/traces/tr-1/artifacts/data", "trace", "tr-1"),
        ("42/models/m-1/artifacts/MLmodel", "logged_model", "m-1"),
        # Non-canonical forms ('.' / empty / encoded '.' segments) must classify to the
        # SAME child the filesystem join resolves to (review finding: they previously
        # fell to the experiment tier and bypassed a child DENY).
        ("42/run-abc/./artifacts/model.pkl", "run", "run-abc"),
        ("42/./traces/tr-1/artifacts/data", "trace", "tr-1"),
        ("42//models/m-1/artifacts/MLmodel", "logged_model", "m-1"),
        ("42/%2E/run-abc/artifacts/model.pkl", "run", "run-abc"),
    ],
)
def test_proxy_artifact_permission_uses_child_tier_for_layout(
    monkeypatch, artifact_path, expected_type, expected_key
):
    captured = {}

    def fake_child_permission(child_type, child_key, experiment_id, username=None):
        captured.update(
            child_type=child_type,
            child_key=child_key,
            experiment_id=experiment_id,
            username=username,
        )
        return SimpleNamespace(can_read=True)

    monkeypatch.setattr(auth_module, "_experiment_child_permission", fake_child_permission)
    perm = auth_module._get_proxy_artifact_permission(
        f"/api/2.0/mlflow-artifacts/artifacts/{artifact_path}", "erin"
    )
    assert perm.can_read is True
    assert captured == {
        "child_type": expected_type,
        "child_key": expected_key,
        "experiment_id": "42",
        "username": "erin",
    }


def test_proxy_artifact_permission_experiment_level_uses_experiment_tier(monkeypatch):
    # An experiment-level path (no run/trace/model segment) must not hit the child tier.
    monkeypatch.setattr(
        auth_module,
        "_experiment_child_permission",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should use experiment tier")),
    )
    monkeypatch.setattr(
        auth_module, "_role_permission_for", lambda **_: SimpleNamespace(can_read=True)
    )
    monkeypatch.setattr(auth_module, "_get_role_permission_or_default", lambda perm: perm)
    perm = auth_module._get_proxy_artifact_permission(
        "/api/2.0/mlflow-artifacts/artifacts/42/artifacts/plot.png", "erin"
    )
    assert perm.can_read is True


def _fake_resp(response_message):
    from mlflow.utils.proto_json_utils import message_to_json

    return SimpleNamespace(json=json.loads(message_to_json(response_message)), data=None)


def test_redact_get_trace_info_v3_assessments_hides_denied(monkeypatch):
    from mlflow.protos import service_pb2 as pb

    resp_msg = pb.GetTraceInfoV3.Response()
    ti = resp_msg.trace.trace_info
    ti.trace_id = "tr-1"
    ti.trace_location.mlflow_experiment.experiment_id = "9"
    ti.assessments.add().assessment_id = "a-1"

    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    monkeypatch.setattr(auth_module, "authenticate_request", lambda: SimpleNamespace(username="u"))
    monkeypatch.setattr(
        auth_module, "_role_based_read_predicate", lambda *a, **k: lambda _eid: False
    )
    resp = _fake_resp(resp_msg)
    auth_module.redact_get_trace_info_v3_assessments(resp)

    out = pb.GetTraceInfoV3.Response()
    auth_module.parse_dict(json.loads(resp.data), out)
    assert list(out.trace.trace_info.assessments) == []


def test_redact_trace_assessments_kept_when_readable(monkeypatch):
    from mlflow.protos import service_pb2 as pb

    resp_msg = pb.SearchTracesV3.Response()
    ti = resp_msg.traces.add()
    ti.trace_id = "tr-1"
    ti.trace_location.mlflow_experiment.experiment_id = "9"
    ti.assessments.add().assessment_id = "a-1"

    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    monkeypatch.setattr(auth_module, "authenticate_request", lambda: SimpleNamespace(username="u"))
    monkeypatch.setattr(
        auth_module, "_role_based_read_predicate", lambda *a, **k: lambda _eid: True
    )
    resp = _fake_resp(resp_msg)
    auth_module.redact_search_traces_v3_assessments(resp)

    out = pb.SearchTracesV3.Response()
    auth_module.parse_dict(json.loads(resp.data), out)
    assert [a.assessment_id for a in out.traces[0].assessments] == ["a-1"]


def test_redact_get_registered_model_versions_hides_denied(monkeypatch):
    from mlflow.protos import model_registry_pb2 as pb

    resp_msg = pb.GetRegisteredModel.Response()
    rm = resp_msg.registered_model
    rm.name = "m1"
    rm.latest_versions.add().name = "m1"

    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    monkeypatch.setattr(auth_module, "authenticate_request", lambda: SimpleNamespace(username="u"))
    monkeypatch.setattr(
        auth_module, "_rm_or_prompt_version_read_predicate", lambda _u: lambda _mv: False
    )
    resp = _fake_resp(resp_msg)
    auth_module.redact_get_registered_model_versions(resp)

    out = pb.GetRegisteredModel.Response()
    auth_module.parse_dict(json.loads(resp.data), out)
    assert list(out.registered_model.latest_versions) == []


def test_list_scorers_gated_on_scorer_version_tier(monkeypatch):
    from mlflow.protos import service_pb2 as pb

    resp_msg = pb.ListScorers.Response()
    s1 = resp_msg.scorers.add()
    s1.experiment_id = 9
    s1.scorer_name = "keep"
    s2 = resp_msg.scorers.add()
    s2.experiment_id = 9
    s2.scorer_name = "denied"

    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    monkeypatch.setattr(auth_module, "authenticate_request", lambda: SimpleNamespace(username="u"))
    monkeypatch.setattr(auth_module.store, "_scorer_pattern", lambda e, n: f"{e}/{n}")

    def fake_predicate(_username, resource_type, parent_type=None):
        if resource_type == "experiment":
            return lambda _e: True
        # scorer_version tier: deny "denied"
        return lambda pattern: not pattern.endswith("/denied")

    monkeypatch.setattr(auth_module, "_role_based_read_predicate", fake_predicate)
    resp = _fake_resp(resp_msg)
    auth_module.filter_list_scorers(resp)

    out = pb.ListScorers.Response()
    auth_module.parse_dict(json.loads(resp.data), out)
    assert [s.scorer_name for s in out.scorers] == ["keep"]


def test_assessment_lookup_keys_requires_trace_id():
    assert auth_module._assessment_lookup_keys("tr-1/a-1") == ("tr-1", "a-1")
    with pytest.raises(MlflowException, match="Expected '<trace_id>/<assessment_id>'"):
        auth_module._assessment_lookup_keys("a-1")


def test_resource_dispatch_prompt_resolves_registry_workspace(monkeypatch):
    # "prompt" is grantable and advertised by the permission APIs, so the unified
    # per-user permission API must dispatch it -- namespaced under "prompt" with the
    # registry's get_registered_model workspace lookup, exactly like the runtime path
    # (review finding: it previously hit the unsupported-type 400, breaking
    # prompt-MANAGE delegation for non-admins).
    monkeypatch.setattr(
        auth_module,
        "_get_model_registry_store",
        lambda: SimpleNamespace(get_registered_model=lambda name: SimpleNamespace(name=name)),
    )
    dispatch = auth_module._resource_dispatch_keys("prompt", "my-prompt")
    assert dispatch is not None
    assert dispatch.resource_key == "my-prompt"
    assert dispatch.workspace_lookup_id == "my-prompt"
    assert dispatch.workspace_label == "prompt"


def test_every_valid_resource_type_has_permission_api_dispatch():
    # Every advertised grantable type (except the workspace pseudo-type, which the API
    # rejects explicitly) must be resolvable by the unified permission API: via the
    # child dispatch, the top-level workspace-fetcher map, or the compound
    # experiment_id/name scorer branch (review finding: prompt was advertised but had
    # no dispatch).
    from mlflow.server.auth.permissions import VALID_RESOURCE_TYPES

    covered = (
        set(auth_module._CHILD_RESOURCE_TYPES)
        | set(auth_module._RESOURCE_WORKSPACE_FETCHER)
        | {"scorer"}  # compound "<experiment_id>/<name>" dispatch branch
    )
    assert set(VALID_RESOURCE_TYPES) - {"workspace"} <= covered


def test_resource_dispatch_assessment_resolves_experiment_via_trace(monkeypatch):
    # The convenience API must resolve the assessment's experiment through its trace,
    # not by passing the assessment id to get_trace_info -- and must verify the CONCRETE
    # assessment exists (a nonexistent id resolves NO_PERMISSIONS via the caller's
    # not-found catch rather than reporting a permission for a child that isn't there).
    trace = SimpleNamespace(experiment_id="9")
    calls = {}

    def fake_get_trace_info(trace_id):
        calls["trace_id"] = trace_id
        return trace

    def fake_get_assessment(trace_id, assessment_id):
        calls["assessment"] = (trace_id, assessment_id)
        return SimpleNamespace()

    monkeypatch.setattr(
        auth_module,
        "_get_tracking_store",
        lambda: SimpleNamespace(
            get_trace_info=fake_get_trace_info,
            get_assessment=fake_get_assessment,
            get_experiment=lambda _e: None,
        ),
    )
    dispatch = auth_module._resource_dispatch_keys("assessment", "tr-1/a-1")
    assert calls["trace_id"] == "tr-1"
    assert calls["assessment"] == ("tr-1", "a-1")
    assert dispatch.resource_key == "a-1"
    assert dispatch.workspace_lookup_id == "9"
    assert dispatch.parent_type == "experiment"
    assert dispatch.parent_id == "9"


def test_graphql_search_datasets_uses_experiment_tier(monkeypatch):
    # Datasets are not a run sub-resource: the prefilter must use experiment READ, not
    # the run tier, so a run-only reader is excluded and a run DENY doesn't hide them.
    monkeypatch.setattr(
        auth_module,
        "_graphql_can_read_runs_in_experiment",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("datasets must not use run tier")),
    )
    monkeypatch.setattr(
        auth_module, "_graphql_can_read_experiment", lambda exp_id, _u: exp_id == "keep"
    )
    mw = auth_module.GraphQLAuthorizationMiddleware()
    input_obj = SimpleNamespace(experiment_ids=["keep", "drop"])
    assert mw._check_authorization("mlflowSearchDatasets", {"input": input_obj}, "bob") is True
    assert input_obj.experiment_ids == ["keep"]


def test_filter_get_mcp_server_redacts_version_on_deny(monkeypatch):
    server = {
        "name": "srv",
        "latest_version": {"version": 3},
        "aliases": {"prod": 3},
        "access_endpoints": [
            {"server_name": "srv", "resolved_version": {"version": 3}, "tools": ["t"]}
        ],
    }
    monkeypatch.setattr(
        auth_module, "_get_mcp_server_permission", lambda *a: SimpleNamespace(can_read=True)
    )
    monkeypatch.setattr(auth_module, "_permission_to_allowed_actions", lambda _p: [])
    monkeypatch.setattr(
        auth_module,
        "_get_mcp_server_version_permission",
        lambda *a: SimpleNamespace(can_read=False),
    )
    out = json.loads(auth_module._filter_get_mcp_server("u", json.dumps(server).encode(), object()))
    assert out["latest_version"] is None
    assert out["aliases"] is None
    ep = out["access_endpoints"][0]
    assert ep["resolved_version"] is None
    assert ep["tools"] is None


def test_filter_get_mcp_server_keeps_version_when_readable(monkeypatch):
    server = {"name": "srv", "latest_version": {"version": 3}, "aliases": {"prod": 3}}
    monkeypatch.setattr(
        auth_module, "_get_mcp_server_permission", lambda *a: SimpleNamespace(can_read=True)
    )
    monkeypatch.setattr(auth_module, "_permission_to_allowed_actions", lambda _p: [])
    monkeypatch.setattr(
        auth_module, "_get_mcp_server_version_permission", lambda *a: SimpleNamespace(can_read=True)
    )
    out = json.loads(auth_module._filter_get_mcp_server("u", json.dumps(server).encode(), object()))
    assert out["latest_version"] == {"version": 3}
    assert out["aliases"] == {"prod": 3}


def test_filter_search_mcp_servers_redacts_version_on_deny(monkeypatch):
    body = json.dumps({
        "mcp_servers": [
            {
                "name": "srv",
                "latest_version": {"version": 3},
                "aliases": {"prod": 3},
                "access_endpoints": [{"server_name": "srv", "resolved_version": {"version": 3}}],
            }
        ]
    }).encode()
    monkeypatch.setattr(
        auth_module,
        "_role_based_permission_resolver",
        lambda *a, **k: lambda _n: SimpleNamespace(can_read=True),
    )
    monkeypatch.setattr(auth_module, "_permission_to_allowed_actions", lambda _p: [])

    def fake_predicate(_username, resource_type, parent_type=None):
        # server row-read allowed; version tier denied -> version fields redacted.
        return lambda _n: resource_type != "mcp_server_version"

    monkeypatch.setattr(auth_module, "_role_based_read_predicate", fake_predicate)
    request = SimpleNamespace(
        query_params=SimpleNamespace(get=lambda k, d=None: d, getlist=lambda _k: [])
    )
    out = json.loads(auth_module._filter_search_mcp_servers("u", body, request))
    server = out["mcp_servers"][0]
    assert server["latest_version"] is None
    assert server["aliases"] is None
    assert server["access_endpoints"][0]["resolved_version"] is None
    # Summary endpoints have no ``tools`` field; redaction must not inject one.
    assert "tools" not in server["access_endpoints"][0]


@pytest.mark.parametrize("default_permission", ["NO_PERMISSIONS", "EDIT"])
def test_role_based_permission_resolver_matches_store_fold(
    monkeypatch, tmp_path, default_permission
):
    # The bulk full-Permission resolver must resolve each id to the SAME Permission as the
    # authoritative point path (_get_role_permission_or_default over the store fold), so
    # allowed-action stamping via the bulk path is identical to the single-resource path --
    # including the positive default_permission floor: under default EDIT an explicit USE
    # grant must still stamp EDIT actions (review finding).
    from mlflow.server.auth.permissions import DENY, EDIT, USE

    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")
    monkeypatch.setattr(
        auth_module,
        "auth_config",
        auth_module.auth_config._replace(default_permission=default_permission),
    )
    store = SqlAlchemyStore()
    store.init_db(f"sqlite:///{tmp_path / 'perm-resolver.db'}")
    monkeypatch.setattr(auth_module, "store", store, raising=False)

    ws = "default"  # DEFAULT_WORKSPACE_NAME when workspaces are disabled
    user = store.create_user("resolver-user", "supersecurepassword", is_admin=False)
    role = store.create_role(name="rr", workspace=ws)
    store.add_role_permission(role.id, "mcp_server", "srv-edit", EDIT.name)
    store.add_role_permission(role.id, "mcp_server", "srv-deny", DENY.name)
    store.add_role_permission(role.id, "mcp_server", "srv-use", USE.name)
    store.assign_role_to_user(user.id, role.id)

    resolver = auth_module._role_based_permission_resolver("resolver-user", "mcp_server")

    def point_perm(name):
        # The exact point-path composition: store fold, then the default floor.
        return auth_module._get_role_permission_or_default(
            lambda: store.get_role_permission_for_resource(user.id, "mcp_server", name, ws)
        )

    # positive (EDIT/USE), DENY, and no-grant (default) all match the floored point path.
    for name in ("srv-edit", "srv-deny", "srv-use", "srv-none"):
        assert resolver(name).name == point_perm(name).name, (default_permission, name)


def test_filter_search_mcp_servers_is_query_bounded(monkeypatch):
    # The MCP server search filter must build the bulk permission resolver (and the version
    # reader) ONCE per response, not once per server, so a multi-server page stays O(1)
    # authorization queries rather than a workspace + grants round trip per distinct name.
    from mlflow.server import auth
    from mlflow.server.auth.permissions import MANAGE

    builds = {"perm": 0, "version": 0}

    def fake_resolver(username, resource_type, parent_type=None):
        builds["perm"] += 1
        return lambda _name: MANAGE  # readable + all actions

    def fake_version_reader(username):
        builds["version"] += 1
        return lambda _name: True

    monkeypatch.setattr(auth, "_role_based_permission_resolver", fake_resolver)
    monkeypatch.setattr(auth, "_mcp_server_version_reader", fake_version_reader)
    monkeypatch.setattr(auth, "_permission_to_allowed_actions", lambda _p: [])

    body = json.dumps({"mcp_servers": [{"name": f"srv-{i}"} for i in range(10)]}).encode()
    request = SimpleNamespace(
        query_params=SimpleNamespace(get=lambda k, d=None: d, getlist=lambda _k: [])
    )
    auth._filter_search_mcp_servers("u", body, request)

    assert builds["perm"] == 1  # built once for the whole page, not once per server
    assert builds["version"] == 1


def test_filter_search_mcp_endpoints_redacts_version_on_deny(monkeypatch):
    body = json.dumps({
        "mcp_access_endpoints": [
            {"server_name": "srv", "resolved_version": {"version": 3}, "server_version": 3}
        ]
    }).encode()
    monkeypatch.setattr(auth_module, "_permission_to_allowed_actions", lambda _p: [])
    monkeypatch.setattr(
        auth_module, "get_routed_asgi_path", lambda _r: "/api/3.0/mlflow/mcp-servers/endpoints"
    )

    def fake_predicate(_username, resource_type, parent_type=None):
        # endpoint row-read (mcp_server) allowed; version tier denied -> fields redacted.
        return lambda _n: resource_type != "mcp_server_version"

    monkeypatch.setattr(auth_module, "_role_based_read_predicate", fake_predicate)
    request = SimpleNamespace(
        query_params=SimpleNamespace(get=lambda k, d=None: d, getlist=lambda _k: [])
    )
    out = json.loads(auth_module._filter_search_mcp_endpoints("u", body, request))
    ep = out["mcp_access_endpoints"][0]
    assert ep["resolved_version"] is None
    assert ep["server_version"] is None


@pytest.mark.parametrize(
    ("routed_path", "expected_server_name"),
    [
        ("/api/3.0/mlflow/mcp-servers/endpoints", None),
        ("/api/3.0/mlflow/mcp-servers/com.test/srv/endpoints", "com.test/srv"),
    ],
)
def test_filter_search_mcp_endpoints_backfill_scoped_to_routed_server(
    monkeypatch, routed_path, expected_server_name
):
    # The redaction filter serves both the global /endpoints search and the per-server
    # /{name}/endpoints route. On the per-server route the readable-row backfill must stay
    # scoped to that server (a bare search would pull other servers' endpoints and advance
    # a token in a different result set).
    from mlflow.store.entities.paged_list import PagedList

    token = base64.b64encode(json.dumps({"offset": 1}).encode()).decode()
    body = json.dumps({"mcp_access_endpoints": [], "next_page_token": token}).encode()
    monkeypatch.setattr(auth_module, "_permission_to_allowed_actions", lambda _p: [])
    monkeypatch.setattr(auth_module, "get_routed_asgi_path", lambda _r: routed_path)
    monkeypatch.setattr(
        auth_module, "_role_based_read_predicate", lambda *_a, **_k: lambda _n: True
    )
    search = mock.Mock(return_value=PagedList([], None))
    monkeypatch.setattr(
        auth_module,
        "_get_tracking_store",
        lambda: SimpleNamespace(search_mcp_access_endpoints=search),
    )
    request = SimpleNamespace(
        query_params=SimpleNamespace(get=lambda k, d=None: d, getlist=lambda _k: [])
    )
    auth_module._filter_search_mcp_endpoints("u", body, request)
    assert search.call_count == 1
    assert search.call_args.kwargs["server_name"] == expected_server_name


def test_redact_registered_model_clears_aliases_on_version_deny(monkeypatch):
    from mlflow.protos import model_registry_pb2 as pb

    resp_msg = pb.GetRegisteredModel.Response()
    rm = resp_msg.registered_model
    rm.name = "m1"
    rm.latest_versions.add().name = "m1"
    alias = rm.aliases.add()
    alias.alias = "prod"
    alias.version = "3"

    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    monkeypatch.setattr(auth_module, "authenticate_request", lambda: SimpleNamespace(username="u"))
    monkeypatch.setattr(
        auth_module, "_rm_or_prompt_version_read_predicate", lambda _u: lambda _e: False
    )
    resp = _fake_resp(resp_msg)
    auth_module.redact_get_registered_model_versions(resp)

    out = pb.GetRegisteredModel.Response()
    auth_module.parse_dict(json.loads(resp.data), out)
    assert list(out.registered_model.latest_versions) == []
    assert list(out.registered_model.aliases) == []


def test_mcp_alias_routes_classified_as_version_paths():
    # /{namespace}/{slug}/aliases/{alias} returns a full version response -> version tier.
    assert auth_module._is_mcp_server_version_path(["ns", "slug", "aliases", "prod"]) is True
    assert auth_module._is_mcp_server_version_path(["ns", "slug", "versions", "3"]) is True
    assert auth_module._is_mcp_server_version_path(["ns", "slug", "tags"]) is False
    # alias routes are not version-CREATE paths (that stays versions-only)
    assert auth_module._is_mcp_server_version_create_path(["ns", "slug", "aliases"]) is False


def test_registered_model_alias_routes_gated_on_version_tier():
    # Alias set/delete mutate version mappings -> version-tier validators, matching the
    # version-tier read on GetModelVersionByAlias (not the parent model/prompt tier).
    from mlflow.protos.model_registry_pb2 import (
        DeleteRegisteredModelAlias,
        GetModelVersionByAlias,
        SetRegisteredModelAlias,
    )

    handlers = auth_module.BEFORE_REQUEST_HANDLERS
    assert (
        handlers[SetRegisteredModelAlias]
        is auth_module._validate_can_update_model_version_or_prompt_version
    )
    assert (
        handlers[DeleteRegisteredModelAlias]
        is auth_module._validate_can_delete_model_version_or_prompt_version
    )
    assert (
        handlers[GetModelVersionByAlias]
        is auth_module._validate_can_read_model_version_or_prompt_version
    )


def test_filter_list_scorers_requires_experiment_and_version_read(monkeypatch):
    # OSS parity: every visible row requires experiment READ AND read on the scorer_version
    # tier (scorer-parent fallback). A version-readable row in an unreadable experiment is
    # dropped, and a version-denied row in a readable experiment is dropped.
    from mlflow.protos import service_pb2 as pb

    resp_msg = pb.ListScorers.Response()
    keep = resp_msg.scorers.add()
    keep.experiment_id = 9
    keep.scorer_name = "keep"
    version_denied = resp_msg.scorers.add()
    version_denied.experiment_id = 9
    version_denied.scorer_name = "denied"
    exp_denied = resp_msg.scorers.add()
    exp_denied.experiment_id = 13  # experiment 13 is not readable
    exp_denied.scorer_name = "keep"

    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    monkeypatch.setattr(auth_module, "authenticate_request", lambda: SimpleNamespace(username="u"))
    monkeypatch.setattr(auth_module.store, "_scorer_pattern", lambda e, n: f"{e}/{n}")

    def fake_predicate(_username, resource_type, parent_type=None):
        if resource_type == "experiment":
            assert parent_type is None
            return lambda exp_id: exp_id != "13"
        assert resource_type == "scorer_version"
        assert parent_type == "scorer"
        return lambda pattern: not pattern.endswith("/denied")

    monkeypatch.setattr(auth_module, "_role_based_read_predicate", fake_predicate)
    resp = _fake_resp(resp_msg)
    auth_module.filter_list_scorers(resp)

    out = pb.ListScorers.Response()
    auth_module.parse_dict(json.loads(resp.data), out)
    assert [(s.experiment_id, s.scorer_name) for s in out.scorers] == [(9, "keep")]


def test_update_registered_model_redacts_versions_only_on_deny(monkeypatch):
    from mlflow.protos import model_registry_pb2 as pb

    def build_resp():
        m = pb.UpdateRegisteredModel.Response()
        m.registered_model.name = "m1"
        m.registered_model.latest_versions.add().name = "m1"
        a = m.registered_model.aliases.add()
        a.alias = "prod"
        a.version = "3"
        return _fake_resp(m)

    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    monkeypatch.setattr(auth_module, "authenticate_request", lambda: SimpleNamespace(username="u"))

    # No version DENY (parent-readable) -> response unchanged (no deviation).
    monkeypatch.setattr(
        auth_module, "_rm_or_prompt_version_read_predicate", lambda _u: lambda _e: True
    )
    resp = build_resp()
    auth_module.redact_update_registered_model_versions(resp)
    out = pb.UpdateRegisteredModel.Response()
    auth_module.parse_dict(json.loads(resp.data), out)
    assert [v.name for v in out.registered_model.latest_versions] == ["m1"]
    assert len(out.registered_model.aliases) == 1

    # version DENY -> version data redacted, but the operation still succeeded (handler ran).
    monkeypatch.setattr(
        auth_module, "_rm_or_prompt_version_read_predicate", lambda _u: lambda _e: False
    )
    resp = build_resp()
    auth_module.redact_update_registered_model_versions(resp)
    out = pb.UpdateRegisteredModel.Response()
    auth_module.parse_dict(json.loads(resp.data), out)
    assert list(out.registered_model.latest_versions) == []
    assert list(out.registered_model.aliases) == []


def test_update_registered_model_redaction_bypassed_for_admin(monkeypatch):
    from mlflow.protos import model_registry_pb2 as pb

    m = pb.UpdateRegisteredModel.Response()
    m.registered_model.name = "m1"
    m.registered_model.latest_versions.add().name = "m1"
    resp = _fake_resp(m)
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: True)
    auth_module.redact_update_registered_model_versions(resp)
    # admin path returns early; response untouched (data stays None -> not rewritten).
    assert resp.data is None


def test_filter_single_mcp_endpoint_redacts_on_version_deny(monkeypatch):
    body = json.dumps({
        "server_name": "srv",
        "resolved_version": {"version": 3},
        "server_version": 3,
        "tools": ["t"],
    }).encode()
    req = SimpleNamespace()

    # version DENY -> version fields nulled.
    monkeypatch.setattr(
        auth_module,
        "_get_mcp_server_version_permission",
        lambda *a: SimpleNamespace(can_read=False),
    )
    out = json.loads(auth_module._filter_single_mcp_endpoint("u", body, req))
    assert out["resolved_version"] is None
    assert out["server_version"] is None
    assert out["tools"] is None

    # version readable -> unchanged.
    monkeypatch.setattr(
        auth_module, "_get_mcp_server_version_permission", lambda *a: SimpleNamespace(can_read=True)
    )
    out = json.loads(auth_module._filter_single_mcp_endpoint("u", body, req))
    assert out["resolved_version"] == {"version": 3}
    assert out["tools"] == ["t"]


def test_mcp_patch_and_endpoint_routes_registered_for_redaction():
    filters = auth_module.FASTAPI_ENDPOINT_RESPONSE_FILTERS
    assert filters[auth_module._update_mcp_server_endpoint] is auth_module._filter_get_mcp_server
    assert (
        filters[auth_module._search_server_access_endpoints_endpoint]
        is auth_module._filter_search_mcp_endpoints
    )
    for ep in (
        auth_module._get_mcp_access_endpoint_endpoint,
        auth_module._create_mcp_access_endpoint_endpoint,
        auth_module._update_mcp_access_endpoint_endpoint,
    ):
        assert filters[ep] is auth_module._filter_single_mcp_endpoint
