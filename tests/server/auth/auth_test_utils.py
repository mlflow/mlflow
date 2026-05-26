from pathlib import Path

from mlflow.environment_variables import MLFLOW_TRACKING_PASSWORD, MLFLOW_TRACKING_USERNAME
from mlflow.server.auth import auth_config
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

from tests.helper_functions import random_str
from tests.tracking.integration_test_utils import _send_rest_tracking_post_request

PERMISSION = "READ"
NEW_PERMISSION = "EDIT"
ADMIN_USERNAME = auth_config.admin_username
ADMIN_PASSWORD = auth_config.admin_password


def write_isolated_auth_config(tmp_path: Path) -> Path:
    """Write a basic_auth.ini under ``tmp_path`` whose ``database_uri`` points
    at an SQLite file inside the same dir, and return the config path.

    Tests that spawn the auth server via ``_init_server`` must point
    ``MLFLOW_AUTH_CONFIG_PATH`` at the returned file (through ``extra_env``) so
    the spawned auth server writes to the temp DB instead of the repo-root
    ``basic_auth.db`` shared with the dev server. Without this, integration
    tests pollute (and depending on the fixture, delete) the developer's local
    auth state.
    """
    config_path = tmp_path / "basic_auth.ini"
    db_path = tmp_path / "basic_auth.db"
    config_path.write_text(
        "[mlflow]\n"
        "default_permission = READ\n"
        f"database_uri = sqlite:///{db_path}\n"
        f"admin_username = {ADMIN_USERNAME}\n"
        f"admin_password = {ADMIN_PASSWORD}\n"
        "authorization_function = mlflow.server.auth:authenticate_request_basic_auth\n"
        "grant_default_workspace_access = false\n"
    )
    return config_path


def create_user(tracking_uri: str, username: str | None = None, password: str | None = None):
    username = random_str() if username is None else username
    password = random_str() if password is None else password
    response = _send_rest_tracking_post_request(
        tracking_uri,
        "/api/2.0/mlflow/users/create",
        {
            "username": username,
            "password": password,
        },
        auth=(ADMIN_USERNAME, ADMIN_PASSWORD),
    )
    response.raise_for_status()
    return username, password


def grant_role_permission(
    tracking_uri: str,
    username: str,
    resource_type: str,
    resource_pattern: str,
    permission: str,
    workspace: str = DEFAULT_WORKSPACE_NAME,
    auth: tuple[str, str] | None = None,
) -> None:
    """
    Grant ``username`` ``permission`` on ``(resource_type, resource_pattern)`` via the
    role API. Creates a throwaway role in ``workspace`` with a single permission row,
    then assigns it to the user. Useful for auth-flow tests that previously called
    ``POST /mlflow/experiments/permissions/create`` (and equivalents) to set up state.
    """
    auth = auth or (ADMIN_USERNAME, ADMIN_PASSWORD)
    role_name = f"_test_{resource_type}_{random_str()}"
    create_resp = _send_rest_tracking_post_request(
        tracking_uri,
        "/api/3.0/mlflow/roles/create",
        {"name": role_name, "workspace": workspace},
        auth=auth,
    )
    create_resp.raise_for_status()
    role_id = create_resp.json()["role"]["id"]
    add_resp = _send_rest_tracking_post_request(
        tracking_uri,
        "/api/3.0/mlflow/roles/permissions/add",
        {
            "role_id": role_id,
            "resource_type": resource_type,
            "resource_pattern": resource_pattern,
            "permission": permission,
        },
        auth=auth,
    )
    add_resp.raise_for_status()
    assign_resp = _send_rest_tracking_post_request(
        tracking_uri,
        "/api/3.0/mlflow/roles/assign",
        {"username": username, "role_id": role_id},
        auth=auth,
    )
    assign_resp.raise_for_status()


class User:
    def __init__(self, username, password, monkeypatch):
        self.username = username
        self.password = password
        self.monkeypatch = monkeypatch

    def __enter__(self):
        self.monkeypatch.setenv(MLFLOW_TRACKING_USERNAME.name, self.username)
        self.monkeypatch.setenv(MLFLOW_TRACKING_PASSWORD.name, self.password)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monkeypatch.delenv(MLFLOW_TRACKING_USERNAME.name, raising=False)
        self.monkeypatch.delenv(MLFLOW_TRACKING_PASSWORD.name, raising=False)
