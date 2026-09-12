import configparser
from pathlib import Path
from typing import NamedTuple

from mlflow.environment_variables import (
    MLFLOW_AUTH_ADMIN_PASSWORD,
    MLFLOW_AUTH_ADMIN_USERNAME,
    MLFLOW_AUTH_CONFIG_PATH,
)

DEFAULT_AUTHORIZATION_FUNCTION = "mlflow.server.auth:authenticate_request_basic_auth"


class AuthConfig(NamedTuple):
    default_permission: str
    database_uri: str
    # Both are None when neither the config file nor the MLFLOW_AUTH_ADMIN_* environment
    # variable provides a non-empty value. MLflow ships no default admin password; the auth app
    # refuses to bootstrap the admin user without explicit values.
    admin_username: str | None
    admin_password: str | None
    authorization_function: str
    grant_default_workspace_access: bool
    workspace_cache_max_size: int
    workspace_cache_ttl_seconds: int
    auth_cache_max_size: int
    auth_cache_ttl_seconds: int
    session_ttl_seconds: int
    read_database_uri: str | None = None


def _get_auth_config_path() -> str:
    return (
        MLFLOW_AUTH_CONFIG_PATH.get() or Path(__file__).parent.joinpath("basic_auth.ini").resolve()
    )


def read_auth_config() -> AuthConfig:
    config_path = _get_auth_config_path()
    config = configparser.ConfigParser()
    config.read(config_path)
    # An environment variable that is set takes precedence even when it is empty, so an empty
    # injected secret surfaces as a bootstrap error instead of silently falling back to a
    # possibly stale value in the file. Empty values normalize to None.
    admin_username = (
        MLFLOW_AUTH_ADMIN_USERNAME.get()
        if MLFLOW_AUTH_ADMIN_USERNAME.is_set()
        else config["mlflow"].get("admin_username")
    ) or None
    admin_password = (
        MLFLOW_AUTH_ADMIN_PASSWORD.get()
        if MLFLOW_AUTH_ADMIN_PASSWORD.is_set()
        else config["mlflow"].get("admin_password")
    ) or None
    return AuthConfig(
        default_permission=config["mlflow"]["default_permission"],
        database_uri=config["mlflow"]["database_uri"],
        admin_username=admin_username,
        admin_password=admin_password,
        authorization_function=config["mlflow"].get(
            "authorization_function", DEFAULT_AUTHORIZATION_FUNCTION
        ),
        grant_default_workspace_access=config.getboolean(
            "mlflow", "grant_default_workspace_access", fallback=False
        ),
        workspace_cache_max_size=config.getint(
            "mlflow", "workspace_cache_max_size", fallback=10000
        ),
        workspace_cache_ttl_seconds=config.getint(
            "mlflow", "workspace_cache_ttl_seconds", fallback=3600
        ),
        auth_cache_max_size=config.getint("mlflow", "auth_cache_max_size", fallback=10000),
        # Off by default — enabling the cache introduces a per-worker staleness window
        # (see basic_auth.ini for details). Operators must explicitly opt in.
        auth_cache_ttl_seconds=config.getint("mlflow", "auth_cache_ttl_seconds", fallback=0),
        # Only consulted by the opt-in session authenticator
        # (mlflow.server.auth.session:authenticate_request_session); irrelevant
        # under the default authorization_function. 12 hours.
        session_ttl_seconds=config.getint("mlflow", "session_ttl_seconds", fallback=43200),
        read_database_uri=config["mlflow"].get("read_database_uri", None),
    )
