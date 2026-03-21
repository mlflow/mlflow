import configparser
from pathlib import Path
from typing import NamedTuple

from mlflow.environment_variables import MLFLOW_AUTH_CONFIG_PATH

DEFAULT_AUTHORIZATION_FUNCTION = "mlflow.server.auth:authenticate_request_basic_auth"


class AuthConfig(NamedTuple):
    default_permission: str
    database_uri: str
    admin_username: str
    admin_password: str
    authorization_function: str
    grant_default_workspace_access: bool
    workspace_cache_max_size: int
    workspace_cache_ttl_seconds: int


def _get_auth_config_path() -> str:
    return (
        MLFLOW_AUTH_CONFIG_PATH.get() or Path(__file__).parent.joinpath("basic_auth.ini").resolve()
    )


def read_auth_config() -> AuthConfig:
    config_path = _get_auth_config_path()
    config = configparser.ConfigParser()
    config.read(config_path)
    return AuthConfig(
        default_permission=config["mlflow"]["default_permission"],
        database_uri=config["mlflow"]["database_uri"],
        admin_username=config["mlflow"]["admin_username"],
        admin_password=config["mlflow"]["admin_password"],
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
    )
