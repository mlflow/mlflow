import configparser
from pathlib import Path
from typing import NamedTuple

from mlflow.environment_variables import MLFLOW_AUTH_CONFIG_PATH


class KeycloakAuthConfig(NamedTuple):
    host: str
    realm_name: str


def _get_auth_config_path() -> str:
    return (
        MLFLOW_AUTH_CONFIG_PATH.get() or Path(__file__).parent.joinpath("keycloak_auth.ini").resolve()
    )


def read_auth_config() -> KeycloakAuthConfig:
    config_path = _get_auth_config_path()
    config = configparser.ConfigParser()
    config.read(config_path)
    return KeycloakAuthConfig(
        host=config["mlflow"]["host"],
        realm_name=config["mlflow"]["realm_name"]
    )
