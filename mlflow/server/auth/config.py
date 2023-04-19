import configparser
from typing import NamedTuple


class AuthConfig(NamedTuple):
    default_permission: str
    database_uri: str
    root_username: str
    root_password: str


def read_auth_config(config_path: str) -> AuthConfig:
    config = configparser.ConfigParser()
    config.read(config_path)
    return AuthConfig(
        default_permission=config["mlflow"]["default_permission"],
        database_uri=config["mlflow"]["database_uri"],
        root_username=config["mlflow"]["root_username"],
        root_password=config["mlflow"]["root_password"],
    )
