import configparser
from typing import NamedTuple


class AuthConfig(NamedTuple):
    default_permission: str
    database_uri: str
    admin_username: str
    admin_password: str


def read_auth_config(config_path: str) -> AuthConfig:
    config = configparser.ConfigParser()
    config.read(config_path)
    return AuthConfig(
        default_permission=config["mlflow"]["default_permission"],
        database_uri=config["mlflow"]["database_uri"],
        admin_username=config["mlflow"]["admin_username"],
        admin_password=config["mlflow"]["admin_password"],
    )
