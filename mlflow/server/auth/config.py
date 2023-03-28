import configparser
import pathlib
from typing import NamedTuple


class AppConfig(NamedTuple):
    default_permission: str
    database_uri: str


def read_app_config() -> AppConfig:
    # TODO: need to pass in config path from user?
    creds_path = pathlib.Path("basic_auth.ini").resolve()
    config = configparser.ConfigParser()
    config.read(str(creds_path))
    return AppConfig(
        default_permission=config["mlflow"]["default_permission"],
        database_uri=config["mlflow"]["database_uri"],
    )


app_config = read_app_config()
