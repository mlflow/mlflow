import configparser
import os
from typing import NamedTuple, Optional, Tuple

from mlflow.environment_variables import MLFLOW_TRACKING_PASSWORD, MLFLOW_TRACKING_USERNAME


class MlflowCreds(NamedTuple):
    username: Optional[str]
    password: Optional[str]


def _get_credentials_path() -> str:
    return os.path.expanduser("~/.mlflow/credentials")


def _read_mlflow_creds_from_file() -> Tuple[Optional[str], Optional[str]]:
    path = _get_credentials_path()
    if not os.path.exists(path):
        return None, None

    config = configparser.ConfigParser()
    config.read(path)
    if "mlflow" not in config:
        return None, None

    mlflow_cfg = config["mlflow"]
    username_key = MLFLOW_TRACKING_USERNAME.name.lower()
    password_key = MLFLOW_TRACKING_PASSWORD.name.lower()
    return mlflow_cfg.get(username_key), mlflow_cfg.get(password_key)


def _read_mlflow_creds_from_env() -> Tuple[Optional[str], Optional[str]]:
    return MLFLOW_TRACKING_USERNAME.get(), MLFLOW_TRACKING_PASSWORD.get()


def read_mlflow_creds() -> MlflowCreds:
    username_file, password_file = _read_mlflow_creds_from_file()
    username_env, password_env = _read_mlflow_creds_from_env()
    return MlflowCreds(
        username=username_env or username_file,
        password=password_env or password_file,
    )
