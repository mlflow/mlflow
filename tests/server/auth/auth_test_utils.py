from mlflow.environment_variables import MLFLOW_TRACKING_PASSWORD, MLFLOW_TRACKING_USERNAME
from mlflow.server.auth import auth_config

from tests.helper_functions import random_str
from tests.tracking.integration_test_utils import _send_rest_tracking_post_request

PERMISSION = "READ"
NEW_PERMISSION = "EDIT"
ADMIN_USERNAME = auth_config.admin_username
ADMIN_PASSWORD = auth_config.admin_password


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
