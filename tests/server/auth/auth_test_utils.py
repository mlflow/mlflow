from mlflow.environment_variables import MLFLOW_TRACKING_PASSWORD, MLFLOW_TRACKING_USERNAME

from tests.helper_functions import random_str
from tests.tracking.integration_test_utils import _send_rest_tracking_post_request


def create_user(tracking_uri):
    username = random_str()
    password = random_str()
    _send_rest_tracking_post_request(
        tracking_uri,
        "/api/2.0/mlflow/users/create",
        {
            "username": username,
            "password": password,
        },
    )
    return username, password


class User:
    def __init__(self, username, password, monkeypatch):
        self.username = username
        self.password = password
        self.monkeypatch = monkeypatch

    def __enter__(self):
        self.monkeypatch.setenvs(
            {
                MLFLOW_TRACKING_USERNAME.name: self.username,
                MLFLOW_TRACKING_PASSWORD.name: self.password,
            }
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monkeypatch.delenvs(
            [MLFLOW_TRACKING_PASSWORD.name, MLFLOW_TRACKING_PASSWORD.name], raising=False
        )
