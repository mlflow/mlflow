from mlflow.tracking._tracking_service.utils import (
    _TRACKING_USERNAME_ENV_VAR,
    _TRACKING_PASSWORD_ENV_VAR,
)
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
                _TRACKING_USERNAME_ENV_VAR: self.username,
                _TRACKING_PASSWORD_ENV_VAR: self.password,
            }
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.monkeypatch.delenvs(
            [_TRACKING_USERNAME_ENV_VAR, _TRACKING_PASSWORD_ENV_VAR], raising=False
        )
