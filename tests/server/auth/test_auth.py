"""
Integration test which starts a local Tracking Server on an ephemeral port,
and ensures authentication is working.
"""
import pytest

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    ErrorCode,
    UNAUTHENTICATED,
)
from mlflow.tracking._tracking_service.utils import (
    _TRACKING_USERNAME_ENV_VAR,
    _TRACKING_PASSWORD_ENV_VAR,
)
from mlflow.utils.os import is_windows
from tests.helper_functions import random_str
from tests.tracking.integration_test_utils import (
    _terminate_server,
    _init_server,
    _send_rest_tracking_post_request,
)


@pytest.fixture
def client(tmp_path):
    path = tmp_path.joinpath("sqlalchemy.db").as_uri()
    backend_uri = ("sqlite://" if is_windows() else "sqlite:////") + path[len("file://") :]

    url, process = _init_server(
        backend_uri=backend_uri,
        root_artifact_uri=tmp_path.joinpath("artifacts").as_uri(),
        app_module="mlflow.server.auth",
    )
    yield MlflowClient(url)
    _terminate_server(process)


def test_authenticate(client, monkeypatch):
    # unauthenticated
    monkeypatch.delenvs([_TRACKING_USERNAME_ENV_VAR, _TRACKING_PASSWORD_ENV_VAR], raising=False)
    with pytest.raises(MlflowException, match=r"You are not authenticated.") as exception_context:
        client.search_experiments()
    assert exception_context.value.error_code == ErrorCode.Name(UNAUTHENTICATED)

    # sign up
    username = random_str()
    password = random_str()
    _send_rest_tracking_post_request(
        client.tracking_uri,
        "/api/2.0/mlflow/users",
        {
            "username": username,
            "password": password,
        },
    )

    # authenticated
    monkeypatch.setenvs(
        {
            _TRACKING_USERNAME_ENV_VAR: username,
            _TRACKING_PASSWORD_ENV_VAR: password,
        }
    )
    client.search_experiments()
