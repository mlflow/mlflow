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
from tests.server.auth.auth_test_utils import create_user, User
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

    # authenticated
    username, password = create_user(client.tracking_uri)
    with User(username, password, monkeypatch):
        client.search_experiments()


def test_search_experiments(client, monkeypatch):
    """
    Use user1 to create 10 experiments,
    grant READ permission to user2 on experiments [0, 3, 4, 5, 6, 8].
    Test whether user2 can search only and all the readable experiments,
    both paged and un-paged.
    """
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)

    readable = [0, 3, 4, 5, 6, 8]

    with User(username1, password1, monkeypatch):
        for i in range(10):
            experiment_id = client.create_experiment(f"exp{i}")
            _send_rest_tracking_post_request(
                client.tracking_uri,
                "/api/2.0/mlflow/experiments/permissions/create",
                json_payload={
                    "experiment_id": experiment_id,
                    "username": username2,
                    "permission": "READ" if i in readable else "NO_PERMISSIONS",
                },
                auth=(username1, password1),
            )

    # test un-paged search
    with User(username1, password1, monkeypatch):
        experiments = client.search_experiments(
            max_results=100,
            filter_string="name LIKE 'exp%'",
            order_by=["name ASC"],
        )
        names = sorted([exp.name for exp in experiments])
        assert names == [f"exp{i}" for i in range(10)]

    with User(username2, password2, monkeypatch):
        experiments = client.search_experiments(
            max_results=100,
            filter_string="name LIKE 'exp%'",
            order_by=["name ASC"],
        )
        names = sorted([exp.name for exp in experiments])
        assert names == [f"exp{i}" for i in readable]

    # test paged search
    with User(username1, password1, monkeypatch):
        page_token = ""
        experiments = []
        while True:
            res = client.search_experiments(
                max_results=4,
                filter_string="name LIKE 'exp%'",
                order_by=["name ASC"],
                page_token=page_token,
            )
            experiments.extend(res)
            page_token = res.token
            if not page_token:
                break

        names = sorted([exp.name for exp in experiments])
        assert names == [f"exp{i}" for i in range(10)]

    with User(username2, password2, monkeypatch):
        page_token = ""
        experiments = []
        while True:
            res = client.search_experiments(
                max_results=4,
                filter_string="name LIKE 'exp%'",
                order_by=["name ASC"],
                page_token=page_token,
            )
            experiments.extend(res)
            page_token = res.token
            if not page_token:
                break

        names = sorted([exp.name for exp in experiments])
        assert names == [f"exp{i}" for i in readable]


def test_search_registered_models(client, monkeypatch):
    """
    Use user1 to create 10 registered_models,
    grant READ permission to user2 on registered_models [0, 3, 4, 5, 6, 8].
    Test whether user2 can search only and all the readable registered_models,
    both paged and un-paged.
    """
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)

    readable = [0, 3, 4, 5, 6, 8]

    with User(username1, password1, monkeypatch):
        for i in range(10):
            rm = client.create_registered_model(f"rm{i}")
            _send_rest_tracking_post_request(
                client.tracking_uri,
                "/api/2.0/mlflow/registered-models/permissions/create",
                json_payload={
                    "name": rm.name,
                    "username": username2,
                    "permission": "READ" if i in readable else "NO_PERMISSIONS",
                },
                auth=(username1, password1),
            )

    # test un-paged search
    with User(username1, password1, monkeypatch):
        registered_models = client.search_registered_models(
            max_results=100,
            filter_string="name LIKE 'rm%'",
            order_by=["name ASC"],
        )
        names = sorted([rm.name for rm in registered_models])
        assert names == [f"rm{i}" for i in range(10)]

    with User(username2, password2, monkeypatch):
        registered_models = client.search_registered_models(
            max_results=100,
            filter_string="name LIKE 'rm%'",
            order_by=["name ASC"],
        )
        names = sorted([rm.name for rm in registered_models])
        assert names == [f"rm{i}" for i in readable]

    # test paged search
    with User(username1, password1, monkeypatch):
        page_token = ""
        registered_models = []
        while True:
            res = client.search_registered_models(
                max_results=4,
                filter_string="name LIKE 'rm%'",
                order_by=["name ASC"],
                page_token=page_token,
            )
            registered_models.extend(res)
            page_token = res.token
            if not page_token:
                break

        names = sorted([rm.name for rm in registered_models])
        assert names == [f"rm{i}" for i in range(10)]

    with User(username2, password2, monkeypatch):
        page_token = ""
        registered_models = []
        while True:
            res = client.search_registered_models(
                max_results=4,
                filter_string="name LIKE 'rm%'",
                order_by=["name ASC"],
                page_token=page_token,
            )
            registered_models.extend(res)
            page_token = res.token
            if not page_token:
                break

        names = sorted([rm.name for rm in registered_models])
        assert names == [f"rm{i}" for i in readable]
