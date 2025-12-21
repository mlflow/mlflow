"""
Integration test which starts a local Tracking Server on an ephemeral port,
and ensures authentication is working.
"""

import re
import subprocess
import sys
import time
from pathlib import Path

import jwt
import psutil
import pytest
import requests

import mlflow
from mlflow import MlflowClient
from mlflow.entities.logged_model_status import LoggedModelStatus
from mlflow.environment_variables import (
    MLFLOW_FLASK_SERVER_SECRET_KEY,
    MLFLOW_TRACKING_PASSWORD,
    MLFLOW_TRACKING_USERNAME,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    RESOURCE_DOES_NOT_EXIST,
    UNAUTHENTICATED,
    ErrorCode,
)
from mlflow.server.auth.routes import GET_REGISTERED_MODEL_PERMISSION, GET_SCORER_PERMISSION
from mlflow.utils.os import is_windows

from tests.helper_functions import random_str
from tests.server.auth.auth_test_utils import ADMIN_PASSWORD, ADMIN_USERNAME, User, create_user
from tests.tracking.integration_test_utils import (
    _init_server,
    _send_rest_tracking_post_request,
    get_safe_port,
)


@pytest.fixture
def client(request, tmp_path):
    path = tmp_path.joinpath("sqlalchemy.db").as_uri()
    backend_uri = ("sqlite://" if is_windows() else "sqlite:////") + path[len("file://") :]
    extra_env = getattr(request, "param", {})
    extra_env[MLFLOW_FLASK_SERVER_SECRET_KEY.name] = "my-secret-key"

    with _init_server(
        backend_uri=backend_uri,
        root_artifact_uri=tmp_path.joinpath("artifacts").as_uri(),
        extra_env=extra_env,
        app="mlflow.server.auth:create_app",
        server_type="flask",
    ) as url:
        yield MlflowClient(url)


def test_authenticate(client, monkeypatch):
    # unauthenticated
    monkeypatch.delenv(MLFLOW_TRACKING_USERNAME.name, raising=False)
    monkeypatch.delenv(MLFLOW_TRACKING_PASSWORD.name, raising=False)
    with pytest.raises(MlflowException, match=r"You are not authenticated.") as exception_context:
        client.search_experiments()
    assert exception_context.value.error_code == ErrorCode.Name(UNAUTHENTICATED)

    # authenticated
    username, password = create_user(client.tracking_uri)
    with User(username, password, monkeypatch):
        client.search_experiments()


@pytest.mark.parametrize(
    ("username", "password"),
    [
        ("", "password"),
        ("username", ""),
        ("", ""),
    ],
)
def test_validate_username_and_password(client, username, password):
    with pytest.raises(requests.exceptions.HTTPError, match=r"BAD REQUEST"):
        create_user(client.tracking_uri, username=username, password=password)


def _mlflow_search_experiments_rest(base_uri, headers):
    response = requests.post(
        f"{base_uri}/api/2.0/mlflow/experiments/search",
        headers=headers,
        json={
            "max_results": 100,
        },
    )
    response.raise_for_status()
    return response


def _mlflow_create_user_rest(base_uri, headers):
    username = random_str()
    password = random_str()
    response = requests.post(
        f"{base_uri}/api/2.0/mlflow/users/create",
        headers=headers,
        json={
            "username": username,
            "password": password,
        },
    )
    response.raise_for_status()
    return username, password


@pytest.mark.parametrize(
    "client",
    [
        {
            "MLFLOW_AUTH_CONFIG_PATH": "tests/server/auth/fixtures/jwt_auth.ini",
            "PYTHONPATH": str(Path.cwd() / "examples" / "jwt_auth"),
        }
    ],
    indirect=True,
)
def test_authenticate_jwt(client):
    # unauthenticated
    with pytest.raises(requests.HTTPError, match=r"401 Client Error: UNAUTHORIZED") as e:
        _mlflow_search_experiments_rest(client.tracking_uri, {})
    assert e.value.response.status_code == 401  # Unauthorized

    # authenticated
    # we need to use jwt to authenticate as admin so that we can create a new user
    bearer_token = jwt.encode({"username": ADMIN_USERNAME}, "secret", algorithm="HS256")
    headers = {"Authorization": f"Bearer {bearer_token}"}
    username, password = _mlflow_create_user_rest(client.tracking_uri, headers)

    # authenticate with the newly created user
    headers = {
        "Authorization": f"Bearer {jwt.encode({'username': username}, 'secret', algorithm='HS256')}"
    }
    _mlflow_search_experiments_rest(client.tracking_uri, headers)

    # invalid token
    bearer_token = jwt.encode({"username": username}, "invalid", algorithm="HS256")
    headers = {"Authorization": f"Bearer {bearer_token}"}
    with pytest.raises(requests.HTTPError, match=r"401 Client Error: UNAUTHORIZED") as e:
        _mlflow_search_experiments_rest(client.tracking_uri, headers)
    assert e.value.response.status_code == 401  # Unauthorized


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


def test_create_and_delete_registered_model(client, monkeypatch):
    username1, password1 = create_user(client.tracking_uri)

    # create a registered model
    with User(username1, password1, monkeypatch):
        rm = client.create_registered_model("test_model")

    # confirm the permission is set correctly
    with User(username1, password1, monkeypatch):
        response = requests.get(
            url=client.tracking_uri + GET_REGISTERED_MODEL_PERMISSION,
            params={"name": rm.name, "username": username1},
            auth=(username1, password1),
        )

    permission = response.json()["registered_model_permission"]
    assert permission["name"] == rm.name
    assert permission["permission"] == "MANAGE"

    # trying to create a model with the same name should fail
    with User(username1, password1, monkeypatch):
        with pytest.raises(MlflowException, match=r"RESOURCE_ALREADY_EXISTS"):
            client.create_registered_model("test_model")

    # delete the registered model
    with User(username1, password1, monkeypatch):
        client.delete_registered_model(rm.name)

    # confirm the registered model permission is also deleted
    with User(username1, password1, monkeypatch):
        response = requests.get(
            url=client.tracking_uri + GET_REGISTERED_MODEL_PERMISSION,
            params={"name": rm.name, "username": username1},
            # Check with admin because the user permission is deleted
            auth=("admin", "password1234"),
        )

    assert response.status_code == 404
    assert response.json()["error_code"] == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)
    assert (
        response.json()["message"]
        == f"Registered model permission with name={rm.name} and username={username1} not found"
    )

    # now we should be able to create a model with the same name
    with User(username1, password1, monkeypatch):
        rm = client.create_registered_model("test_model")
    assert rm.name == "test_model"


def _wait(url: str):
    t = time.time()
    while time.time() - t < 5:
        try:
            if requests.get(f"{url}/health").ok:
                yield
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)

    pytest.fail("Server did not start")


def _kill_all(pid: str):
    parent = psutil.Process(pid)
    for child in parent.children(recursive=True):
        child.kill()
    parent.kill()


def test_proxy_log_artifacts(monkeypatch, tmp_path):
    backend_uri = f"sqlite:///{tmp_path / 'sqlalchemy.db'}"
    port = get_safe_port()
    host = "localhost"
    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlflow",
            "server",
            "--app-name",
            "basic-auth",
            "--backend-store-uri",
            backend_uri,
            "--host",
            host,
            "--port",
            str(port),
            "--workers",
            "1",
            "--gunicorn-opts",
            "--log-level debug",
        ],
        env={MLFLOW_FLASK_SERVER_SECRET_KEY.name: "my-secret-key"},
    ) as prc:
        try:
            url = f"http://{host}:{port}"
            for _ in _wait(url):
                break

            mlflow.set_tracking_uri(url)
            client = MlflowClient(url)
            tmp_file = tmp_path / "test.txt"
            tmp_file.touch()
            username1, password1 = create_user(url)
            with User(username1, password1, monkeypatch):
                exp_id = client.create_experiment("exp")
                run = client.create_run(exp_id)
                client.log_artifact(run.info.run_id, tmp_file)
                client.list_artifacts(run.info.run_id)

            username2, password2 = create_user(url)
            with User(username2, password2, monkeypatch):
                client.list_artifacts(run.info.run_id)
                with pytest.raises(requests.HTTPError, match="Permission denied"):
                    client.log_artifact(run.info.run_id, tmp_file)

                # Ensure that the regular expression captures an experiment ID correctly
                tmp_file_with_numbers = tmp_path / "123456.txt"
                tmp_file_with_numbers.touch()
                with pytest.raises(requests.HTTPError, match="Permission denied"):
                    client.log_artifact(run.info.run_id, tmp_file)
        finally:
            # Kill the server process to prevent `prc.wait()` (called when exiting the context
            # manager) from waiting forever.
            _kill_all(prc.pid)


def test_create_user_from_ui_fails_without_csrf_token(client):
    response = requests.post(
        client.tracking_uri + "/api/2.0/mlflow/users/create-ui",
        json={"username": "test", "password": "test"},
        auth=(ADMIN_USERNAME, ADMIN_PASSWORD),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    assert "The CSRF token is missing" in response.text


def test_create_user_ui(client):
    # needs to be a session as the CSRF protection will set some
    # cookies that need to be present for server side validation
    with requests.Session() as session:
        page = session.get(client.tracking_uri + "/signup", auth=(ADMIN_USERNAME, ADMIN_PASSWORD))

        csrf_regex = re.compile(r"name=\"csrf_token\" value=\"([\S]+)\"")
        match = csrf_regex.search(page.text)

        # assert that the CSRF token is sent in the form
        assert match is not None

        csrf_token = match.group(1)

        response = session.post(
            client.tracking_uri + "/api/2.0/mlflow/users/create-ui",
            data={"username": random_str(), "password": random_str(), "csrf_token": csrf_token},
            auth=(ADMIN_USERNAME, ADMIN_PASSWORD),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        assert "Successfully signed up user" in response.text


def test_logged_model(client: MlflowClient, monkeypatch: pytest.MonkeyPatch):
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)

    class Model(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input):
            return model_input

    with User(username1, password1, monkeypatch):
        exp_id = client.create_experiment("exp")
        model = client.create_logged_model(experiment_id=exp_id)
        client.finalize_logged_model(model_id=model.model_id, status=LoggedModelStatus.READY)
        client.set_logged_model_tags(model_id=model.model_id, tags={"key": "value"})
        client.delete_logged_model_tag(model_id=model.model_id, key="key")
        models = client.search_logged_models(experiment_ids=[exp_id])
        assert len(models) == 1

    with User(username2, password2, monkeypatch):
        loaded_model = client.get_logged_model(model.model_id)
        assert loaded_model.model_id == model.model_id

        models = client.search_logged_models(experiment_ids=[exp_id])
        assert len(models) == 1

        with pytest.raises(MlflowException, match="Permission denied"):
            client.finalize_logged_model(model_id=model.model_id, status=LoggedModelStatus.READY)
        with pytest.raises(MlflowException, match="Permission denied"):
            client.set_logged_model_tags(model_id=model.model_id, tags={"key": "value"})
        with pytest.raises(MlflowException, match="Permission denied"):
            client.delete_logged_model_tag(model_id=model.model_id, key="key")
        with pytest.raises(MlflowException, match="Permission denied"):
            client.delete_logged_model(model_id=model.model_id)


def test_search_logged_models(client: MlflowClient, monkeypatch: pytest.MonkeyPatch):
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)
    readable = [0, 3, 4, 5, 6, 8]
    with User(username1, password1, monkeypatch):
        experiment_ids: list[str] = []
        for i in range(10):
            experiment_id = client.create_experiment(f"exp-{i}")
            experiment_ids.append(experiment_id)
            _send_rest_tracking_post_request(
                client.tracking_uri,
                "/api/2.0/mlflow/experiments/permissions/create",
                json_payload={
                    "experiment_id": experiment_id,
                    "username": username2,
                    "permission": "READ" if (i in readable) else "NO_PERMISSIONS",
                },
                auth=(username1, password1),
            )
            client.create_logged_model(experiment_id=experiment_id)

        models = client.search_logged_models(experiment_ids=experiment_ids)
        assert len(models) == 10

        # Pagination
        models = client.search_logged_models(experiment_ids=experiment_ids, max_results=2)
        assert len(models) == 2
        assert models.token is not None

        models = client.search_logged_models(
            experiment_ids=experiment_ids, max_results=2, page_token=models.token
        )
        assert len(models) == 2
        assert models.token is not None

        models = client.search_logged_models(experiment_ids=experiment_ids, page_token=models.token)
        assert len(models) == 6
        assert models.token is None

    with User(username2, password2, monkeypatch):
        models = client.search_logged_models(experiment_ids=experiment_ids)
        assert len(models) == len(readable)

        # Pagination
        models = client.search_logged_models(experiment_ids=experiment_ids, max_results=2)
        assert len(models) == 2
        assert models.token is not None

        models = client.search_logged_models(
            experiment_ids=experiment_ids, max_results=2, page_token=models.token
        )
        assert len(models) == 2
        assert models.token is not None

        models = client.search_logged_models(experiment_ids=experiment_ids, page_token=models.token)
        assert len(models) == 2
        assert models.token is None


def test_search_runs(client: MlflowClient, monkeypatch: pytest.MonkeyPatch):
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)

    readable = [0, 2]

    with User(username1, password1, monkeypatch):
        experiment_ids: list[str] = []
        run_counts = [8, 10, 7]
        all_runs = {}

        for i in range(3):
            experiment_id = client.create_experiment(f"exp-{i}")
            experiment_ids.append(experiment_id)
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

            all_runs[experiment_id] = []
            for _ in range(run_counts[i]):
                run = client.create_run(experiment_id)
                all_runs[experiment_id].append(run.info.run_id)

    expected_readable_runs = set(all_runs[experiment_ids[0]] + all_runs[experiment_ids[2]])

    with User(username1, password1, monkeypatch):
        runs = client.search_runs(experiment_ids=experiment_ids)
        assert len(runs) == sum(run_counts)

    with User(username2, password2, monkeypatch):
        runs = client.search_runs(experiment_ids=experiment_ids)
        returned_run_ids = {run.info.run_id for run in runs}
        assert returned_run_ids == expected_readable_runs
        assert len(runs) == len(expected_readable_runs)

        page_token = None
        all_paginated_runs = []
        while True:
            runs = client.search_runs(
                experiment_ids=experiment_ids,
                max_results=3,
                page_token=page_token,
            )
            all_paginated_runs.extend([run.info.run_id for run in runs])
            page_token = runs.token
            if not page_token:
                break

        assert len(all_paginated_runs) == len(set(all_paginated_runs))
        assert set(all_paginated_runs) == expected_readable_runs

        inaccessible_runs = set(all_runs[experiment_ids[1]])
        returned_inaccessible = set(all_paginated_runs) & inaccessible_runs
        assert len(returned_inaccessible) == 0


def test_register_and_delete_scorer(client, monkeypatch):
    username1, password1 = create_user(client.tracking_uri)

    with User(username1, password1, monkeypatch):
        experiment_id = client.create_experiment("test_experiment")

    scorer_json = '{"name": "test_scorer", "type": "pyfunc"}'

    with User(username1, password1, monkeypatch):
        response = _send_rest_tracking_post_request(
            client.tracking_uri,
            "/api/3.0/mlflow/scorers/register",
            json_payload={
                "experiment_id": experiment_id,
                "name": "test_scorer",
                "serialized_scorer": scorer_json,
            },
            auth=(username1, password1),
        )

    scorer_name = response.json()["name"]
    assert scorer_name == "test_scorer"

    with User(username1, password1, monkeypatch):
        response = requests.get(
            url=client.tracking_uri + GET_SCORER_PERMISSION,
            params={
                "experiment_id": experiment_id,
                "scorer_name": scorer_name,
                "username": username1,
            },
            auth=(username1, password1),
        )

    permission = response.json()["scorer_permission"]
    assert permission["experiment_id"] == experiment_id
    assert permission["scorer_name"] == scorer_name
    assert permission["permission"] == "MANAGE"

    with User(username1, password1, monkeypatch):
        requests.delete(
            url=client.tracking_uri + "/api/3.0/mlflow/scorers/delete",
            json={
                "experiment_id": experiment_id,
                "name": scorer_name,
            },
            auth=(username1, password1),
        )

    with User(username1, password1, monkeypatch):
        response = requests.get(
            url=client.tracking_uri + GET_SCORER_PERMISSION,
            params={
                "experiment_id": experiment_id,
                "scorer_name": scorer_name,
                "username": username1,
            },
            auth=("admin", "password1234"),
        )

    assert response.status_code == 404
    assert response.json()["error_code"] == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_scorer_permission_denial(client, monkeypatch):
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)

    with User(username1, password1, monkeypatch):
        experiment_id = client.create_experiment("test_experiment")

    scorer_json = '{"name": "test_scorer", "type": "pyfunc"}'

    with User(username1, password1, monkeypatch):
        response = _send_rest_tracking_post_request(
            client.tracking_uri,
            "/api/3.0/mlflow/scorers/register",
            json_payload={
                "experiment_id": experiment_id,
                "name": "test_scorer",
                "serialized_scorer": scorer_json,
            },
            auth=(username1, password1),
        )

    scorer_name = response.json()["name"]

    with User(username2, password2, monkeypatch):
        # user2 has default READ permission, so they CAN read the scorer
        response = requests.get(
            url=client.tracking_uri + "/api/3.0/mlflow/scorers/get",
            params={
                "experiment_id": experiment_id,
                "name": scorer_name,
            },
            auth=(username2, password2),
        )
        response.raise_for_status()
        assert response.json()["scorer"]["scorer_name"] == scorer_name

        # But they CANNOT delete it (READ permission doesn't allow delete)
        response = requests.delete(
            url=client.tracking_uri + "/api/3.0/mlflow/scorers/delete",
            json={
                "experiment_id": experiment_id,
                "name": scorer_name,
            },
            auth=(username2, password2),
        )
        with pytest.raises(requests.HTTPError, match="403"):
            response.raise_for_status()


def test_scorer_read_permission(client, monkeypatch):
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)

    with User(username1, password1, monkeypatch):
        experiment_id = client.create_experiment("test_experiment")

    scorer_json = '{"name": "test_scorer", "type": "pyfunc"}'

    with User(username1, password1, monkeypatch):
        response = _send_rest_tracking_post_request(
            client.tracking_uri,
            "/api/3.0/mlflow/scorers/register",
            json_payload={
                "experiment_id": experiment_id,
                "name": "test_scorer",
                "serialized_scorer": scorer_json,
            },
            auth=(username1, password1),
        )

    scorer_name = response.json()["name"]

    _send_rest_tracking_post_request(
        client.tracking_uri,
        "/api/3.0/mlflow/scorers/permissions/create",
        json_payload={
            "experiment_id": experiment_id,
            "scorer_name": scorer_name,
            "username": username2,
            "permission": "READ",
        },
        auth=(username1, password1),
    )

    with User(username2, password2, monkeypatch):
        response = requests.get(
            url=client.tracking_uri + "/api/3.0/mlflow/scorers/get",
            params={
                "experiment_id": experiment_id,
                "name": scorer_name,
            },
            auth=(username2, password2),
        )
        response.raise_for_status()
        assert response.json()["scorer"]["scorer_name"] == scorer_name

    with User(username2, password2, monkeypatch):
        response = requests.delete(
            url=client.tracking_uri + "/api/3.0/mlflow/scorers/delete",
            json={
                "experiment_id": experiment_id,
                "name": scorer_name,
            },
            auth=(username2, password2),
        )
        with pytest.raises(requests.HTTPError, match="403"):
            response.raise_for_status()


def _graphql_query(tracking_uri, query, variables=None, auth=None):
    return requests.post(
        f"{tracking_uri}/graphql",
        json={"query": query, "variables": variables or {}},
        auth=auth,
    )


def test_graphql_requires_authentication(client, monkeypatch):
    monkeypatch.delenv(MLFLOW_TRACKING_USERNAME.name, raising=False)
    monkeypatch.delenv(MLFLOW_TRACKING_PASSWORD.name, raising=False)

    query = """
    query {
        mlflowGetExperiment(input: {experimentId: "0"}) {
            experiment {
                experimentId
                name
            }
        }
    }
    """
    response = _graphql_query(client.tracking_uri, query)
    assert response.status_code == 401


def test_graphql_get_experiment_authorization(client, monkeypatch):
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)

    with User(username1, password1, monkeypatch):
        experiment_id = client.create_experiment("graphql_test_exp")
        _send_rest_tracking_post_request(
            client.tracking_uri,
            "/api/2.0/mlflow/experiments/permissions/create",
            json_payload={
                "experiment_id": experiment_id,
                "username": username2,
                "permission": "NO_PERMISSIONS",
            },
            auth=(username1, password1),
        )

    query = """
    query($expId: String!) {
        mlflowGetExperiment(input: {experimentId: $expId}) {
            experiment {
                experimentId
                name
            }
        }
    }
    """

    # user1 (creator) should be able to read the experiment
    response = _graphql_query(
        client.tracking_uri,
        query,
        variables={"expId": experiment_id},
        auth=(username1, password1),
    )
    assert response.status_code == 200
    data = response.json()
    experiment_data = data["data"]["mlflowGetExperiment"]["experiment"]
    assert experiment_data["experimentId"] == experiment_id
    assert experiment_data["name"] == "graphql_test_exp"

    # user2 (NO_PERMISSIONS) should NOT be able to read the experiment
    response = _graphql_query(
        client.tracking_uri,
        query,
        variables={"expId": experiment_id},
        auth=(username2, password2),
    )
    assert response.status_code == 200
    data = response.json()
    # With authorization denied, the result should be null
    assert data.get("data", {}).get("mlflowGetExperiment") is None


def test_graphql_get_run_authorization(client, monkeypatch):
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)

    with User(username1, password1, monkeypatch):
        experiment_id = client.create_experiment("graphql_run_test_exp")
        run = client.create_run(experiment_id)
        run_id = run.info.run_id
        client.set_terminated(run_id)

        _send_rest_tracking_post_request(
            client.tracking_uri,
            "/api/2.0/mlflow/experiments/permissions/create",
            json_payload={
                "experiment_id": experiment_id,
                "username": username2,
                "permission": "NO_PERMISSIONS",
            },
            auth=(username1, password1),
        )

    query = """
    query($runId: String!) {
        mlflowGetRun(input: {runId: $runId}) {
            run {
                info {
                    runId
                    experimentId
                }
            }
        }
    }
    """

    # user1 (creator) should be able to read the run
    response = _graphql_query(
        client.tracking_uri,
        query,
        variables={"runId": run_id},
        auth=(username1, password1),
    )
    assert response.status_code == 200
    data = response.json()
    run_data = data["data"]["mlflowGetRun"]["run"]
    assert run_data["info"]["runId"] == run_id
    assert run_data["info"]["experimentId"] == experiment_id

    # user2 (NO_PERMISSIONS) should NOT be able to read the run
    response = _graphql_query(
        client.tracking_uri,
        query,
        variables={"runId": run_id},
        auth=(username2, password2),
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("data", {}).get("mlflowGetRun") is None


def test_graphql_search_runs_authorization(client, monkeypatch):
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)

    with User(username1, password1, monkeypatch):
        exp1_id = client.create_experiment("graphql_search_exp1")
        exp2_id = client.create_experiment("graphql_search_exp2")

        run1 = client.create_run(exp1_id)
        client.set_terminated(run1.info.run_id)

        run2 = client.create_run(exp2_id)
        client.set_terminated(run2.info.run_id)

        # Grant READ on exp1 to user2, NO_PERMISSIONS on exp2
        _send_rest_tracking_post_request(
            client.tracking_uri,
            "/api/2.0/mlflow/experiments/permissions/create",
            json_payload={
                "experiment_id": exp1_id,
                "username": username2,
                "permission": "READ",
            },
            auth=(username1, password1),
        )
        _send_rest_tracking_post_request(
            client.tracking_uri,
            "/api/2.0/mlflow/experiments/permissions/create",
            json_payload={
                "experiment_id": exp2_id,
                "username": username2,
                "permission": "NO_PERMISSIONS",
            },
            auth=(username1, password1),
        )

    query = """
    query($expIds: [String]!) {
        mlflowSearchRuns(input: {experimentIds: $expIds}) {
            runs {
                info {
                    runId
                    experimentId
                }
            }
        }
    }
    """

    # user1 should see both runs
    response = _graphql_query(
        client.tracking_uri,
        query,
        variables={"expIds": [exp1_id, exp2_id]},
        auth=(username1, password1),
    )
    assert response.status_code == 200
    data = response.json()
    runs = data.get("data", {}).get("mlflowSearchRuns", {}).get("runs", [])
    assert len(runs) == 2

    # user2 should only see run from exp1 (exp2 is filtered out)
    response = _graphql_query(
        client.tracking_uri,
        query,
        variables={"expIds": [exp1_id, exp2_id]},
        auth=(username2, password2),
    )
    assert response.status_code == 200
    data = response.json()
    runs = data.get("data", {}).get("mlflowSearchRuns", {}).get("runs", [])
    assert len(runs) == 1
    assert runs[0]["info"]["experimentId"] == exp1_id


def test_graphql_list_artifacts_authorization(client, monkeypatch):
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)

    with User(username1, password1, monkeypatch):
        experiment_id = client.create_experiment("graphql_artifacts_test_exp")
        run = client.create_run(experiment_id)
        run_id = run.info.run_id
        client.set_terminated(run_id)

        _send_rest_tracking_post_request(
            client.tracking_uri,
            "/api/2.0/mlflow/experiments/permissions/create",
            json_payload={
                "experiment_id": experiment_id,
                "username": username2,
                "permission": "NO_PERMISSIONS",
            },
            auth=(username1, password1),
        )

    query = """
    query($runId: String!) {
        mlflowListArtifacts(input: {runId: $runId}) {
            rootUri
            files {
                path
            }
        }
    }
    """

    # user1 (creator) should be able to list artifacts
    response = _graphql_query(
        client.tracking_uri,
        query,
        variables={"runId": run_id},
        auth=(username1, password1),
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("data", {}).get("mlflowListArtifacts") is not None

    # user2 (NO_PERMISSIONS) should NOT be able to list artifacts
    response = _graphql_query(
        client.tracking_uri,
        query,
        variables={"runId": run_id},
        auth=(username2, password2),
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("data", {}).get("mlflowListArtifacts") is None


def test_graphql_nonexistent_experiment(client, monkeypatch):
    username, password = create_user(client.tracking_uri)

    query = """
    query($expId: String!) {
        mlflowGetExperiment(input: {experimentId: $expId}) {
            experiment {
                experimentId
                name
            }
        }
    }
    """

    response = _graphql_query(
        client.tracking_uri,
        query,
        variables={"expId": "999999999"},
        auth=(username, password),
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("data", {}).get("mlflowGetExperiment") is None


def test_graphql_nonexistent_run(client, monkeypatch):
    username, password = create_user(client.tracking_uri)

    query = """
    query($runId: String!) {
        mlflowGetRun(input: {runId: $runId}) {
            run {
                info {
                    runId
                    experimentId
                }
            }
        }
    }
    """

    response = _graphql_query(
        client.tracking_uri,
        query,
        variables={"runId": "00000000000000000000000000000000"},
        auth=(username, password),
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("data", {}).get("mlflowGetRun") is None


def test_get_metric_history_bulk_interval_auth(client: MlflowClient, monkeypatch):
    username1, password1 = create_user(client.tracking_uri)
    username2, password2 = create_user(client.tracking_uri)

    with User(username1, password1, monkeypatch):
        experiment_id = client.create_experiment("test_metric_history_experiment")
        run = client.create_run(experiment_id)
        run_id = run.info.run_id
        client.log_metric(run_id, "test_metric", 1.0, step=0)
        client.log_metric(run_id, "test_metric", 2.0, step=1)

        _send_rest_tracking_post_request(
            client.tracking_uri,
            "/api/2.0/mlflow/experiments/permissions/create",
            json_payload={
                "experiment_id": experiment_id,
                "username": username2,
                "permission": "READ",
            },
            auth=(username1, password1),
        )

    with User(username2, password2, monkeypatch):
        response = requests.get(
            url=client.tracking_uri + "/ajax-api/2.0/mlflow/metrics/get-history-bulk-interval",
            params={
                "run_ids": run_id,
                "metric_key": "test_metric",
                "max_results": 100,
            },
            auth=(username2, password2),
        )
        response.raise_for_status()
        data = response.json()
        assert "metrics" in data
        assert len(data["metrics"]) == 2
