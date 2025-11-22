"""
Integration test which starts a local Tracking Server on an ephemeral port,
and ensures authentication is working.
"""

import json
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import jwt
import psutil
import pytest
import requests
from flask import Response, request

import mlflow
from mlflow import MlflowClient
from mlflow.entities.logged_model_status import LoggedModelStatus
from mlflow.environment_variables import (
    MLFLOW_ENABLE_WORKSPACES,
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
from mlflow.server import auth as auth_module
from mlflow.server.auth.permissions import MANAGE, NO_PERMISSIONS, READ
from mlflow.server.auth.routes import (
    CREATE_PROMPTLAB_RUN,
    GET_ARTIFACT,
    GET_METRIC_HISTORY_BULK,
    GET_METRIC_HISTORY_BULK_INTERVAL,
    GET_MODEL_VERSION_ARTIFACT,
    GET_REGISTERED_MODEL_PERMISSION,
    GET_SCORER_PERMISSION,
    GET_TRACE_ARTIFACT,
    SEARCH_DATASETS,
    UPLOAD_ARTIFACT,
)
from mlflow.server.auth.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracking._workspace import context as workspace_context
from mlflow.utils.os import is_windows
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME

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


def test_proxy_artifact_path_detection():
    assert auth_module._is_proxy_artifact_path("/api/2.0/mlflow-artifacts/artifacts/foo")
    assert auth_module._is_proxy_artifact_path("/ajax-api/2.0/mlflow-artifacts/artifacts/foo")


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
    assert permission["workspace"] == DEFAULT_WORKSPACE_NAME

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
    expected_message = (
        "Registered model permission with "
        f"workspace={DEFAULT_WORKSPACE_NAME}, name={rm.name} "
        f"and username={username1} not found"
    )
    assert response.json()["message"] == expected_message

    # now we should be able to create a model with the same name
    with User(username1, password1, monkeypatch):
        rm = client.create_registered_model("test_model")
    assert rm.name == "test_model"


def test_cleanup_workspace_permissions_handler(monkeypatch):
    calls: list[str] = []

    def mock_delete(workspace_name: str) -> None:
        calls.append(workspace_name)

    monkeypatch.setattr(
        auth_module.store,
        "delete_workspace_permissions_for_workspace",
        mock_delete,
        raising=True,
    )

    workspace_name = f"team-{random_str(10)}"
    with auth_module.app.test_request_context(
        f"/api/2.0/mlflow/workspaces/{workspace_name}", method="DELETE"
    ):
        request.view_args = {"workspace_name": workspace_name}
        response = Response(status=204)
        auth_module._after_request(response)

    assert calls == [workspace_name]


class _TrackingStore:
    def __init__(
        self,
        experiment_workspaces: dict[str, str],
        run_experiments: dict[str, str],
        trace_experiments: dict[str, str],
        experiment_names: dict[str, str] | None = None,
        logged_model_experiments: dict[str, str] | None = None,
    ):
        self._experiment_workspaces = experiment_workspaces
        self._run_experiments = run_experiments
        self._trace_experiments = trace_experiments
        self._experiment_names = experiment_names or {}
        self._logged_model_experiments = logged_model_experiments or {}

    def get_experiment(self, experiment_id: str):
        return SimpleNamespace(workspace=self._experiment_workspaces[experiment_id])

    def get_experiment_by_name(self, experiment_name: str):
        experiment_id = self._experiment_names.get(experiment_name)
        if experiment_id is None:
            return None
        return SimpleNamespace(
            experiment_id=experiment_id,
            workspace=self._experiment_workspaces[experiment_id],
        )

    def get_run(self, run_id: str):
        return SimpleNamespace(info=SimpleNamespace(experiment_id=self._run_experiments[run_id]))

    def get_trace_info(self, request_id: str):
        return SimpleNamespace(experiment_id=self._trace_experiments[request_id])

    def get_logged_model(self, model_id: str):
        experiment_id = self._logged_model_experiments[model_id]
        return SimpleNamespace(experiment_id=experiment_id)


class _RegistryStore:
    def __init__(self, model_workspaces: dict[str, str]):
        self._model_workspaces = model_workspaces

    def get_registered_model(self, name: str):
        return SimpleNamespace(workspace=self._model_workspaces[name])


@pytest.fixture
def workspace_permission_setup(tmp_path, monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    monkeypatch.setattr(
        auth_module,
        "auth_config",
        auth_module.auth_config._replace(default_permission=NO_PERMISSIONS.name),
    )

    db_uri = f"sqlite:///{tmp_path / 'auth-store.db'}"
    auth_store = SqlAlchemyStore()
    auth_store.init_db(db_uri)
    monkeypatch.setattr(auth_module, "store", auth_store, raising=False)

    username = "alice"
    auth_store.create_user(username, "supersecurepassword", is_admin=False)

    tracking_store = _TrackingStore(
        experiment_workspaces={"exp-1": "team-a", "exp-2": "team-a", "1": "team-a"},
        run_experiments={"run-1": "exp-1", "run-2": "exp-2"},
        trace_experiments={"trace-1": "exp-1"},
        experiment_names={"Primary Experiment": "exp-1"},
        logged_model_experiments={"model-1": "exp-1"},
    )
    monkeypatch.setattr(auth_module, "_get_tracking_store", lambda: tracking_store)

    registry_store = _RegistryStore({"model-xyz": "team-a"})
    monkeypatch.setattr(auth_module, "_get_model_registry_store", lambda: registry_store)

    monkeypatch.setattr(
        auth_module,
        "authenticate_request",
        lambda: SimpleNamespace(username=username),
    )

    auth_store.set_workspace_permission("team-a", username, "experiments", MANAGE.name)
    auth_store.set_workspace_permission("team-a", username, "registered_models", READ.name)

    token = workspace_context.set_current_workspace("team-a")

    try:
        yield {"store": auth_store, "username": username}
    finally:
        workspace_context.reset_workspace(token)
        workspace_context.clear_workspace()
        auth_store.engine.dispose()


def _set_workspace_permission(
    store: SqlAlchemyStore, username: str, resource_type: str, permission: str
):
    store.set_workspace_permission("team-a", username, resource_type, permission)


def test_workspace_permission_grants_default_access(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")

    default_permission = MANAGE.name
    monkeypatch.setattr(
        auth_module,
        "auth_config",
        auth_module.auth_config._replace(
            default_permission=default_permission,
            grant_default_workspace_access=True,
        ),
        raising=False,
    )

    class DummyStore:
        def supports_workspaces(self):
            return True

        def get_workspace_permission(self, workspace_name, username, resource_type):
            return None

        def list_accessible_workspace_names(self, username):
            return []

    dummy_store = DummyStore()
    monkeypatch.setattr(auth_module, "store", dummy_store, raising=False)

    default_workspace = DEFAULT_WORKSPACE_NAME
    monkeypatch.setattr(auth_module, "_resolve_default_workspace_name", lambda: default_workspace)

    auth = SimpleNamespace(username="alice")
    permission = auth_module._workspace_permission(auth, "experiments", default_workspace)
    assert permission is not None
    assert permission.can_manage

    token = workspace_context.set_current_workspace(default_workspace)
    try:
        monkeypatch.setattr(auth_module, "authenticate_request", lambda: auth)
        assert auth_module.validate_can_create_experiment()
    finally:
        workspace_context.reset_workspace(token)
        workspace_context.clear_workspace()


def test_filter_list_workspaces_includes_default_when_autogrant(monkeypatch):
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    auth = SimpleNamespace(username="alice")
    monkeypatch.setattr(auth_module, "authenticate_request", lambda: auth)
    monkeypatch.setattr(
        auth_module,
        "auth_config",
        auth_module.auth_config._replace(
            grant_default_workspace_access=True,
            default_permission=READ.name,
        ),
        raising=False,
    )

    default_workspace = "team-default"
    monkeypatch.setattr(auth_module, "_resolve_default_workspace_name", lambda: default_workspace)

    class DummyStore:
        def supports_workspaces(self):
            return True

        def list_accessible_workspace_names(self, username):
            return []

    monkeypatch.setattr(auth_module, "store", DummyStore(), raising=False)

    response = Response(
        json.dumps(
            {
                "workspaces": [
                    {"name": default_workspace},
                    {"name": "other-workspace"},
                ]
            }
        ),
        mimetype="application/json",
    )

    auth_module.filter_list_workspaces(response)
    payload = json.loads(response.get_data(as_text=True))
    assert payload["workspaces"] == [{"name": default_workspace}]


def test_validate_can_view_workspace_allows_default_autogrant(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)
    auth = SimpleNamespace(username="alice")
    monkeypatch.setattr(auth_module, "authenticate_request", lambda: auth)
    monkeypatch.setattr(
        auth_module,
        "auth_config",
        auth_module.auth_config._replace(
            grant_default_workspace_access=True,
            default_permission=READ.name,
        ),
        raising=False,
    )

    default_workspace = "team-default"
    monkeypatch.setattr(auth_module, "_resolve_default_workspace_name", lambda: default_workspace)

    class DummyStore:
        def supports_workspaces(self):
            return True

        def list_accessible_workspace_names(self, username):
            return []

    monkeypatch.setattr(auth_module, "store", DummyStore(), raising=False)

    with auth_module.app.test_request_context(
        f"/api/2.0/mlflow/workspaces/{default_workspace}", method="GET"
    ):
        request.view_args = {"workspace_name": default_workspace}
        assert auth_module.validate_can_view_workspace()

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/workspaces/other-team", method="GET"
    ):
        request.view_args = {"workspace_name": "other-team"}
        assert not auth_module.validate_can_view_workspace()


def test_experiment_validators_allow_manage_permission(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, "experiments", MANAGE.name)

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/experiments/get", method="GET", query_string={"experiment_id": "exp-1"}
    ):
        assert auth_module.validate_can_read_experiment()
        assert auth_module.validate_can_update_experiment()
        assert auth_module.validate_can_delete_experiment()
        assert auth_module.validate_can_manage_experiment()

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/experiments/get-by-name",
        method="GET",
        query_string={"experiment_name": "Primary Experiment"},
    ):
        assert auth_module.validate_can_read_experiment_by_name()

    token = workspace_context.set_current_workspace("team-a")
    try:
        assert auth_module.validate_can_create_experiment()
    finally:
        workspace_context.reset_workspace(token)


def test_experiment_validators_read_permission_blocks_writes(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, "experiments", READ.name)

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/experiments/get", method="GET", query_string={"experiment_id": "exp-1"}
    ):
        assert auth_module.validate_can_read_experiment()
        assert not auth_module.validate_can_update_experiment()
        assert not auth_module.validate_can_delete_experiment()
        assert not auth_module.validate_can_manage_experiment()

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/experiments/get-by-name",
        method="GET",
        query_string={"experiment_name": "Primary Experiment"},
    ):
        assert auth_module.validate_can_read_experiment_by_name()

    token = workspace_context.set_current_workspace("team-a")
    try:
        assert not auth_module.validate_can_create_experiment()
    finally:
        workspace_context.reset_workspace(token)


def test_experiment_artifact_proxy_validators_respect_permissions(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, "experiments", MANAGE.name)

    with auth_module.app.test_request_context(
        "/ajax-api/2.0/mlflow-artifacts/artifacts/1/path",
        method="GET",
    ):
        request.view_args = {"artifact_path": "1/path"}
        assert auth_module.validate_can_read_experiment_artifact_proxy()
        assert auth_module.validate_can_update_experiment_artifact_proxy()
        assert auth_module.validate_can_delete_experiment_artifact_proxy()

    _set_workspace_permission(store, username, "experiments", READ.name)

    with auth_module.app.test_request_context(
        "/ajax-api/2.0/mlflow-artifacts/artifacts/1/path",
        method="GET",
    ):
        request.view_args = {"artifact_path": "1/path"}
        assert auth_module.validate_can_read_experiment_artifact_proxy()
        assert not auth_module.validate_can_update_experiment_artifact_proxy()
        assert not auth_module.validate_can_delete_experiment_artifact_proxy()


def test_experiment_artifact_proxy_without_experiment_id_uses_workspace_permissions(
    workspace_permission_setup,
):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, "experiments", READ.name)

    with auth_module.app.test_request_context(
        "/ajax-api/2.0/mlflow-artifacts/artifacts/uploads/path",
        method="GET",
    ):
        request.view_args = {"artifact_path": "uploads/path"}
        assert auth_module.validate_can_read_experiment_artifact_proxy()
        assert not auth_module.validate_can_update_experiment_artifact_proxy()


def test_experiment_artifact_proxy_without_experiment_id_denied_without_workspace_permission(
    workspace_permission_setup,
):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, "experiments", NO_PERMISSIONS.name)

    with auth_module.app.test_request_context(
        "/ajax-api/2.0/mlflow-artifacts/artifacts/uploads/path",
        method="GET",
    ):
        request.view_args = {"artifact_path": "uploads/path"}
        assert not auth_module.validate_can_read_experiment_artifact_proxy()


def test_filter_experiment_ids_respects_workspace_permissions(
    workspace_permission_setup, monkeypatch
):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    monkeypatch.setattr(auth_module, "sender_is_admin", lambda: False)

    experiment_ids = ["exp-1", "exp-2"]
    assert auth_module.filter_experiment_ids(experiment_ids) == experiment_ids

    _set_workspace_permission(store, username, "experiments", NO_PERMISSIONS.name)
    assert auth_module.filter_experiment_ids(experiment_ids) == []


def test_run_validators_allow_manage_permission(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, "experiments", MANAGE.name)

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/runs/get", method="GET", query_string={"run_id": "run-1"}
    ):
        assert auth_module.validate_can_read_run()
        assert auth_module.validate_can_update_run()
        assert auth_module.validate_can_delete_run()
        assert auth_module.validate_can_manage_run()


def test_run_validators_read_permission_blocks_writes(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, "experiments", READ.name)

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/runs/get", method="GET", query_string={"run_id": "run-1"}
    ):
        assert auth_module.validate_can_read_run()
        assert not auth_module.validate_can_update_run()
        assert not auth_module.validate_can_delete_run()
        assert not auth_module.validate_can_manage_run()


def test_logged_model_validators_respect_permissions(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]

    _set_workspace_permission(store, username, "experiments", MANAGE.name)
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/logged-models/get",
        method="GET",
        query_string={"model_id": "model-1"},
    ):
        assert auth_module.validate_can_read_logged_model()
        assert auth_module.validate_can_update_logged_model()
        assert auth_module.validate_can_delete_logged_model()
        assert auth_module.validate_can_manage_logged_model()

    _set_workspace_permission(store, username, "experiments", READ.name)
    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/logged-models/get",
        method="GET",
        query_string={"model_id": "model-1"},
    ):
        assert auth_module.validate_can_read_logged_model()
        assert not auth_module.validate_can_update_logged_model()
        assert not auth_module.validate_can_delete_logged_model()
        assert not auth_module.validate_can_manage_logged_model()


def test_scorer_validators_use_workspace_permissions(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, "experiments", MANAGE.name)

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/scorers/get",
        method="GET",
        query_string={"experiment_id": "exp-1", "name": "score-1"},
    ):
        assert auth_module.validate_can_read_scorer()
        assert auth_module.validate_can_update_scorer()
        assert auth_module.validate_can_delete_scorer()
        assert auth_module.validate_can_manage_scorer()

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/scorers/permissions/create",
        method="POST",
        json={
            "experiment_id": "exp-1",
            "scorer_name": "score-1",
            "username": "bob",
            "permission": "READ",
        },
    ):
        assert auth_module.validate_can_manage_scorer_permission()


def test_scorer_validators_read_permission_blocks_writes(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    _set_workspace_permission(store, username, "experiments", READ.name)

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/scorers/get",
        method="GET",
        query_string={"experiment_id": "exp-1", "name": "score-1"},
    ):
        assert auth_module.validate_can_read_scorer()
        assert not auth_module.validate_can_update_scorer()
        assert not auth_module.validate_can_delete_scorer()
        assert not auth_module.validate_can_manage_scorer()

    with auth_module.app.test_request_context(
        "/api/3.0/mlflow/scorers/permissions/create",
        method="POST",
        json={
            "experiment_id": "exp-1",
            "scorer_name": "score-1",
            "username": "bob",
            "permission": "READ",
        },
    ):
        assert not auth_module.validate_can_manage_scorer_permission()


def test_registered_model_validators_require_manage_for_writes(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]

    token = workspace_context.set_current_workspace("team-a")
    try:
        _set_workspace_permission(store, username, "registered_models", MANAGE.name)
        with auth_module.app.test_request_context(
            "/api/2.0/mlflow/registered-models/get",
            method="GET",
            query_string={"name": "model-xyz"},
        ):
            request_token = workspace_context.set_current_workspace("team-a")
            try:
                assert auth_module.validate_can_read_registered_model()
                assert auth_module.validate_can_update_registered_model()
                assert auth_module.validate_can_delete_registered_model()
                assert auth_module.validate_can_manage_registered_model()
            finally:
                workspace_context.reset_workspace(request_token)
        perm = auth_module._workspace_permission(
            auth_module.authenticate_request(), "registered_models", "team-a"
        )
        assert perm is not None
        assert perm.can_manage
        inner_token = workspace_context.set_current_workspace("team-a")
        try:
            assert workspace_context.get_current_workspace() == "team-a"
            assert auth_module.validate_can_create_registered_model()
        finally:
            workspace_context.reset_workspace(inner_token)

        _set_workspace_permission(store, username, "registered_models", READ.name)
        with auth_module.app.test_request_context(
            "/api/2.0/mlflow/registered-models/get",
            method="GET",
            query_string={"name": "model-xyz"},
        ):
            request_token = workspace_context.set_current_workspace("team-a")
            try:
                assert auth_module.validate_can_read_registered_model()
                assert not auth_module.validate_can_update_registered_model()
                assert not auth_module.validate_can_delete_registered_model()
                assert not auth_module.validate_can_manage_registered_model()
            finally:
                workspace_context.reset_workspace(request_token)
        inner_token = workspace_context.set_current_workspace("team-a")
        try:
            assert not auth_module.validate_can_create_registered_model()
        finally:
            workspace_context.reset_workspace(inner_token)
    finally:
        workspace_context.reset_workspace(token)


def test_validate_can_view_workspace_requires_access(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/workspaces/team-a",
        method="GET",
    ):
        request.view_args = {"workspace_name": "team-a"}
        assert auth_module.validate_can_view_workspace()

    store.delete_workspace_permission("team-a", username, "experiments")
    store.delete_workspace_permission("team-a", username, "registered_models")

    with auth_module.app.test_request_context(
        "/api/2.0/mlflow/workspaces/team-a",
        method="GET",
    ):
        request.view_args = {"workspace_name": "team-a"}
        assert not auth_module.validate_can_view_workspace()


def test_run_artifact_validators_use_workspace_permissions(workspace_permission_setup):
    with auth_module.app.test_request_context(
        GET_ARTIFACT,
        method="GET",
        query_string={"run_id": "run-1"},
    ):
        assert auth_module.validate_can_read_run_artifact()

    with auth_module.app.test_request_context(
        UPLOAD_ARTIFACT,
        method="POST",
        query_string={"run_id": "run-1"},
    ):
        assert auth_module.validate_can_update_run_artifact()


def test_model_version_artifact_validator_uses_workspace_permissions(workspace_permission_setup):
    with auth_module.app.test_request_context(
        GET_MODEL_VERSION_ARTIFACT,
        method="GET",
        query_string={"name": "model-xyz"},
    ):
        assert auth_module.validate_can_read_model_version_artifact()


def test_metric_history_bulk_validator_uses_workspace_permissions(workspace_permission_setup):
    with auth_module.app.test_request_context(
        GET_METRIC_HISTORY_BULK,
        method="GET",
        query_string=[("run_id", "run-1"), ("run_id", "run-2")],
    ):
        assert auth_module.validate_can_read_metric_history_bulk()


def test_metric_history_bulk_interval_validator_uses_workspace_permissions(
    workspace_permission_setup,
):
    with auth_module.app.test_request_context(
        GET_METRIC_HISTORY_BULK_INTERVAL,
        method="GET",
        query_string=[
            ("run_id", "run-1"),
            ("run_id", "run-2"),
            ("metric_key", "loss"),
        ],
    ):
        assert auth_module.validate_can_read_metric_history_bulk_interval()


def test_search_datasets_validator_uses_workspace_permissions(workspace_permission_setup):
    with auth_module.app.test_request_context(
        SEARCH_DATASETS,
        method="POST",
        json={"experiment_ids": ["exp-1", "exp-2"]},
    ):
        assert auth_module.validate_can_search_datasets()


def test_create_promptlab_run_validator_uses_workspace_permissions(workspace_permission_setup):
    with auth_module.app.test_request_context(
        CREATE_PROMPTLAB_RUN,
        method="POST",
        json={"experiment_id": "exp-2"},
    ):
        assert auth_module.validate_can_create_promptlab_run()


def test_trace_artifact_validator_uses_workspace_permissions(workspace_permission_setup):
    with auth_module.app.test_request_context(
        GET_TRACE_ARTIFACT,
        method="GET",
        query_string={"request_id": "trace-1"},
    ):
        assert auth_module.validate_can_read_trace_artifact()


def test_experiment_artifact_proxy_without_workspaces_falls_back_to_default(monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "false")
    monkeypatch.setattr(
        auth_module,
        "auth_config",
        auth_module.auth_config._replace(default_permission=READ.name),
        raising=False,
    )
    monkeypatch.setattr(
        auth_module,
        "authenticate_request",
        lambda: SimpleNamespace(username="carol"),
    )

    with auth_module.app.test_request_context(
        "/ajax-api/2.0/mlflow-artifacts/artifacts/uploads/path",
        method="GET",
    ):
        request.view_args = {"artifact_path": "uploads/path"}
        assert auth_module.validate_can_read_experiment_artifact_proxy()


def test_run_artifact_validators_denied_without_workspace_permission(workspace_permission_setup):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    store.set_workspace_permission("team-a", username, "experiments", NO_PERMISSIONS.name)

    with auth_module.app.test_request_context(
        GET_ARTIFACT,
        method="GET",
        query_string={"run_id": "run-1"},
    ):
        assert not auth_module.validate_can_read_run_artifact()

    with auth_module.app.test_request_context(
        UPLOAD_ARTIFACT,
        method="POST",
        query_string={"run_id": "run-1"},
    ):
        assert not auth_module.validate_can_update_run_artifact()


def test_model_version_artifact_validator_denied_without_workspace_permission(
    workspace_permission_setup,
):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    store.set_workspace_permission("team-a", username, "registered_models", NO_PERMISSIONS.name)

    with auth_module.app.test_request_context(
        GET_MODEL_VERSION_ARTIFACT,
        method="GET",
        query_string={"name": "model-xyz"},
    ):
        assert not auth_module.validate_can_read_model_version_artifact()


def test_metric_history_bulk_validator_denied_without_workspace_permission(
    workspace_permission_setup,
):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    store.set_workspace_permission("team-a", username, "experiments", NO_PERMISSIONS.name)

    with auth_module.app.test_request_context(
        GET_METRIC_HISTORY_BULK,
        method="GET",
        query_string=[("run_id", "run-1"), ("run_id", "run-2")],
    ):
        assert not auth_module.validate_can_read_metric_history_bulk()


def test_metric_history_bulk_interval_validator_denied_without_workspace_permission(
    workspace_permission_setup,
):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    store.set_workspace_permission("team-a", username, "experiments", NO_PERMISSIONS.name)

    with auth_module.app.test_request_context(
        GET_METRIC_HISTORY_BULK_INTERVAL,
        method="GET",
        query_string=[
            ("run_id", "run-1"),
            ("run_id", "run-2"),
            ("metric_key", "loss"),
        ],
    ):
        assert not auth_module.validate_can_read_metric_history_bulk_interval()


def test_search_datasets_validator_denied_without_workspace_permission(
    workspace_permission_setup,
):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    store.set_workspace_permission("team-a", username, "experiments", NO_PERMISSIONS.name)

    with auth_module.app.test_request_context(
        SEARCH_DATASETS,
        method="POST",
        json={"experiment_ids": ["exp-1", "exp-2"]},
    ):
        assert not auth_module.validate_can_search_datasets()


def test_create_promptlab_run_validator_denied_without_workspace_permission(
    workspace_permission_setup,
):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    store.set_workspace_permission("team-a", username, "experiments", NO_PERMISSIONS.name)

    with auth_module.app.test_request_context(
        CREATE_PROMPTLAB_RUN,
        method="POST",
        json={"experiment_id": "exp-2"},
    ):
        assert not auth_module.validate_can_create_promptlab_run()


def test_trace_artifact_validator_denied_without_workspace_permission(
    workspace_permission_setup,
):
    store = workspace_permission_setup["store"]
    username = workspace_permission_setup["username"]
    store.set_workspace_permission("team-a", username, "experiments", NO_PERMISSIONS.name)

    with auth_module.app.test_request_context(
        GET_TRACE_ARTIFACT,
        method="GET",
        query_string={"request_id": "trace-1"},
    ):
        assert not auth_module.validate_can_read_trace_artifact()


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
                    client.log_artifact(run.info.run_id, tmp_file_with_numbers)
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
