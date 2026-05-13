from contextlib import contextmanager

import pytest
import requests

from mlflow import MlflowException
from mlflow.environment_variables import (
    MLFLOW_AUTH_CONFIG_PATH,
    MLFLOW_ENABLE_WORKSPACES,
    MLFLOW_FLASK_SERVER_SECRET_KEY,
    MLFLOW_RBAC_SEED_DEFAULT_ROLES,
    MLFLOW_TRACKING_PASSWORD,
    MLFLOW_TRACKING_USERNAME,
    MLFLOW_WORKSPACE_STORE_URI,
)
from mlflow.protos.databricks_pb2 import PERMISSION_DENIED, UNAUTHENTICATED, ErrorCode
from mlflow.server.auth.client import AuthServiceClient
from mlflow.utils.os import is_windows
from mlflow.utils.workspace_utils import WORKSPACE_HEADER_NAME

from tests.helper_functions import random_str
from tests.server.auth.auth_test_utils import (
    ADMIN_PASSWORD,
    ADMIN_USERNAME,
    User,
    create_user,
    grant_role_permission,
    write_isolated_auth_config,
)
from tests.tracking.integration_test_utils import _init_server


@pytest.fixture(autouse=True)
def clear_credentials(monkeypatch):
    monkeypatch.delenv(MLFLOW_TRACKING_USERNAME.name, raising=False)
    monkeypatch.delenv(MLFLOW_TRACKING_PASSWORD.name, raising=False)


@pytest.fixture
def workspace_client(tmp_path):
    auth_config_path = write_isolated_auth_config(tmp_path)
    path = tmp_path.joinpath("sqlalchemy.db").as_uri()
    backend_uri = ("sqlite://" if is_windows() else "sqlite:////") + path[len("file://") :]

    with _init_server(
        backend_uri=backend_uri,
        root_artifact_uri=tmp_path.joinpath("artifacts").as_uri(),
        app="mlflow.server.auth:create_app",
        extra_env={
            MLFLOW_FLASK_SERVER_SECRET_KEY.name: "my-secret-key",
            MLFLOW_AUTH_CONFIG_PATH.name: str(auth_config_path),
            MLFLOW_ENABLE_WORKSPACES.name: "true",
            MLFLOW_WORKSPACE_STORE_URI.name: backend_uri,
            # Force seeding on so tests don't depend on the caller's shell env.
            MLFLOW_RBAC_SEED_DEFAULT_ROLES.name: "true",
        },
        server_type="flask",
    ) as url:
        yield AuthServiceClient(url), url


@contextmanager
def assert_unauthenticated():
    with pytest.raises(MlflowException, match=r"You are not authenticated.") as exception_context:
        yield
    assert exception_context.value.error_code == ErrorCode.Name(UNAUTHENTICATED)


@contextmanager
def assert_unauthorized():
    with pytest.raises(MlflowException, match=r"Permission denied.") as exception_context:
        yield
    assert exception_context.value.error_code == ErrorCode.Name(PERMISSION_DENIED)


def _create_workspace(tracking_uri: str, workspace_name: str):
    response = requests.post(
        f"{tracking_uri}/api/3.0/mlflow/workspaces",
        json={"name": workspace_name},
        auth=(ADMIN_USERNAME, ADMIN_PASSWORD),
    )
    response.raise_for_status()


@pytest.fixture
def workspace_setup(workspace_client):
    client, tracking_uri = workspace_client
    workspace_name = f"team-{random_str()}"
    _create_workspace(tracking_uri, workspace_name)
    username, password = create_user(tracking_uri)
    return client, tracking_uri, workspace_name, username, password


def _create_experiment(
    tracking_uri: str, workspace_name: str, auth: tuple[str, str] = (ADMIN_USERNAME, ADMIN_PASSWORD)
) -> str:
    resp = requests.post(
        f"{tracking_uri}/api/2.0/mlflow/experiments/create",
        json={"name": f"exp-{random_str()}"},
        auth=auth,
        headers={WORKSPACE_HEADER_NAME: workspace_name},
    )
    assert resp.ok, f"create_experiment failed with {resp.status_code}: {resp.text}"
    return resp.json()["experiment_id"]


def _create_run(
    tracking_uri: str,
    workspace_name: str,
    experiment_id: str,
    auth: tuple[str, str] = (ADMIN_USERNAME, ADMIN_PASSWORD),
) -> str:
    resp = requests.post(
        f"{tracking_uri}/api/2.0/mlflow/runs/create",
        json={"experiment_id": experiment_id},
        auth=auth,
        headers={WORKSPACE_HEADER_NAME: workspace_name},
    )
    assert resp.ok, f"create_run failed with {resp.status_code}: {resp.text}"
    return resp.json()["run"]["info"]["run_id"]


def _create_registered_model(
    tracking_uri: str,
    workspace_name: str,
    model_name: str,
    auth: tuple[str, str] = (ADMIN_USERNAME, ADMIN_PASSWORD),
):
    resp = requests.post(
        f"{tracking_uri}/api/2.0/mlflow/registered-models/create",
        json={"name": model_name},
        auth=auth,
        headers={WORKSPACE_HEADER_NAME: workspace_name},
    )
    assert resp.ok, f"create_registered_model failed with {resp.status_code}: {resp.text}"


def _create_model_version(
    tracking_uri: str,
    workspace_name: str,
    model_name: str,
    run_id: str,
    auth: tuple[str, str] = (ADMIN_USERNAME, ADMIN_PASSWORD),
) -> str:
    resp = requests.post(
        f"{tracking_uri}/api/2.0/mlflow/model-versions/create",
        json={"name": model_name, "source": f"runs:/{run_id}/model", "run_id": run_id},
        auth=auth,
        headers={WORKSPACE_HEADER_NAME: workspace_name},
    )
    assert resp.ok, f"create_model_version failed with {resp.status_code}: {resp.text}"
    return resp.json()["model_version"]["version"]


def _graphql_search_runs(
    tracking_uri: str, workspace_name: str, auth: tuple[str, str], experiment_ids: list[str]
):
    query = """
    query SearchRuns($input: MlflowSearchRunsInput){
      mlflowSearchRuns(input: $input){
        runs { info { runId experimentId } }
      }
    }
    """
    variables = {"input": {"experimentIds": experiment_ids, "maxResults": 50}}
    resp = requests.post(
        f"{tracking_uri}/graphql",
        json={"query": query, "variables": variables},
        auth=auth,
        headers={WORKSPACE_HEADER_NAME: workspace_name},
    )
    resp.raise_for_status()
    payload = resp.json()
    assert payload.get("errors") in (None, [])
    search_runs = payload["data"]["mlflowSearchRuns"]
    if search_runs is None:
        return []
    return search_runs["runs"]


def _graphql_search_model_versions(
    tracking_uri: str,
    workspace_name: str,
    auth: tuple[str, str],
    filter_string: str | None = None,
):
    query = """
    query SearchModelVersions($input: MlflowSearchModelVersionsInput){
      mlflowSearchModelVersions(input: $input){
        modelVersions { name version runId }
      }
    }
    """
    variables = {"input": {"filter": filter_string}}
    resp = requests.post(
        f"{tracking_uri}/graphql",
        json={"query": query, "variables": variables},
        auth=auth,
        headers={WORKSPACE_HEADER_NAME: workspace_name},
    )
    resp.raise_for_status()
    payload = resp.json()
    assert payload.get("errors") in (None, [])
    return payload["data"]["mlflowSearchModelVersions"]["modelVersions"]


def test_create_workspace_seeds_default_roles(workspace_client, monkeypatch):
    # The workspace_client fixture forces ``MLFLOW_RBAC_SEED_DEFAULT_ROLES=true`` in
    # the server subprocess so this test is deterministic regardless of the caller's
    # shell environment.
    client, tracking_uri = workspace_client
    workspace_name = f"team-{random_str(10)}"
    _create_workspace(tracking_uri, workspace_name)

    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        roles = client.list_roles(workspace_name)

    role_names = sorted(r.name for r in roles)
    assert role_names == ["admin", "user"]

    # Each role got its expected permission row. Look up by name and inspect.
    by_name = {r.name: r for r in roles}
    with User(ADMIN_USERNAME, ADMIN_PASSWORD, monkeypatch):
        admin_perms = client.list_role_permissions(by_name["admin"].id)
        user_perms = client.list_role_permissions(by_name["user"].id)

    # The simplified two-tier model: ``admin`` carries the workspace-admin grant
    # (resource_type='workspace', MANAGE), while ``user`` carries the
    # workspace-wide access+create grant (resource_type='workspace', USE).
    assert [(p.resource_type, p.resource_pattern, p.permission) for p in admin_perms] == [
        ("workspace", "*", "MANAGE")
    ]
    assert [(p.resource_type, p.resource_pattern, p.permission) for p in user_perms] == [
        ("workspace", "*", "USE")
    ]


def test_run_access_controls_across_workspaces(workspace_setup, monkeypatch):
    client, tracking_uri, workspace_a, username, password = workspace_setup
    workspace_b = f"team-{random_str()}"
    _create_workspace(tracking_uri, workspace_b)

    # Allow the regular user to create resources in both workspaces for setup.
    grant_role_permission(tracking_uri, username, "workspace", "*", "MANAGE", workspace=workspace_a)
    grant_role_permission(tracking_uri, username, "workspace", "*", "MANAGE", workspace=workspace_b)

    exp_a = _create_experiment(tracking_uri, workspace_a, auth=(username, password))
    run_a = _create_run(tracking_uri, workspace_a, exp_a, auth=(username, password))
    exp_b = _create_experiment(tracking_uri, workspace_b, auth=(username, password))
    run_b = _create_run(tracking_uri, workspace_b, exp_b, auth=(username, password))

    # Use a separate limited user who only has access to workspace A.
    limited_user, limited_password = create_user(tracking_uri)
    grant_role_permission(
        tracking_uri, limited_user, "workspace", "*", "USE", workspace=workspace_a
    )

    # Positive: limited user can read run in workspace A.
    resp_ok = requests.get(
        f"{tracking_uri}/api/2.0/mlflow/runs/get",
        params={"run_id": run_a},
        auth=(limited_user, limited_password),
        headers={WORKSPACE_HEADER_NAME: workspace_a},
    )
    assert resp_ok.status_code == 200

    # REST: run in workspace B should be forbidden for limited user.
    resp = requests.get(
        f"{tracking_uri}/api/2.0/mlflow/runs/get",
        params={"run_id": run_b},
        auth=(limited_user, limited_password),
        headers={WORKSPACE_HEADER_NAME: workspace_b},
    )
    assert resp.status_code == 403
    assert "Permission denied" in resp.text

    # GraphQL: only runs from authorized workspace should appear.
    runs = _graphql_search_runs(
        tracking_uri,
        workspace_a,
        auth=(limited_user, limited_password),
        experiment_ids=[exp_a, exp_b],
    )
    returned_ids = {run["info"]["runId"] for run in runs}
    assert returned_ids == {run_a}

    # Switching to an unauthorized workspace should yield no readable runs.
    runs_in_b = _graphql_search_runs(
        tracking_uri, workspace_b, auth=(limited_user, limited_password), experiment_ids=[exp_b]
    )
    assert runs_in_b == []


def test_registered_model_access_controls_across_workspaces(workspace_setup, monkeypatch):
    client, tracking_uri, workspace_a, username, password = workspace_setup
    workspace_b = f"team-{random_str()}"
    _create_workspace(tracking_uri, workspace_b)

    grant_role_permission(tracking_uri, username, "workspace", "*", "MANAGE", workspace=workspace_a)
    grant_role_permission(tracking_uri, username, "workspace", "*", "MANAGE", workspace=workspace_b)

    # Create resources in both workspaces as the regular user.
    exp_a = _create_experiment(tracking_uri, workspace_a, auth=(username, password))
    run_a = _create_run(tracking_uri, workspace_a, exp_a, auth=(username, password))
    model_a = f"model-a-{random_str()}"
    _create_registered_model(tracking_uri, workspace_a, model_a, auth=(username, password))
    _create_model_version(tracking_uri, workspace_a, model_a, run_a, auth=(username, password))

    exp_b = _create_experiment(tracking_uri, workspace_b, auth=(username, password))
    run_b = _create_run(tracking_uri, workspace_b, exp_b, auth=(username, password))
    model_b = f"model-b-{random_str()}"
    _create_registered_model(tracking_uri, workspace_b, model_b, auth=(username, password))
    _create_model_version(tracking_uri, workspace_b, model_b, run_b, auth=(username, password))

    limited_user, limited_password = create_user(tracking_uri)
    grant_role_permission(
        tracking_uri, limited_user, "workspace", "*", "USE", workspace=workspace_a
    )

    # Positive: limited user can read model in authorized workspace.
    resp_ok = requests.get(
        f"{tracking_uri}/api/2.0/mlflow/registered-models/get",
        params={"name": model_a},
        auth=(limited_user, limited_password),
        headers={WORKSPACE_HEADER_NAME: workspace_a},
    )
    assert resp_ok.status_code == 200

    # GraphQL: only model versions from the permitted workspace should appear.
    versions = _graphql_search_model_versions(
        tracking_uri, workspace_a, auth=(limited_user, limited_password), filter_string=None
    )
    assert {v["name"] for v in versions} == {model_a}

    # REST: direct model get in another workspace should be forbidden.
    resp = requests.get(
        f"{tracking_uri}/api/2.0/mlflow/registered-models/get",
        params={"name": model_b},
        auth=(limited_user, limited_password),
        headers={WORKSPACE_HEADER_NAME: workspace_b},
    )
    assert resp.status_code == 403
    assert "Permission denied" in resp.text
