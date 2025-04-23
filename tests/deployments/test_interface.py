import pytest

from mlflow.deployments.interface import get_deploy_client
import mlflow.deployments.utils

mlflow.deployments.utils._deployments_target = None


def test_get_deploy_client_no_args():
    mlflow.deployments.utils._deployments_target = None
    assert get_deploy_client() is None


def test_get_deploy_client_none():
    mlflow.deployments.utils._deployments_target = None
    assert get_deploy_client(None) is None


def test_get_deploy_client_from_set_deployments_target():
    from mlflow.deployments import set_deployments_target

    set_deployments_target("databricks")
    assert get_deploy_client(None) is not None


@pytest.fixture
def set_deployment_envs(monkeypatch):
    monkeypatch.setenvs(
        {
            "MLFLOW_DEPLOYMENTS_TARGET": "databricks",
        }
    )


def test_get_deploy_client_from_env(set_deployment_envs):
    assert get_deploy_client(None) is not None
