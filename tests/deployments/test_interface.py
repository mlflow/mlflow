import pytest

from mlflow.deployments.interface import get_deploy_client


def test_get_deploy_client_no_args():
    assert get_deploy_client() is None


def test_get_deploy_client_none():
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
