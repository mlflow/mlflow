import pytest

from mlflow.deployments.interface import get_deploy_client
from mlflow.exceptions import MlflowException


def test_get_deploy_client():
    client = get_deploy_client("databricks")
    assert client.target_uri == "databricks"


def test_get_deploy_client_from_set_deployments_target(monkeypatch):
    monkeypatch.setattr("mlflow.deployments.utils._deployments_target", "http://localhost")
    client = get_deploy_client()
    assert client.target_uri == "http://localhost"


def test_get_deploy_client_from_env(monkeypatch):
    monkeypatch.setenvs(
        {
            "MLFLOW_DEPLOYMENTS_TARGET": "databricks",
        }
    )
    client = get_deploy_client()
    assert client.target_uri == "databricks"


def test_get_deploy_client_raise_if_no_target():
    with pytest.raises(
        MlflowException, match="No deployments target is found. Please either pass `target_uri`"
    ):
        get_deploy_client()
