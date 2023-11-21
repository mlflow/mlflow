from mlflow.deployments import get_deploy_client
from mlflow.deployments.mlflow import MLflowDeploymentClient


def test_get_deploy_client():
    client = get_deploy_client("http://localhost:5000")
    assert isinstance(client, MLflowDeploymentClient)
