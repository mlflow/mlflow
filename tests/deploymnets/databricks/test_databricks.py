from mlflow.deployments import get_deploy_client


def test_get_deploy_client():
    get_deploy_client("databricks")
