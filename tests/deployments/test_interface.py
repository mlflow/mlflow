from mlflow.deployments.interface import get_deploy_client


def test_get_deploy_client_no_args():
    assert get_deploy_client() is None


def test_get_deploy_client_none():
    assert get_deploy_client(None) is None


def test_get_deploy_client_from_set_deployments_target():
    from mlflow.deployments import set_deployments_target

    set_deployments_target("databricks")
    assert get_deploy_client(None) is not None


def test_get_deploy_client_from_env():
    import os

    os.environ["MLFLOW_DEPLOYMENT_CLIENT"] = "databricks"
    assert get_deploy_client(None) is not None
