import subprocess
import sys

import pytest

import mlflow.deployments.utils
from mlflow.deployments.interface import get_deploy_client

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
    monkeypatch.setenv("MLFLOW_DEPLOYMENTS_TARGET", "databricks")


def test_get_deploy_client_from_env(set_deployment_envs):
    assert get_deploy_client(None) is not None


def test_get_deploy_client_after_the_importlib_metadata_backport_is_imported():
    # Importing the backport makes the standard library's `entry_points()` return the backport's
    # `EntryPoint` objects. Run in a subprocess, since the import changes discovery process-wide.
    pytest.importorskip("importlib_metadata")
    code = """
import importlib_metadata

from mlflow.deployments import get_deploy_client
from mlflow.deployments.databricks import DatabricksDeploymentClient

assert isinstance(get_deploy_client("databricks"), DatabricksDeploymentClient)
"""
    subprocess.check_call([sys.executable, "-c", code])
