import pytest
import os
from mlflow import deployments
from mlflow.deployments.plugin_manager import DeploymentPlugins
from mlflow.exceptions import MlflowException


f_model_uri = "fake_model_uri"
f_deployment_id = "fake_deployment_name"
f_flavor = "fake_flavor"
f_target = "faketarget"


def test_create_success():
    client = deployments.get_deploy_client(f_target)
    ret = client.create_deployment(f_deployment_id, f_model_uri, f_flavor, config={})
    assert isinstance(ret, dict)
    assert ret["name"] == f_deployment_id
    assert ret["flavor"] == f_flavor

    ret2 = client.create_deployment(f_deployment_id, f_model_uri)
    assert ret2["flavor"] is None


def test_delete_success():
    client = deployments.get_deploy_client(f_target)
    assert client.delete_deployment(f_deployment_id) is None


def test_update_success():
    client = deployments.get_deploy_client(f_target)
    res = client.update_deployment(f_deployment_id, f_model_uri, f_flavor)
    assert res["flavor"] == f_flavor


def test_list_success():
    client = deployments.get_deploy_client(f_target)
    ret = client.list_deployments()
    assert ret[0] == f_deployment_id


def test_get_success():
    client = deployments.get_deploy_client(f_target)
    ret = client.get_deployment(f_deployment_id)
    assert ret["key1"] == "val1"


def test_wrong_target_name():
    with pytest.raises(MlflowException):
        deployments.get_deploy_client("wrong_target")


def test_plugin_doesnot_have_required_attrib():
    class DummyPlugin:
        ...  # pylint: disable=pointless-statement

    dummy_plugin = DummyPlugin()
    plugin_manager = DeploymentPlugins()
    plugin_manager.registry["dummy"] = dummy_plugin
    with pytest.raises(MlflowException):
        plugin_manager["dummy"]  # pylint: disable=pointless-statement


def test_plugin_raising_error():
    client = deployments.get_deploy_client(f_target)
    # special case to raise error
    os.environ["raiseError"] = "True"
    with pytest.raises(RuntimeError):
        client.list_deployments()
    os.environ["raiseError"] = "False"


def test_target_uri_parsing():
    deployments.get_deploy_client(f_target)
    deployments.get_deploy_client("{target}:/somesuffix".format(target=f_target))
    with pytest.raises(MlflowException):
        deployments.get_deploy_client("{target}://somesuffix".format(target=f_target))
