from unittest import mock

import pytest

from mlflow import deployments
from mlflow.deployments.plugin_manager import DeploymentPlugins
from mlflow.exceptions import MlflowException

f_model_uri = "fake_model_uri"
f_endpoint_name = "fake_endpoint_name"
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
    assert ret[0]["name"] == f_deployment_id


def test_get_success():
    client = deployments.get_deploy_client(f_target)
    ret = client.get_deployment(f_deployment_id)
    assert ret["key1"] == "val1"


def test_endpoint_create_success():
    client = deployments.get_deploy_client(f_target)
    endpoint = client.create_endpoint(f_endpoint_name)
    assert isinstance(endpoint, dict)
    assert endpoint["name"] == f_endpoint_name


def test_endpoint_delete_success():
    client = deployments.get_deploy_client(f_target)
    assert client.delete_endpoint(f_endpoint_name) is None


def test_endpoint_update_success():
    client = deployments.get_deploy_client(f_target)
    assert client.update_endpoint(f_endpoint_name) is None


def test_endpoint_list_success():
    client = deployments.get_deploy_client(f_target)
    endpoints = client.list_endpoints()
    assert endpoints[0]["name"] == f_endpoint_name


def test_endpoint_get_success():
    client = deployments.get_deploy_client(f_target)
    endpoint = client.get_endpoint(f_endpoint_name)
    assert endpoint["name"] == f_endpoint_name


def test_wrong_target_name():
    with pytest.raises(
        MlflowException, match='No plugin found for managing model deployments to "wrong_target"'
    ):
        deployments.get_deploy_client("wrong_target")


def test_plugin_doesnot_have_required_attrib():
    class DummyPlugin:
        pass

    dummy_plugin = DummyPlugin()
    plugin_manager = DeploymentPlugins()
    plugin_manager.registry["dummy"] = dummy_plugin
    with pytest.raises(MlflowException, match="Plugin registered for the target dummy"):
        plugin_manager["dummy"]


def test_plugin_raising_error(monkeypatch):
    client = deployments.get_deploy_client(f_target)
    # special case to raise error
    monkeypatch.setenv("raiseError", "True")
    with pytest.raises(RuntimeError, match="Error requested"):
        client.list_deployments()


def test_target_uri_parsing():
    deployments.get_deploy_client(f_target)
    deployments.get_deploy_client(f"{f_target}:/somesuffix")
    deployments.get_deploy_client(f"{f_target}://somesuffix")


def test_explain_with_no_target_implementation():
    from mlflow_test_plugin import fake_deployment_plugin

    mock_error = MlflowException("MOCK ERROR")
    target_client = deployments.get_deploy_client(f_target)
    plugin = fake_deployment_plugin.PluginDeploymentClient
    with mock.patch.object(plugin, "explain", return_value=mock_error) as mock_explain:
        res = target_client.explain(f_target, "test")
        assert type(res) == MlflowException
        mock_explain.assert_called_once()


def test_explain_with_target_implementation():
    target_client = deployments.get_deploy_client(f_target)
    res = target_client.explain(f_target, "test")
    assert res == "1"
