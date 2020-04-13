import pytest
from mlflow import deployments
from mlflow.deployments.plugin_manager import DeploymentPlugins
from mlflow.exceptions import MlflowException


f_model_uri = 'fake_model_uri'
f_deployment_id = 'fake_deployment_id'
f_flavor = 'fake_flavor'
f_target = 'fake_target'


def test_create_success(patched_plugin_store):  # pylint: disable=W0613
    ret = deployments.create_deployment(f_target, f_model_uri, f_flavor)
    assert isinstance(ret, dict)
    assert ret['deployment_id'] == f_deployment_id
    assert ret['flavor'] == f_flavor

    ret2 = deployments.create_deployment(f_target, f_model_uri)
    assert ret['deployment_id'] == ret2['deployment_id']
    assert ret2['flavor'] is None


def test_delete_success(patched_plugin_store):  # pylint: disable=W0613
    assert deployments.delete_deployment(f_target, f_deployment_id) is None


def test_update_success(patched_plugin_store):  # pylint: disable=W0613
    res = deployments.update_deployment(f_target, f_deployment_id, f_model_uri, f_flavor)
    assert res['flavor'] == f_flavor


def test_update_flavor_without_model_uri():
    with pytest.raises(RuntimeError):
        deployments.update_deployment(f_target, f_deployment_id, flavor='pytorch')


def test_update_without_any_arguments():
    with pytest.raises(RuntimeError):
        deployments.update_deployment(f_target, f_deployment_id)


def test_missing_arguments(patched_plugin_store):  # pylint: disable=W0613
    with pytest.raises(TypeError):
        deployments.create_deployment(f_target)  # pylint: disable=no-value-for-parameter


def test_list_success(patched_plugin_store):  # pylint: disable=W0613
    ret = deployments.list_deployments(f_target)
    assert ret[0] == f_deployment_id


def test_get_success(patched_plugin_store):  # pylint: disable=W0613
    ret = deployments.get_deployment(f_target, f_deployment_id)
    assert ret['key1'] == 'val1'


def test_wrong_target_name():
    with pytest.raises(MlflowException):
        deployments.create_deployment('wrong_target', f_model_uri, f_flavor)


def test_plugin_not_inherited_from_BasePlugin():
    class DummyPlugin:
        ...  # pylint: disable=W0104

    dummy_plugin = DummyPlugin()
    plugin_manager = DeploymentPlugins()
    plugin_manager.register('dummy',  dummy_plugin)
    with pytest.raises(MlflowException):
        plugin_manager.register_entrypoints()


def test_entrypoints_not_reloading(patched_plugin_store):  # pylint: disable=W0613
    from mlflow.deployments.interface import plugin_store
    deployments.list_deployments(f_target)  # triggering the reload of entrypoints
    assert plugin_store.entrypoint_calling_count == 0  # count is set to 0
    deployments.list_deployments(f_target)  # triggering the reload
    assert plugin_store.entrypoint_calling_count == 0  # Guard autoset
    plugin_store._has_plugins_loaded = False  # Turning the guard again off
    deployments.list_deployments(f_target)  # triggering the reload
    assert plugin_store.entrypoint_calling_count == 1  # count increased


def test_plugin_raising_error(patched_plugin_store):  # pylint: disable=W0613
    with pytest.raises(RuntimeError):
        deployments.list_deployments(f_target, raiseError='True')  # special case to raise error
