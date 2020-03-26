import pytest
from mlflow import deployments
from mlflow.deployments.plugin_manager import DeploymentPlugins
from mlflow.exceptions import MlflowException


f_model_uri = 'fake_model_uri'
f_deployment_id = 'fake_deployment_id'
f_flavor = 'fake_flavor'
f_target = 'fake_target'


def test_create_success(patched_plugin_store):
    ret = deployments.create(f_target, f_model_uri, f_flavor)
    assert isinstance(ret, dict)
    assert ret['deployment_id'] == f_deployment_id
    assert ret['flavor'] == f_flavor

    ret2 = deployments.create(f_target, f_model_uri)
    assert ret['deployment_id'] == ret2['deployment_id']
    assert ret2['flavor'] is None


def test_delete_success(patched_plugin_store):
    ret = deployments.delete(f_target, f_deployment_id)
    assert ret is None


def test_update_success(patched_plugin_store):
    ret = deployments.update(f_target, f_deployment_id, f_model_uri)
    assert ret is None


def test_update_flavor_without_model_uri():
    with pytest.raises(RuntimeError):
        deployments.update(f_target, f_deployment_id, flavor='pytorch')


def test_update_without_any_arguments():
    with pytest.raises(RuntimeError):
        deployments.update(f_target, f_deployment_id)


def test_missing_arguments(patched_plugin_store):
    with pytest.raises(TypeError):
        deployments.create(f_target)


def test_list_success(patched_plugin_store):
    ret = deployments.list(f_target)
    assert ret[0] == f_deployment_id


def test_describe_success(patched_plugin_store):
    ret = deployments.describe(f_target, f_deployment_id)
    assert ret['key1'] == 'val1'


def test_invalid_interface(patched_plugin_store):
    with pytest.raises(AttributeError):
        deployments.wrong_interface(f_target)


def test_wrong_target_name():
    with pytest.raises(MlflowException):
        deployments.create('wrong_target', f_model_uri, f_flavor)


def test_plugin_not_inherited_from_BasePlugin():
    class DummyPlugin:
        ...

    dummy_plugin = DummyPlugin()
    plugin_manager = DeploymentPlugins()
    plugin_manager.register('dummy',  dummy_plugin)
    with pytest.raises(MlflowException):
        plugin_manager.register_entrypoints()


def test_entrypoints_not_reloading(patched_plugin_store):
    from mlflow.deployments.interface import plugin_store
    plugin_store._has_plugins_loaded = False  # Turning the guard off
    deployments.list(f_target)  # triggering the reload of entrypoints
    assert plugin_store.entrypoint_calling_count == 0  # count is set to 0
    deployments.list(f_target)  # triggering the reload
    assert plugin_store.entrypoint_calling_count == 0  # Guard autoset
    plugin_store._has_plugins_loaded = False  # Turning the guard again off
    deployments.list(f_target)  # triggering the reload
    assert plugin_store.entrypoint_calling_count == 1  # count increased


def test_plugin_raising_error(patched_plugin_store):
    with pytest.raises(RuntimeError):
        ret = deployments.list(f_target, raiseError=True)  # special case to raise error
