from mlflow.deployments.plugin_manager import DeploymentPlugins
from mlflow.deployments import BasePlugin
import pytest


f_model_uri = 'fake_model_uri'
f_deployment_id = 'fake_deployment_id'
f_flavor = 'fake_flavor'


class FakePlugin(BasePlugin):
    def create(self, model_uri, flavor=None, **kwargs):
        return {'deployment_id': f_deployment_id, 'flavor': flavor}

    def delete(self, deployment_id, **kwargs):
        return None

    def update(self, deployment_id, model_uri=None, flavor=False, **kwargs):
        return {'flavor': flavor}

    def list(self, **kwargs):
        if kwargs.get('raiseError'):
            raise RuntimeError('Error requested')
        return [f_deployment_id]

    def get(self, deployment_id, **kwargs):
        return {'key1': 'val1', 'key2': 'val2'}


def custom_getitem_for_plugin_store(self, name):  # pylint: disable=W0613
    if not self.has_plugins_loaded and self.auto_register:
        self.register_entrypoints()
    return FakePlugin()


def custom_register_entrypoints_for_plugin_store(fn):
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, 'entrypoint_calling_count'):
            self.entrypoint_calling_count = 0
        else:
            self.entrypoint_calling_count += 1
        return fn(self, *args, **kwargs)
    return wrapper


@pytest.fixture()
def patched_plugin_store(monkeypatch):
    monkeypatch.setattr(DeploymentPlugins, '__getitem__', custom_getitem_for_plugin_store)
    monkeypatch.setattr(DeploymentPlugins, 'register_entrypoints',
                        custom_register_entrypoints_for_plugin_store(
                            DeploymentPlugins.register_entrypoints))
    yield None
