import pytest
import entrypoints
from . import test_plugin


f_target = 'fake_target'


def custom_entrypoint_group_all(name):
    if name == 'mlflow.deployments':
        # This entrypoint itself is fake so we monkeypatch the ``load`` to handle that
        return [entrypoints.EntryPoint(f_target, 'test_plugin', None)]


def custom_entrypoint_load(self):
    if self.name == f_target:
        return test_plugin


@pytest.fixture()
def patched_plugin_store(monkeypatch):
    monkeypatch.setattr(entrypoints, 'get_group_all', custom_entrypoint_group_all)
    monkeypatch.setattr(entrypoints.EntryPoint, 'load', custom_entrypoint_load)
    yield None
