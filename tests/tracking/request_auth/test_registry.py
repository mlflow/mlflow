from importlib import reload
from unittest import mock

import pytest

import mlflow.tracking.request_auth.registry
from mlflow.tracking.request_auth.registry import RequestAuthProviderRegistry, fetch_auth


def test_request_auth_provider_registry_register():
    provider_class = mock.Mock()

    registry = RequestAuthProviderRegistry()
    registry.register(provider_class)

    assert set(registry) == {provider_class.return_value}


def test_request_auth_provider_registry_register_entrypoints():
    provider_class = mock.Mock()
    mock_entrypoint = mock.Mock()
    mock_entrypoint.load.return_value = provider_class

    with mock.patch(
        "mlflow.utils.plugins._get_entry_points", return_value=[mock_entrypoint]
    ) as mock_get_group_all:
        registry = RequestAuthProviderRegistry()
        registry.register_entrypoints()

    assert set(registry) == {provider_class.return_value}
    mock_entrypoint.load.assert_called_once_with()
    mock_get_group_all.assert_called_once_with("mlflow.request_auth_provider")


@pytest.mark.parametrize(
    "exception", [AttributeError("test exception"), ImportError("test exception")]
)
def test_request_auth_provider_registry_register_entrypoints_handles_exception(exception):
    mock_entrypoint = mock.Mock()
    mock_entrypoint.load.side_effect = exception

    with mock.patch(
        "mlflow.utils.plugins._get_entry_points", return_value=[mock_entrypoint]
    ) as mock_get_group_all:
        registry = RequestAuthProviderRegistry()
        # Check that the raised warning contains the message from the original exception
        with pytest.warns(UserWarning, match="test exception"):
            registry.register_entrypoints()

    mock_entrypoint.load.assert_called_once_with()
    mock_get_group_all.assert_called_once_with("mlflow.request_auth_provider")


def test_registry_instance_loads_entrypoints():
    class MockRequestAuthProvider:
        pass

    mock_entrypoint = mock.Mock()
    mock_entrypoint.load.return_value = MockRequestAuthProvider

    with mock.patch(
        "mlflow.utils.plugins._get_entry_points", return_value=[mock_entrypoint]
    ) as mock_get_group_all:
        # Entrypoints are registered at import time, so we need to reload the module to register the
        # entrypoint given by the mocked entrypoints.get_group_all
        reload(mlflow.tracking.request_auth.registry)

    assert MockRequestAuthProvider in _currently_registered_request_auth_provider_classes()
    mock_get_group_all.assert_called_once_with("mlflow.request_auth_provider")


def _currently_registered_request_auth_provider_classes():
    return {
        provider.__class__
        for provider in mlflow.tracking.request_auth.registry._request_auth_provider_registry
    }


def test_run_context_provider_registry_with_installed_plugin():
    """This test requires the package in tests/resources/mlflow-test-plugin to be installed"""

    reload(mlflow.tracking.request_auth.registry)

    from mlflow_test_plugin.request_auth_provider import PluginRequestAuthProvider

    assert PluginRequestAuthProvider in _currently_registered_request_auth_provider_classes()

    auth_provider_name = "test_auth_provider_name"
    assert fetch_auth(auth_provider_name)["auth_name"] == "test_auth_provider_name"


def test_fetch_auth():
    reload(mlflow.tracking.request_auth.registry)
    auth_provider_name = "test_auth_provider_name"
    assert fetch_auth(auth_provider_name)["auth_name"] == auth_provider_name
