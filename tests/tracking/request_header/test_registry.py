import pytest
from unittest import mock
from importlib import reload

import mlflow.tracking.request_header.registry
from mlflow.tracking.request_header.registry import (
    RequestHeaderProviderRegistry,
    resolve_request_headers,
)
from mlflow.tracking.request_header.databricks_request_header_provider import (
    DatabricksRequestHeaderProvider,
)

# pylint: disable=unused-argument


def test_request_header_context_provider_registry_register():
    provider_class = mock.Mock()

    registry = RequestHeaderProviderRegistry()
    registry.register(provider_class)

    assert set(registry) == {provider_class.return_value}


def test_request_header_provider_registry_register_entrypoints():
    provider_class = mock.Mock()
    mock_entrypoint = mock.Mock()
    mock_entrypoint.load.return_value = provider_class

    with mock.patch(
        "entrypoints.get_group_all", return_value=[mock_entrypoint]
    ) as mock_get_group_all:
        registry = RequestHeaderProviderRegistry()
        registry.register_entrypoints()

    assert set(registry) == {provider_class.return_value}
    mock_entrypoint.load.assert_called_once_with()
    mock_get_group_all.assert_called_once_with("mlflow.request_header_provider")


@pytest.mark.parametrize(
    "exception", [AttributeError("test exception"), ImportError("test exception")]
)
def test_request_header_provider_registry_register_entrypoints_handles_exception(exception):
    mock_entrypoint = mock.Mock()
    mock_entrypoint.load.side_effect = exception

    with mock.patch(
        "entrypoints.get_group_all", return_value=[mock_entrypoint]
    ) as mock_get_group_all:
        registry = RequestHeaderProviderRegistry()
        # Check that the raised warning contains the message from the original exception
        with pytest.warns(UserWarning, match="test exception"):
            registry.register_entrypoints()

    mock_entrypoint.load.assert_called_once_with()
    mock_get_group_all.assert_called_once_with("mlflow.request_header_provider")


def _currently_registered_request_header_provider_classes():
    return {
        provider.__class__
        for provider in mlflow.tracking.request_header.registry._request_header_provider_registry
    }


def test_registry_instance_defaults():
    expected_classes = {DatabricksRequestHeaderProvider}
    assert expected_classes.issubset(_currently_registered_request_header_provider_classes())


def test_registry_instance_loads_entrypoints():
    class MockRequestHeaderProvider:
        pass

    mock_entrypoint = mock.Mock()
    mock_entrypoint.load.return_value = MockRequestHeaderProvider

    with mock.patch(
        "entrypoints.get_group_all", return_value=[mock_entrypoint]
    ) as mock_get_group_all:
        # Entrypoints are registered at import time, so we need to reload the module to register the
        # entrypoint given by the mocked entrypoints.get_group_all
        reload(mlflow.tracking.request_header.registry)

    assert MockRequestHeaderProvider in _currently_registered_request_header_provider_classes()
    mock_get_group_all.assert_called_once_with("mlflow.request_header_provider")


def test_run_context_provider_registry_with_installed_plugin():
    """This test requires the package in tests/resources/mlflow-test-plugin to be installed"""

    reload(mlflow.tracking.request_header.registry)

    from mlflow_test_plugin.request_header_provider import PluginRequestHeaderProvider

    assert PluginRequestHeaderProvider in _currently_registered_request_header_provider_classes()

    # The test plugin's request header provider always returns False from in_context to avoid
    # polluting request headers in developers' environments. The following mock overrides this to
    # perform the integration test.
    with mock.patch.object(PluginRequestHeaderProvider, "in_context", return_value=True):
        assert resolve_request_headers()["test"] == "header"


@pytest.fixture
def mock_request_header_providers():
    base_provider = mock.Mock()
    base_provider.in_context.return_value = True
    base_provider.request_headers.return_value = {
        "one": "one-val",
        "two": "two-val",
        "three": "three-val",
    }

    skipped_provider = mock.Mock()
    skipped_provider.in_context.return_value = False

    exception_provider = mock.Mock()
    exception_provider.in_context.return_value = True
    exception_provider.request_headers.return_value = {
        "random-header": "This val will never make it to header resolution"
    }
    exception_provider.request_headers.side_effect = Exception(
        "This should be caught by logic in resolve_request_headers()"
    )

    override_provider = mock.Mock()
    override_provider.in_context.return_value = True
    override_provider.request_headers.return_value = {"one": "override", "new": "new-val"}

    providers = [base_provider, skipped_provider, exception_provider, override_provider]

    with mock.patch(
        "mlflow.tracking.request_header.registry._request_header_provider_registry", providers
    ):
        yield

    skipped_provider.tags.assert_not_called()


def test_resolve_request_headers(mock_request_header_providers):
    request_headers_arg = {"two": "arg-override", "arg": "arg-val"}
    assert resolve_request_headers(request_headers_arg) == {
        "one": "one-val override",
        "two": "arg-override",
        "three": "three-val",
        "new": "new-val",
        "arg": "arg-val",
    }


def test_resolve_request_headers_no_arg(mock_request_header_providers):
    assert resolve_request_headers() == {
        "one": "one-val override",
        "two": "two-val",
        "three": "three-val",
        "new": "new-val",
    }
