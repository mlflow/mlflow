from importlib import reload
from unittest import mock
import pytest

import mlflow.tracking.context.registry
from mlflow.tracking.context.default_context import DefaultRunContext
from mlflow.tracking.context.git_context import GitRunContext
from mlflow.tracking.context.databricks_notebook_context import DatabricksNotebookRunContext
from mlflow.tracking.context.databricks_job_context import DatabricksJobRunContext
from mlflow.tracking.context.registry import RunContextProviderRegistry, resolve_tags

# pylint: disable=unused-argument


def test_run_context_provider_registry_register():
    provider_class = mock.Mock()

    registry = RunContextProviderRegistry()
    registry.register(provider_class)

    assert set(registry) == {provider_class.return_value}


def test_run_context_provider_registry_register_entrypoints():
    provider_class = mock.Mock()
    mock_entrypoint = mock.Mock()
    mock_entrypoint.load.return_value = provider_class

    with mock.patch(
        "entrypoints.get_group_all", return_value=[mock_entrypoint]
    ) as mock_get_group_all:
        registry = RunContextProviderRegistry()
        registry.register_entrypoints()

    assert set(registry) == {provider_class.return_value}
    mock_entrypoint.load.assert_called_once_with()
    mock_get_group_all.assert_called_once_with("mlflow.run_context_provider")


@pytest.mark.parametrize(
    "exception", [AttributeError("test exception"), ImportError("test exception")]
)
def test_run_context_provider_registry_register_entrypoints_handles_exception(exception):
    mock_entrypoint = mock.Mock()
    mock_entrypoint.load.side_effect = exception

    with mock.patch(
        "entrypoints.get_group_all", return_value=[mock_entrypoint]
    ) as mock_get_group_all:
        registry = RunContextProviderRegistry()
        # Check that the raised warning contains the message from the original exception
        with pytest.warns(UserWarning, match="test exception"):
            registry.register_entrypoints()

    mock_entrypoint.load.assert_called_once_with()
    mock_get_group_all.assert_called_once_with("mlflow.run_context_provider")


def _currently_registered_run_context_provider_classes():
    return {
        provider.__class__
        for provider in mlflow.tracking.context.registry._run_context_provider_registry
    }


def test_registry_instance_defaults():
    expected_classes = {
        DefaultRunContext,
        GitRunContext,
        DatabricksNotebookRunContext,
        DatabricksJobRunContext,
    }
    assert expected_classes.issubset(_currently_registered_run_context_provider_classes())


def test_registry_instance_loads_entrypoints():
    class MockRunContext:
        pass

    mock_entrypoint = mock.Mock()
    mock_entrypoint.load.return_value = MockRunContext

    with mock.patch(
        "entrypoints.get_group_all", return_value=[mock_entrypoint]
    ) as mock_get_group_all:
        # Entrypoints are registered at import time, so we need to reload the module to register the
        # entrypoint given by the mocked extrypoints.get_group_all
        reload(mlflow.tracking.context.registry)

    assert MockRunContext in _currently_registered_run_context_provider_classes()
    mock_get_group_all.assert_called_once_with("mlflow.run_context_provider")


@pytest.mark.large
def test_run_context_provider_registry_with_installed_plugin(tmp_wkdir):
    """This test requires the package in tests/resources/mlflow-test-plugin to be installed"""

    reload(mlflow.tracking.context.registry)

    from mlflow_test_plugin.run_context_provider import PluginRunContextProvider

    assert PluginRunContextProvider in _currently_registered_run_context_provider_classes()

    # The test plugin's context provider always returns False from in_context
    # to avoid polluting tags in developers' environments. The following mock overrides this to
    # perform the integration test.
    with mock.patch.object(PluginRunContextProvider, "in_context", return_value=True):
        assert resolve_tags()["test"] == "tag"


@pytest.fixture
def mock_run_context_providers():
    base_provider = mock.Mock()
    base_provider.in_context.return_value = True
    base_provider.tags.return_value = {"one": "one-val", "two": "two-val", "three": "three-val"}

    skipped_provider = mock.Mock()
    skipped_provider.in_context.return_value = False

    exception_provider = mock.Mock()
    exception_provider.in_context.return_value = True
    exception_provider.tags.return_value = {
        "random-key": "This val will never make it to tag resolution"
    }
    exception_provider.tags.side_effect = Exception(
        "This should be caught by logic in resolve_tags()"
    )

    override_provider = mock.Mock()
    override_provider.in_context.return_value = True
    override_provider.tags.return_value = {"one": "override", "new": "new-val"}

    providers = [base_provider, skipped_provider, exception_provider, override_provider]

    with mock.patch("mlflow.tracking.context.registry._run_context_provider_registry", providers):
        yield

    skipped_provider.tags.assert_not_called()


def test_resolve_tags(mock_run_context_providers):
    tags_arg = {"two": "arg-override", "arg": "arg-val"}
    assert resolve_tags(tags_arg) == {
        "one": "override",
        "two": "arg-override",
        "three": "three-val",
        "new": "new-val",
        "arg": "arg-val",
    }


def test_resolve_tags_no_arg(mock_run_context_providers):
    assert resolve_tags() == {
        "one": "override",
        "two": "two-val",
        "three": "three-val",
        "new": "new-val",
    }
