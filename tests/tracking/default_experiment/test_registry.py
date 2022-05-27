from importlib import reload
from unittest import mock
import pytest

import mlflow.tracking.default_experiment.registry
from mlflow.tracking.default_experiment.databricks_notebook_experiment_provider import (
    DatabricksNotebookExperimentProvider,
)
from mlflow.tracking.default_experiment.databricks_job_experiment_provider import (
    DatabricksJobExperimentProvider,
)
from mlflow.tracking.default_experiment.registry import (
    DefaultExperimentProviderRegistry,
    get_experiment_id,
)

# pylint: disable=unused-argument


def test_default_experiment_provider_registry_register():
    provider_class = mock.Mock()

    registry = DefaultExperimentProviderRegistry()
    registry.register(provider_class)

    assert set(registry) == {provider_class.return_value}


def test_default_experiment_provider_registry_register_entrypoints():
    provider_class = mock.Mock()
    mock_entrypoint = mock.Mock()
    mock_entrypoint.load.return_value = provider_class

    with mock.patch(
        "entrypoints.get_group_all", return_value=[mock_entrypoint]
    ) as mock_get_group_all:
        registry = DefaultExperimentProviderRegistry()
        registry.register_entrypoints()

    assert set(registry) == {provider_class.return_value}
    mock_entrypoint.load.assert_called_once_with()
    mock_get_group_all.assert_called_once_with("mlflow.default_experiment_provider")


@pytest.mark.parametrize(
    "exception", [AttributeError("test exception"), ImportError("test exception")]
)
def test_default_experiment_provider_registry_register_entrypoints_handles_exception(exception):
    mock_entrypoint = mock.Mock()
    mock_entrypoint.load.side_effect = exception

    with mock.patch(
        "entrypoints.get_group_all", return_value=[mock_entrypoint]
    ) as mock_get_group_all:
        registry = DefaultExperimentProviderRegistry()
        # Check that the raised warning contains the message from the original exception
        with pytest.warns(UserWarning, match="test exception"):
            registry.register_entrypoints()

    mock_entrypoint.load.assert_called_once_with()
    mock_get_group_all.assert_called_once_with("mlflow.default_experiment_provider")


def _currently_registered_default_experiment_provider_classes():
    return {
        provider.__class__
        for provider in mlflow.tracking.default_experiment.registry._default_experiment_provider_registry  # pylint: disable=line-too-long
    }


def test_registry_instance_defaults():
    expected_classes = {
        DatabricksNotebookExperimentProvider,
        DatabricksJobExperimentProvider,
    }
    assert expected_classes.issubset(_currently_registered_default_experiment_provider_classes())


def test_registry_instance_loads_entrypoints():
    class MockRunContext:
        pass

    mock_entrypoint = mock.Mock()
    mock_entrypoint.load.return_value = MockRunContext

    with mock.patch(
        "entrypoints.get_group_all", return_value=[mock_entrypoint]
    ) as mock_get_group_all:
        # Entrypoints are registered at import time, so we need to reload the module to register the
        # entrypoint given by the mocked entrypoints.get_group_all
        reload(mlflow.tracking.default_experiment.registry)

    assert MockRunContext in _currently_registered_default_experiment_provider_classes()
    mock_get_group_all.assert_called_once_with("mlflow.default_experiment_provider")


def test_default_experiment_provider_registry_with_installed_plugin(tmp_wkdir):
    """This test requires the package in tests/resources/mlflow-test-plugin to be installed"""

    reload(mlflow.tracking.default_experiment.registry)

    from mlflow_test_plugin.default_experiment_provider import PluginDefaultExperimentProvider

    assert (
        PluginDefaultExperimentProvider
        in _currently_registered_default_experiment_provider_classes()
    )

    # The test plugin's context provider always returns False from in_context
    # to avoid polluting get_experiment_id in developers' environments.
    # The following mock overrides this to perform the integration test.
    with mock.patch.object(PluginDefaultExperimentProvider, "in_context", return_value=True):
        assert get_experiment_id() == "experiment_id_1"


@pytest.fixture
def mock_default_experiment_providers():
    base_provider = mock.Mock()
    base_provider.in_context.return_value = True
    base_provider.get_experiment_id.return_value = "experiment_id_1"

    skipped_provider = mock.Mock()
    skipped_provider.in_context.return_value = False

    exception_provider = mock.Mock()
    exception_provider.in_context.return_value = True
    exception_provider.get_experiment_id.side_effect = Exception(
        "This should be caught by logic in get_experiment_id()"
    )

    providers = [base_provider, skipped_provider, exception_provider]

    with mock.patch(
        "mlflow.tracking.default_experiment.registry._default_experiment_provider_registry",
        providers,
    ):
        yield

    skipped_provider.get_experiment_id.assert_not_called()


@pytest.fixture
def mock_default_experiment_multiple_context_providers():
    base_provider = mock.Mock()
    base_provider.in_context.return_value = True
    base_provider.get_experiment_id.return_value = "experiment_id_1"

    unused_provider = mock.Mock()
    unused_provider.in_context.return_value = True
    unused_provider.get_experiment_id.return_value = "experiment_id_2"

    providers = [base_provider, unused_provider]

    with mock.patch(
        "mlflow.tracking.default_experiment.registry._default_experiment_provider_registry",
        providers,
    ):
        yield

    unused_provider.get_experiment_id.assert_not_called()


def test_get_experiment_id(mock_default_experiment_providers):
    assert get_experiment_id() == "experiment_id_1"


def test_get_experiment_id_multiple_context(mock_default_experiment_multiple_context_providers):
    assert get_experiment_id() == "experiment_id_1"
