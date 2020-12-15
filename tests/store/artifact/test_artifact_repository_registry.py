from importlib import reload
import pytest
from unittest import mock

import mlflow
from mlflow.store.artifact import artifact_repository_registry
from mlflow.store.artifact.artifact_repository_registry import ArtifactRepositoryRegistry


def test_standard_artifact_registry():
    mock_entrypoint = mock.Mock()
    mock_entrypoint.name = "mock-scheme"

    with mock.patch("entrypoints.get_group_all", return_value=[mock_entrypoint]):
        # Entrypoints are registered at import time, so we need to reload the
        # module to register the entrypoint given by the mocked
        # extrypoints.get_group_all
        reload(artifact_repository_registry)

        expected_artifact_repository_registry = {
            "",
            "s3",
            "gs",
            "wasbs",
            "ftp",
            "sftp",
            "dbfs",
            "mock-scheme",
        }

    assert expected_artifact_repository_registry.issubset(
        artifact_repository_registry._artifact_repository_registry._registry.keys()
    )


@pytest.mark.large
def test_plugin_registration_via_installed_package():
    """This test requires the package in tests/resources/mlflow-test-plugin to be installed"""

    reload(artifact_repository_registry)

    assert "file-plugin" in artifact_repository_registry._artifact_repository_registry._registry

    from mlflow_test_plugin.local_artifact import PluginLocalArtifactRepository

    test_uri = "file-plugin:test-path"

    plugin_repo = artifact_repository_registry.get_artifact_repository(test_uri)

    assert isinstance(plugin_repo, PluginLocalArtifactRepository)
    assert plugin_repo.is_plugin


def test_plugin_registration():
    artifact_repository_registry = ArtifactRepositoryRegistry()

    mock_plugin = mock.Mock()
    artifact_repository_registry.register("mock-scheme", mock_plugin)
    assert "mock-scheme" in artifact_repository_registry._registry
    repository_instance = artifact_repository_registry.get_artifact_repository(
        artifact_uri="mock-scheme://fake-host/fake-path"
    )
    assert repository_instance == mock_plugin.return_value

    mock_plugin.assert_called_once_with("mock-scheme://fake-host/fake-path")


def test_get_unknown_scheme():
    artifact_repository_registry = ArtifactRepositoryRegistry()

    with pytest.raises(
        mlflow.exceptions.MlflowException, match="Could not find a registered artifact repository"
    ):
        artifact_repository_registry.get_artifact_repository("unknown-scheme://")


def test_plugin_registration_via_entrypoints():
    mock_plugin_function = mock.Mock()
    mock_entrypoint = mock.Mock(load=mock.Mock(return_value=mock_plugin_function))
    mock_entrypoint.name = "mock-scheme"

    with mock.patch(
        "entrypoints.get_group_all", return_value=[mock_entrypoint]
    ) as mock_get_group_all:

        artifact_repository_registry = ArtifactRepositoryRegistry()
        artifact_repository_registry.register_entrypoints()

    assert (
        artifact_repository_registry.get_artifact_repository("mock-scheme://fake-host/fake-path")
        == mock_plugin_function.return_value
    )

    mock_plugin_function.assert_called_once_with("mock-scheme://fake-host/fake-path")
    mock_get_group_all.assert_called_once_with("mlflow.artifact_repository")


@pytest.mark.parametrize(
    "exception", [AttributeError("test exception"), ImportError("test exception")]
)
def test_plugin_registration_failure_via_entrypoints(exception):
    mock_entrypoint = mock.Mock(load=mock.Mock(side_effect=exception))
    mock_entrypoint.name = "mock-scheme"

    with mock.patch(
        "entrypoints.get_group_all", return_value=[mock_entrypoint]
    ) as mock_get_group_all:

        repo_registry = ArtifactRepositoryRegistry()

        # Check that the raised warning contains the message from the original exception
        with pytest.warns(UserWarning, match="test exception"):
            repo_registry.register_entrypoints()

    mock_entrypoint.load.assert_called_once()
    mock_get_group_all.assert_called_once_with("mlflow.artifact_repository")
