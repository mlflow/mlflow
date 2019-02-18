import mock
from six.moves import reload_module as reload

import mlflow
from mlflow.store.artifact_repository_registry import ArtifactRepositoryRegistry


def test_standard_artifact_registry():
    mock_entrypoint = mock.Mock()
    mock_entrypoint.name = "mock-scheme"

    with mock.patch(
        "entrypoints.get_group_all", return_value=[mock_entrypoint]
    ):
        # Entrypoints are registered at import time, so we need to reload the
        # module to register the entrypoint given by the mocked
        # extrypoints.get_group_all
        reload(mlflow.store.artifact_repository_registry)

        expected_artifact_repository_registry = {
            '',
            's3',
            'gs',
            'wasbs',
            'ftp',
            'sftp',
            'dbfs',
            'mock-scheme'
        }

    assert expected_artifact_repository_registry.issubset(
        mlflow.store.artifact_repository_registry._artifact_repository_registry._registry.keys()
    )


@pytest.mark.large
def test_plugin_registration_via_installed_package():
    """This test requires the package in tests/resources/mlflow-test-plugin to be installed"""

    reload(mlflow.store.artifact_repository_registry)

    assert (
        "file-plugin" in
        mlflow.store.artifact_repository_registry._artifact_repository_registry._registry
    )

    from mlflow_test_plugin import PluginLocalArtifactRepository

    test_uri = "file-plugin:test-path"

    plugin_repo = mlflow.store.artifact_repository_registry.get_artifact_repository(test_uri)

    assert isinstance(plugin_repo, PluginLocalArtifactRepository)
    assert plugin_repo.artifact_uri == test_uri


def test_plugin_registration():
    artifact_repository_registry = ArtifactRepositoryRegistry()

    mock_plugin = mock.Mock()
    artifact_repository_registry.register("mock-scheme", mock_plugin)
    assert "mock-scheme" in artifact_repository_registry._registry
    repository_instance = artifact_repository_registry.get_artifact_repository(
        artifact_uri="mock-scheme://fake-host/fake-path"
    )
    assert repository_instance == mock_plugin.return_value


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
