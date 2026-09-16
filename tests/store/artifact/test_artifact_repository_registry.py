from importlib import reload
from unittest import mock

import pytest

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.store.artifact import artifact_repository_registry
from mlflow.store.artifact.artifact_repository_registry import (
    ArtifactRepositoryRegistry,
    get_artifact_repository,
)
from mlflow.store.artifact.ftp_artifact_repo import FTPArtifactRepository
from mlflow.store.artifact.host_policy import _SERVER_ARTIFACT_ROOT_ENV_VAR


def test_standard_artifact_registry():
    mock_entrypoint = mock.Mock()
    mock_entrypoint.name = "mock-scheme"

    with mock.patch("mlflow.utils.plugins._get_entry_points", return_value=[mock_entrypoint]):
        # Entrypoints are registered at import time, so we need to reload the
        # module to register the entrypoint given by the mocked
        # entrypoints.get_group_all
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


def test_plugin_registration_via_installed_package():
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

    mock_plugin.assert_called_once_with(
        "mock-scheme://fake-host/fake-path", tracking_uri=None, registry_uri=None
    )


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
        "mlflow.utils.plugins._get_entry_points", return_value=[mock_entrypoint]
    ) as mock_get_group_all:
        artifact_repository_registry = ArtifactRepositoryRegistry()
        artifact_repository_registry.register_entrypoints()

    assert (
        artifact_repository_registry.get_artifact_repository("mock-scheme://fake-host/fake-path")
        == mock_plugin_function.return_value
    )

    mock_plugin_function.assert_called_once_with(
        "mock-scheme://fake-host/fake-path", tracking_uri=None, registry_uri=None
    )
    mock_get_group_all.assert_called_once_with("mlflow.artifact_repository")


@pytest.mark.parametrize(
    "exception", [AttributeError("test exception"), ImportError("test exception")]
)
def test_plugin_registration_failure_via_entrypoints(exception):
    mock_entrypoint = mock.Mock(load=mock.Mock(side_effect=exception))
    mock_entrypoint.name = "mock-scheme"

    with mock.patch(
        "mlflow.utils.plugins._get_entry_points", return_value=[mock_entrypoint]
    ) as mock_get_group_all:
        repo_registry = ArtifactRepositoryRegistry()

        # Check that the raised warning contains the message from the original exception
        with pytest.warns(UserWarning, match="test exception"):
            repo_registry.register_entrypoints()

    mock_entrypoint.load.assert_called_once()
    mock_get_group_all.assert_called_once_with("mlflow.artifact_repository")


def test_server_process_refuses_foreign_host_artifact_uri(monkeypatch):
    monkeypatch.setenv(_SERVER_ARTIFACT_ROOT_ENV_VAR, "ftp://ftp-host:21/pub")

    with pytest.raises(MlflowException, match="does not connect to artifact location"):
        get_artifact_repository("ftp://other-host/pub/run/artifacts")

    repo = get_artifact_repository("ftp://ftp-host/pub/run/artifacts")
    assert isinstance(repo, FTPArtifactRepository)


def test_server_process_refuses_foreign_host_behind_runs_uri(monkeypatch):
    monkeypatch.setenv(_SERVER_ARTIFACT_ROOT_ENV_VAR, "./mlruns")

    with (
        mock.patch(
            "mlflow.store.artifact.runs_artifact_repo.RunsArtifactRepository.get_underlying_uri",
            return_value="sftp://other-host/data/run/artifacts/model",
        ),
        pytest.raises(MlflowException, match="does not connect to artifact location"),
    ):
        get_artifact_repository("runs:/run/model")


def test_client_process_builds_repositories_for_any_host(monkeypatch):
    monkeypatch.delenv(_SERVER_ARTIFACT_ROOT_ENV_VAR, raising=False)
    repo = get_artifact_repository("ftp://other-host/pub/run/artifacts")
    assert isinstance(repo, FTPArtifactRepository)
