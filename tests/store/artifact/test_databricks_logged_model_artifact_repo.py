from pathlib import Path
from unittest import mock

import pytest
from databricks.sdk.service.files import DirectoryEntry

from mlflow.entities.file_info import FileInfo
from mlflow.store.artifact.databricks_logged_model_artifact_repo import (
    DatabricksLoggedModelArtifactRepository,
)


@pytest.fixture(autouse=True)
def set_fake_databricks_creds(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATABRICKS_HOST", "https://localhost:8080")
    monkeypatch.setenv("DATABRICKS_TOKEN", "token")


def test_log_artifact(tmp_path: Path):
    local_file = tmp_path / "local_file.txt"
    local_file.write_text("test content")
    mock_databricks_artifact_repo = mock.MagicMock()
    with (
        mock.patch(
            "mlflow.store.artifact.databricks_sdk_artifact_repo.DatabricksSdkArtifactRepository.files_api"
        ) as mock_files_api,
        mock.patch(
            "mlflow.store.artifact.databricks_logged_model_artifact_repo.DatabricksArtifactRepository",
            return_value=mock_databricks_artifact_repo,
        ),
    ):
        repo = DatabricksLoggedModelArtifactRepository(
            "dbfs:/databricks/mlflow-tracking/1/logged_models/1"
        )

        # Simulate success
        repo.log_artifact(str(local_file), "artifact_path")
        mock_files_api.upload.assert_called_once()

        # Simulate failure and fallback
        mock_files_api.upload.side_effect = RuntimeError("Upload failed")
        with pytest.raises(RuntimeError, match=r"^Upload failed$"):
            repo.databricks_sdk_repo.log_artifact(str(local_file), "artifact_path")

        repo.log_artifact(str(local_file), "artifact_path")
        mock_databricks_artifact_repo.log_artifact.assert_called_once()


def test_log_artifacts(tmp_path: Path):
    local_dir = tmp_path / "local_dir"
    local_dir.mkdir()
    (local_dir / "file1.txt").write_text("content1")
    (local_dir / "file2.txt").write_text("content2")
    mock_databricks_artifact_repo = mock.MagicMock()
    with (
        mock.patch(
            "mlflow.store.artifact.databricks_sdk_artifact_repo.DatabricksSdkArtifactRepository.files_api"
        ) as mock_files_api,
        mock.patch(
            "mlflow.store.artifact.databricks_logged_model_artifact_repo.DatabricksArtifactRepository",
            return_value=mock_databricks_artifact_repo,
        ),
    ):
        repo = DatabricksLoggedModelArtifactRepository(
            "dbfs:/databricks/mlflow-tracking/1/logged_models/1"
        )

        # Simulate success
        repo.log_artifacts(str(local_dir), "artifact_path")
        mock_files_api.upload.assert_called()

        # Simulate failure and fallback
        mock_files_api.upload.side_effect = RuntimeError("Upload failed")
        with pytest.raises(RuntimeError, match=r"^Upload failed$"):
            repo.databricks_sdk_repo.log_artifacts(str(local_dir), "artifact_path")

        mock_databricks_artifact_repo.log_artifact.side_effect = RuntimeError("Fallback failed")
        with pytest.raises(RuntimeError, match=r"^Fallback failed$"):
            repo.databricks_artifact_repo.log_artifact("test", "artifact_path")

        repo.log_artifacts(str(local_dir), "artifact_path")
        mock_databricks_artifact_repo.log_artifacts.assert_called_once()


def test_download_file(tmp_path: Path):
    local_file = tmp_path / "downloaded_file.txt"
    mock_databricks_artifact_repo = mock.MagicMock()
    with (
        mock.patch(
            "mlflow.store.artifact.databricks_sdk_artifact_repo.DatabricksSdkArtifactRepository.files_api"
        ) as mock_files_api,
        mock.patch(
            "mlflow.store.artifact.databricks_logged_model_artifact_repo.DatabricksArtifactRepository"
        ) as mock_fallback_repo,
    ):
        mock_fallback_repo.return_value = mock_databricks_artifact_repo
        repo = DatabricksLoggedModelArtifactRepository(
            "dbfs:/databricks/mlflow-tracking/1/logged_models/1"
        )

        # Simulate success
        mock_files_api.download.return_value.contents.read.side_effect = [b"test", b""]
        repo._download_file("remote_file_path", str(local_file))
        mock_files_api.download.assert_called_once()
        mock_databricks_artifact_repo._download_file.assert_not_called()

        # Simulate failure and fallback
        mock_files_api.download.side_effect = RuntimeError("Download failed")
        with pytest.raises(RuntimeError, match=r"^Download failed$"):
            repo.databricks_sdk_repo._download_file("remote_file_path", str(local_file))

        repo._download_file("remote_file_path", str(local_file))
        mock_databricks_artifact_repo._download_file.assert_called_once()


def test_list_artifacts():
    mock_databricks_artifact_repo = mock.MagicMock()
    with (
        mock.patch(
            "mlflow.store.artifact.databricks_sdk_artifact_repo.DatabricksSdkArtifactRepository.files_api"
        ) as mock_files_api,
        mock.patch(
            "mlflow.store.artifact.databricks_logged_model_artifact_repo.DatabricksArtifactRepository",
            return_value=mock_databricks_artifact_repo,
        ),
    ):
        repo = DatabricksLoggedModelArtifactRepository(
            "dbfs:/databricks/mlflow-tracking/1/logged_models/1"
        )

        # Simulate success with a non-empty list
        mock_files_api.list_directory_contents.return_value = [
            DirectoryEntry(path="artifact1", is_directory=False, file_size=123),
            DirectoryEntry(path="dir", is_directory=True),
        ]
        artifacts = repo.list_artifacts("artifact_path")
        mock_files_api.list_directory_contents.assert_called_once()
        assert len(artifacts) == 2

        # Simulate failure and fallback
        mock_files_api.list_directory_contents.side_effect = RuntimeError("List failed")
        with pytest.raises(RuntimeError, match=r"^List failed$"):
            repo.databricks_sdk_repo.list_artifacts("artifact_path")

        mock_databricks_artifact_repo.list_artifacts.return_value = [
            FileInfo(path="fallback_artifact", is_dir=False, file_size=456)
        ]
        artifacts = repo.list_artifacts("artifact_path")
        mock_databricks_artifact_repo.list_artifacts.assert_called_once()
        assert len(artifacts) == 1
