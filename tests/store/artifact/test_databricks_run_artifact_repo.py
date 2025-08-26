from pathlib import Path
from unittest import mock

import pytest
from databricks.sdk.service.files import DirectoryEntry

from mlflow.entities.file_info import FileInfo
from mlflow.store.artifact.databricks_run_artifact_repo import (
    DatabricksRunArtifactRepository,
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
            "mlflow.store.artifact.databricks_tracking_artifact_repo.DatabricksArtifactRepository",
            return_value=mock_databricks_artifact_repo,
        ),
    ):
        repo = DatabricksRunArtifactRepository("dbfs:/databricks/mlflow-tracking/1/123")

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
            "mlflow.store.artifact.databricks_tracking_artifact_repo.DatabricksArtifactRepository",
            return_value=mock_databricks_artifact_repo,
        ),
    ):
        repo = DatabricksRunArtifactRepository("dbfs:/databricks/mlflow-tracking/1/456")

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
            "mlflow.store.artifact.databricks_tracking_artifact_repo.DatabricksArtifactRepository"
        ) as mock_fallback_repo,
    ):
        mock_fallback_repo.return_value = mock_databricks_artifact_repo
        repo = DatabricksRunArtifactRepository("dbfs:/databricks/mlflow-tracking/1/789")

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
            "mlflow.store.artifact.databricks_tracking_artifact_repo.DatabricksArtifactRepository",
            return_value=mock_databricks_artifact_repo,
        ),
    ):
        repo = DatabricksRunArtifactRepository("dbfs:/databricks/mlflow-tracking/1/123")

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


def test_constructor_with_valid_uri():
    """Test that the constructor works with valid run URIs."""
    with (
        mock.patch(
            "mlflow.store.artifact.databricks_sdk_artifact_repo.DatabricksSdkArtifactRepository"
        ),
        mock.patch(
            "mlflow.store.artifact.databricks_tracking_artifact_repo.DatabricksArtifactRepository"
        ),
    ):
        # Test basic run URI
        repo = DatabricksRunArtifactRepository("dbfs:/databricks/mlflow-tracking/1/123")
        assert repo is not None

        # Test run URI with relative path
        repo = DatabricksRunArtifactRepository("dbfs:/databricks/mlflow-tracking/1/456/artifacts")
        assert repo is not None

        # Test run URI with nested relative path
        repo = DatabricksRunArtifactRepository(
            "dbfs:/databricks/mlflow-tracking/1/789/artifacts/model"
        )
        assert repo is not None


def test_constructor_with_invalid_uri():
    """Test that the constructor raises an error with invalid URIs."""
    with pytest.raises(
        Exception,  # The exact exception type depends on the parent class
        match="Invalid artifact URI",
    ):
        DatabricksRunArtifactRepository("dbfs:/invalid/uri")

    with pytest.raises(Exception, match="Invalid artifact URI"):
        DatabricksRunArtifactRepository("dbfs:/databricks/mlflow-tracking/1")

    with pytest.raises(Exception, match="Invalid artifact URI"):
        DatabricksRunArtifactRepository("dbfs:/databricks/mlflow-tracking/1/logged_models/1")


def test_is_run_uri():
    """Test the is_run_uri static method."""
    # Valid run URIs
    assert DatabricksRunArtifactRepository.is_run_uri("dbfs:/databricks/mlflow-tracking/1/123")
    assert DatabricksRunArtifactRepository.is_run_uri(
        "dbfs:/databricks/mlflow-tracking/1/456/artifacts"
    )
    assert DatabricksRunArtifactRepository.is_run_uri(
        "dbfs:/databricks/mlflow-tracking/1/789/artifacts/model"
    )

    # Invalid URIs
    assert not DatabricksRunArtifactRepository.is_run_uri("dbfs:/databricks/mlflow-tracking/1")
    assert not DatabricksRunArtifactRepository.is_run_uri(
        "dbfs:/databricks/mlflow-tracking/1/logged_models/1"
    )
    assert not DatabricksRunArtifactRepository.is_run_uri("dbfs:/databricks/mlflow-tracking/1/tr-1")
    assert not DatabricksRunArtifactRepository.is_run_uri("dbfs:/invalid/uri")
    assert not DatabricksRunArtifactRepository.is_run_uri("s3://bucket/path")


def test_uri_parsing():
    """Test that URI components are correctly parsed."""
    with (
        mock.patch(
            "mlflow.store.artifact.databricks_sdk_artifact_repo.DatabricksSdkArtifactRepository"
        ),
        mock.patch(
            "mlflow.store.artifact.databricks_tracking_artifact_repo.DatabricksArtifactRepository"
        ),
    ):
        repo = DatabricksRunArtifactRepository("dbfs:/databricks/mlflow-tracking/123/456")

        # Test that the regex pattern matches correctly
        match = repo._get_uri_regex().search("dbfs:/databricks/mlflow-tracking/123/456")
        assert match is not None
        assert match.group("experiment_id") == "123"
        assert match.group("run_id") == "456"
        assert match.group("relative_path") is None

        # Test with relative path
        match = repo._get_uri_regex().search("dbfs:/databricks/mlflow-tracking/123/456/artifacts")
        assert match is not None
        assert match.group("experiment_id") == "123"
        assert match.group("run_id") == "456"
        assert match.group("relative_path") == "/artifacts"


def test_build_root_path():
    """Test that the root path is built correctly."""
    with (
        mock.patch(
            "mlflow.store.artifact.databricks_sdk_artifact_repo.DatabricksSdkArtifactRepository"
        ),
        mock.patch(
            "mlflow.store.artifact.databricks_tracking_artifact_repo.DatabricksArtifactRepository"
        ),
    ):
        repo = DatabricksRunArtifactRepository("dbfs:/databricks/mlflow-tracking/123/456")

        # Test root path without relative path
        match = repo._get_uri_regex().search("dbfs:/databricks/mlflow-tracking/123/456")
        root_path = repo._build_root_path("123", match, "")
        assert root_path == "/WorkspaceInternal/Mlflow/Artifacts/123/Runs/456"

        # Test root path with relative path
        match = repo._get_uri_regex().search("dbfs:/databricks/mlflow-tracking/123/456/artifacts")
        root_path = repo._build_root_path("123", match, "/artifacts")
        assert root_path == "/WorkspaceInternal/Mlflow/Artifacts/123/Runs/456/artifacts"


def test_expected_uri_format():
    """Test that the expected URI format is correct."""
    with (
        mock.patch(
            "mlflow.store.artifact.databricks_sdk_artifact_repo.DatabricksSdkArtifactRepository"
        ),
        mock.patch(
            "mlflow.store.artifact.databricks_tracking_artifact_repo.DatabricksArtifactRepository"
        ),
    ):
        repo = DatabricksRunArtifactRepository("dbfs:/databricks/mlflow-tracking/1/123")
        assert repo._get_expected_uri_format() == "databricks/mlflow-tracking/<EXP_ID>/<RUN_ID>"
