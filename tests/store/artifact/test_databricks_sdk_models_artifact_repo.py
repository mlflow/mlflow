import io
from types import SimpleNamespace
from unittest import mock

import pytest
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import NotFound
from databricks.sdk.service.files import DirectoryEntry, DownloadResponse

from mlflow.entities.file_info import FileInfo
from mlflow.entities.model_registry import ModelVersion
from mlflow.store._unity_catalog.registry.rest_store import (
    UcModelRegistryStore,
)
from mlflow.store.artifact.databricks_sdk_models_artifact_repo import (
    DatabricksSDKModelsArtifactRepository,
)
from mlflow.store.artifact.unity_catalog_models_artifact_repo import (
    UnityCatalogModelsArtifactRepository,
)

TEST_MODEL_NAME = "catalog.schema.model"
TEST_CATALOG = "catalog"
TEST_SCHEMA = "schema"
TEST_MODEL = "model"
TEST_MODEL_VERSION = 1
TEST_MODEL_BASE_PATH = f"/Models/{TEST_CATALOG}/{TEST_SCHEMA}/{TEST_MODEL}/{TEST_MODEL_VERSION}"


@pytest.fixture
def mock_databricks_workspace_client():
    mock_databricks_workspace_client = mock.MagicMock(autospec=WorkspaceClient)
    with mock.patch(
        "mlflow.store.artifact.databricks_sdk_models_artifact_repo._get_databricks_workspace_client",
        return_value=mock_databricks_workspace_client,
    ):
        yield mock_databricks_workspace_client


def test_list_artifacts_empty(mock_databricks_workspace_client):
    repo = DatabricksSDKModelsArtifactRepository(TEST_MODEL_NAME, TEST_MODEL_VERSION)
    mock_databricks_workspace_client.files.list_directory_contents.return_value = iter([])
    assert repo.list_artifacts() == []


def test_list_artifacts_listfile(mock_databricks_workspace_client):
    repo = DatabricksSDKModelsArtifactRepository(TEST_MODEL_NAME, TEST_MODEL_VERSION)
    mock_databricks_workspace_client.files.get_directory_metadata.side_effect = NotFound
    assert repo.list_artifacts() == []


def test_list_artifacts_single_file(mock_databricks_workspace_client):
    repo = DatabricksSDKModelsArtifactRepository(TEST_MODEL_NAME, TEST_MODEL_VERSION)

    entry = DirectoryEntry(is_directory=False, path=f"{TEST_MODEL_BASE_PATH}/file")
    mock_databricks_workspace_client.files.list_directory_contents.return_value = iter([entry])

    assert repo.list_artifacts() == [FileInfo(is_dir=False, path="file", file_size=None)]


def test_list_artifacts_many_files(mock_databricks_workspace_client):
    repo = DatabricksSDKModelsArtifactRepository(TEST_MODEL_NAME, TEST_MODEL_VERSION)

    # Directory structure:
    root = DirectoryEntry(is_directory=True, path=f"{TEST_MODEL_BASE_PATH}/root")
    file1 = DirectoryEntry(
        is_directory=False, path=f"{TEST_MODEL_BASE_PATH}/root/file1", file_size=1
    )
    file2 = DirectoryEntry(
        is_directory=False, path=f"{TEST_MODEL_BASE_PATH}/root/file2", file_size=2
    )
    subdir1 = DirectoryEntry(is_directory=True, path=f"{TEST_MODEL_BASE_PATH}/root/subdir1")
    file3 = DirectoryEntry(
        is_directory=False, path=f"{TEST_MODEL_BASE_PATH}/root/subdir1/file3", file_size=3
    )
    subdir2 = DirectoryEntry(is_directory=True, path=f"{TEST_MODEL_BASE_PATH}/root/subdir2")
    file4 = DirectoryEntry(
        is_directory=False, path=f"{TEST_MODEL_BASE_PATH}/root/subdir2/file4", file_size=4
    )
    file5 = DirectoryEntry(
        is_directory=False, path=f"{TEST_MODEL_BASE_PATH}/root/subdir2/file5", file_size=5
    )

    def list_directory_contents_side_effect(path):
        if path is None or path == TEST_MODEL_BASE_PATH:
            return iter([root])
        elif path == f"{TEST_MODEL_BASE_PATH}/root":
            return iter([file1, file2, subdir1, subdir2])
        elif path == f"{TEST_MODEL_BASE_PATH}/root/subdir1":
            return iter([file3])
        elif path == f"{TEST_MODEL_BASE_PATH}/root/subdir2":
            return iter([file4, file5])

    mock_databricks_workspace_client.files.list_directory_contents.side_effect = (
        list_directory_contents_side_effect
    )

    observed_artifacts = repo.list_artifacts()
    assert observed_artifacts == [FileInfo(is_dir=True, path="root", file_size=None)]

    observed_artifacts = repo.list_artifacts("root")
    assert observed_artifacts == [
        FileInfo(is_dir=False, path="root/file1", file_size=1),
        FileInfo(is_dir=False, path="root/file2", file_size=2),
        FileInfo(is_dir=True, path="root/subdir1", file_size=None),
        FileInfo(is_dir=True, path="root/subdir2", file_size=None),
    ]

    observed_artifacts = repo.list_artifacts("root/subdir1")
    assert observed_artifacts == [
        FileInfo(is_dir=False, path="root/subdir1/file3", file_size=3),
    ]

    observed_artifacts = repo.list_artifacts("root/subdir2")
    assert observed_artifacts == [
        FileInfo(is_dir=False, path="root/subdir2/file4", file_size=4),
        FileInfo(is_dir=False, path="root/subdir2/file5", file_size=5),
    ]


def test_upload_to_cloud(mock_databricks_workspace_client, tmp_path):
    # write some content to a file at a local path
    file_name = "a.txt"
    file_content = b"file_content"
    local_file_path = tmp_path.joinpath(file_name)
    local_file_path.write_bytes(file_content)

    # assert that databricks sdk file upload function is called to upload that content
    # to expected_remote_file_path
    repo = DatabricksSDKModelsArtifactRepository(TEST_MODEL_NAME, TEST_MODEL_VERSION)
    repo._upload_to_cloud(None, local_file_path, file_name)

    expected_remote_file_path = f"{TEST_MODEL_BASE_PATH}/a.txt"
    mock_databricks_workspace_client.files.upload.assert_called_once_with(
        expected_remote_file_path, mock.ANY, overwrite=mock.ANY
    )


def test_download_from_cloud(mock_databricks_workspace_client, tmp_path):
    # write some content to a file at a local path
    file_name = "a.txt"
    local_file_path = tmp_path.joinpath(file_name)

    # assert that databricks sdk file download function is called to download
    # from expected_remote_file_path to local_file_path
    mock_databricks_workspace_client.files.download.return_value = DownloadResponse(
        contents=io.BytesIO(b"file_content")
    )
    repo = DatabricksSDKModelsArtifactRepository(TEST_MODEL_NAME, TEST_MODEL_VERSION)
    repo._download_from_cloud(file_name, local_file_path)

    expected_remote_file_path = f"{TEST_MODEL_BASE_PATH}/a.txt"
    mock_databricks_workspace_client.files.download.assert_called_once_with(
        expected_remote_file_path
    )


def test_log_artifact(mock_databricks_workspace_client, tmp_path):
    # write some content to a file at a local path
    file_name = "a.txt"
    file_content = b"file_content"
    local_file_path = tmp_path.joinpath(file_name)
    local_file_path.write_bytes(file_content)

    # assert that databricks sdk file upload function is called to upload that content
    # to expected_remote_file_path
    repo = DatabricksSDKModelsArtifactRepository(TEST_MODEL_NAME, TEST_MODEL_VERSION)
    repo.log_artifact(local_file_path, file_name)

    expected_remote_file_path = f"{TEST_MODEL_BASE_PATH}/a.txt"
    mock_databricks_workspace_client.files.upload.assert_called_once_with(
        expected_remote_file_path, mock.ANY, overwrite=mock.ANY
    )


def test_mlflow_use_databricks_sdk_model_artifacts_repo_for_uc(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC", "true")
    monkeypatch.setenv("DATABRICKS_HOST", "my-host")
    monkeypatch.setenv("DATABRICKS_TOKEN", "my-token")
    with mock.patch(
        "mlflow.utils._unity_catalog_utils.call_endpoint",
        side_effect=[
            Exception("lineage emission fails"),
        ],
    ):
        uc_repo = UnityCatalogModelsArtifactRepository("models:/a.b.c/1", "databricks-uc")
        repo = uc_repo._get_artifact_repo()
        assert isinstance(repo, DatabricksSDKModelsArtifactRepository)

        store = UcModelRegistryStore(store_uri="databricks-uc", tracking_uri=str(tmp_path))
        assert isinstance(
            store._get_artifact_repo(
                ModelVersion(
                    name="name",
                    version="version",
                    creation_timestamp=1,
                ),
                TEST_MODEL_NAME,
            ),
            DatabricksSDKModelsArtifactRepository,
        )


def test_mlflow_use_databricks_sdk_model_artifacts_repo_for_uc_seg(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "my-host")
    monkeypatch.setenv("DATABRICKS_TOKEN", "my-token")
    with mock.patch(
        "mlflow.utils._unity_catalog_utils.call_endpoint",
        side_effect=[
            SimpleNamespace(is_databricks_sdk_models_artifact_repository_enabled=True),
            Exception("lineage emission fails"),
            SimpleNamespace(is_databricks_sdk_models_artifact_repository_enabled=True),
        ],
    ):
        uc_repo = UnityCatalogModelsArtifactRepository("models:/a.b.c/1", "databricks-uc")
        repo = uc_repo._get_artifact_repo()
        assert isinstance(repo, DatabricksSDKModelsArtifactRepository)

        store = UcModelRegistryStore(store_uri="databricks-uc", tracking_uri=str(tmp_path))
        assert isinstance(
            store._get_artifact_repo(
                ModelVersion(
                    name="name",
                    version="version",
                    creation_timestamp=1,
                ),
                TEST_MODEL_NAME,
            ),
            DatabricksSDKModelsArtifactRepository,
        )
