import io
from types import SimpleNamespace
from unittest import mock

import pytest
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import NotFound
from databricks.sdk.service.files import DirectoryEntry, DownloadResponse

from mlflow.entities.file_info import FileInfo
from mlflow.store._unity_catalog.registry.rest_store import UcModelRegistryStore
from mlflow.store.artifact.databricks_sdk_models_artifact_repo import (
    DatabricksSDKModelsArtifactRepository,
    _get_databricks_workspace_client,
)
from mlflow.store.artifact.unity_catalog_models_artifact_repo import (
    UnityCatalogModelsArtifactRepository,
)
from mlflow.utils.rest_utils import MlflowHostCreds

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
    file_name = "a.txt"
    file_content = b"file_content"
    local_file_path = tmp_path.joinpath(file_name)
    local_file_path.write_bytes(file_content)

    repo = DatabricksSDKModelsArtifactRepository(TEST_MODEL_NAME, TEST_MODEL_VERSION)
    repo._upload_to_cloud(None, local_file_path, file_name)

    expected_remote_file_path = f"{TEST_MODEL_BASE_PATH}/a.txt"
    mock_databricks_workspace_client.files.upload.assert_called_once_with(
        expected_remote_file_path, mock.ANY, overwrite=mock.ANY
    )


def test_download_from_cloud(mock_databricks_workspace_client, tmp_path):
    file_name = "a.txt"
    local_file_path = tmp_path.joinpath(file_name)

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
    file_name = "a.txt"
    file_content = b"file_content"
    local_file_path = tmp_path.joinpath(file_name)
    local_file_path.write_bytes(file_content)

    repo = DatabricksSDKModelsArtifactRepository(TEST_MODEL_NAME, TEST_MODEL_VERSION)
    repo.log_artifact(local_file_path, file_name)

    expected_remote_file_path = f"{TEST_MODEL_BASE_PATH}/a.txt"
    mock_databricks_workspace_client.files.upload.assert_called_once_with(
        expected_remote_file_path, mock.ANY, overwrite=mock.ANY
    )


def test_get_workspace_client_uses_resolved_creds_when_token_present():
    creds = MlflowHostCreds(host="https://my-host", token="my-token")
    with (
        mock.patch(
            "mlflow.utils.databricks_utils.get_databricks_host_creds",
            return_value=creds,
        ) as mock_get_creds,
        mock.patch("databricks.sdk.WorkspaceClient") as mock_workspace_client,
    ):
        _get_databricks_workspace_client("databricks-uc")

    mock_get_creds.assert_called_once_with("databricks-uc")
    mock_workspace_client.assert_called_once_with(host="https://my-host", token="my-token")


@pytest.mark.parametrize(
    "creds",
    [
        MlflowHostCreds(host="https://my-host"),  # host resolved but no token
        None,  # creds resolution returns nothing usable
    ],
)
def test_get_workspace_client_falls_back_to_default_auth_without_token(creds):
    with (
        mock.patch(
            "mlflow.utils.databricks_utils.get_databricks_host_creds",
            return_value=creds,
        ),
        mock.patch("databricks.sdk.WorkspaceClient") as mock_workspace_client,
    ):
        _get_databricks_workspace_client("databricks-uc")

    mock_workspace_client.assert_called_once_with()


def test_get_workspace_client_falls_back_when_creds_resolution_raises():
    with (
        mock.patch(
            "mlflow.utils.databricks_utils.get_databricks_host_creds",
            side_effect=Exception("cannot resolve creds"),
        ),
        mock.patch("databricks.sdk.WorkspaceClient") as mock_workspace_client,
    ):
        _get_databricks_workspace_client("databricks-uc")

    mock_workspace_client.assert_called_once_with()


def test_uc_models_repo_uses_sdk_repo_when_env_var_enabled(
    mock_databricks_workspace_client, monkeypatch
):
    monkeypatch.setenv("MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC", "true")
    monkeypatch.setenv("DATABRICKS_HOST", "my-host")
    monkeypatch.setenv("DATABRICKS_TOKEN", "my-token")

    uc_repo = UnityCatalogModelsArtifactRepository("models:/a.b.c/1", "databricks-uc")
    assert isinstance(uc_repo._get_artifact_repo(), DatabricksSDKModelsArtifactRepository)


def test_uc_models_repo_uses_scoped_cloud_repo_by_default(monkeypatch):
    monkeypatch.delenv("MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC", raising=False)
    monkeypatch.setenv("DATABRICKS_HOST", "my-host")
    monkeypatch.setenv("DATABRICKS_TOKEN", "my-token")

    uc_repo = UnityCatalogModelsArtifactRepository("models:/a.b.c/1", "databricks-uc")
    sentinel = object()
    with (
        mock.patch.object(uc_repo, "_get_scoped_token", return_value=mock.MagicMock()),
        mock.patch.object(uc_repo, "_get_blob_storage_path", return_value="s3://bucket/path"),
        mock.patch(
            "mlflow.store.artifact.unity_catalog_models_artifact_repo."
            "get_artifact_repo_from_storage_info",
            return_value=sentinel,
        ) as mock_factory,
    ):
        assert uc_repo._get_artifact_repo() is sentinel
        mock_factory.assert_called_once()


def test_uc_registry_store_uses_sdk_repo_for_upload_when_env_var_enabled(
    mock_databricks_workspace_client, monkeypatch, tmp_path
):
    monkeypatch.setenv("MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC", "true")
    monkeypatch.setenv("DATABRICKS_HOST", "my-host")
    monkeypatch.setenv("DATABRICKS_TOKEN", "my-token")

    store = UcModelRegistryStore(store_uri="databricks-uc", tracking_uri=str(tmp_path))
    model_version = SimpleNamespace(version="1", storage_location="s3://bucket/path")
    repo = store._get_artifact_repo(model_version, TEST_MODEL_NAME)
    assert isinstance(repo, DatabricksSDKModelsArtifactRepository)


def test_uc_registry_store_uses_scoped_cloud_repo_for_upload_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC", raising=False)
    monkeypatch.setenv("DATABRICKS_HOST", "my-host")
    monkeypatch.setenv("DATABRICKS_TOKEN", "my-token")

    store = UcModelRegistryStore(store_uri="databricks-uc", tracking_uri=str(tmp_path))
    model_version = SimpleNamespace(version="1", storage_location="s3://bucket/path")
    sentinel = object()
    with (
        mock.patch.object(
            store, "_get_temporary_model_version_write_credentials", return_value=mock.MagicMock()
        ) as mock_creds,
        mock.patch(
            "mlflow.store._unity_catalog.registry.rest_store.get_artifact_repo_from_storage_info",
            return_value=sentinel,
        ) as mock_factory,
    ):
        assert store._get_artifact_repo(model_version, TEST_MODEL_NAME) is sentinel
        mock_creds.assert_called_once()
        mock_factory.assert_called_once()
