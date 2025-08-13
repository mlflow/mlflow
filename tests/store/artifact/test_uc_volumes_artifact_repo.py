import posixpath
from unittest import mock

import pytest
from databricks.sdk.service.files import DirectoryEntry

from mlflow.entities.file_info import FileInfo
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.uc_volume_artifact_repo import UCVolumesArtifactRepository

HOST = "http://localhost:5000"


def join_non_empty(*args):
    """
    Join path components, ignoring empty components.
    """
    non_empty_args = [a for a in args if a]
    return posixpath.join(*non_empty_args)


@pytest.fixture(autouse=True)
def set_creds(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "http://localhost:5000")
    monkeypatch.setenv("DATABRICKS_TOKEN", "abc")


@pytest.fixture
def artifact_repo(monkeypatch):
    return get_artifact_repository("dbfs:/Volumes/catalog/schema/volume/run_id/artifacts")


@pytest.mark.parametrize(
    "artifact_uri",
    [
        "dbfs:/Volumes/catalog/schema/volume/path",
        "dbfs:/volumes/catalog/schema/volume/path",
        "dbfs:/Volume/catalog/schema/volume/path",
        "dbfs:/volume/catalog/schema/volume/path",
        "dbfs:/Volumes/catalog/schema/volume/some/path",
        "dbfs://profile@databricks/Volumes/catalog/schema/volume/some/path",
    ],
)
def test_get_artifact_repository(artifact_uri: str):
    repo = get_artifact_repository(artifact_uri)
    assert isinstance(repo, UCVolumesArtifactRepository)


@pytest.mark.parametrize(
    "artifact_uri",
    [
        "dbfs:/Volumes/catalog",
        "dbfs:/Volumes/catalog/schema",
        "dbfs:/Volumes/catalog/schema/volume",
        "dbfs://profile@databricks/Volumes/catalog",
        "dbfs://profile@databricks/Volumes/catalog/schema",
        "dbfs://profile@databricks/Volumes/catalog/schema/volume",
    ],
)
def test_get_artifact_repository_invalid_uri(artifact_uri: str):
    with pytest.raises(MlflowException, match="UC volumes URI must be of the form"):
        get_artifact_repository(artifact_uri)


@pytest.mark.parametrize("artifact_path", [None, "dir"])
def test_log_artifact(artifact_repo, artifact_path, tmp_path):
    with mock.patch(
        "mlflow.store.artifact.databricks_sdk_artifact_repo.DatabricksSdkArtifactRepository.files_api"
    ) as mock_files_api:
        tmp_file = tmp_path.joinpath("local_file")
        tmp_file.touch()
        artifact_repo.log_artifact(tmp_file, artifact_path)
        mock_files_api.upload.assert_called_once_with(
            join_non_empty(
                "/Volumes/catalog/schema/volume/run_id/artifacts", artifact_path, tmp_file.name
            ),
            mock.ANY,
            overwrite=True,
        )


@pytest.mark.parametrize("artifact_path", [None, "dir"])
def test_log_artifacts(artifact_repo, artifact_path, tmp_path):
    with mock.patch(
        "mlflow.store.artifact.databricks_sdk_artifact_repo.DatabricksSdkArtifactRepository.files_api"
    ) as mock_files_api:
        file1 = tmp_path.joinpath("local_file1")
        file1.touch()
        subdir = tmp_path.joinpath("dir")
        subdir.mkdir()
        file2 = subdir.joinpath("local_file2")
        file2.touch()
        artifact_repo.log_artifacts(tmp_path, artifact_path)
        assert mock_files_api.upload.call_count == 2
        mock_files_api.upload.assert_any_call(
            join_non_empty(
                "/Volumes/catalog/schema/volume/run_id/artifacts", artifact_path, file1.name
            ),
            mock.ANY,
            overwrite=True,
        )
        mock_files_api.upload.assert_any_call(
            join_non_empty(
                "/Volumes/catalog/schema/volume/run_id/artifacts", artifact_path, "dir", file2.name
            ),
            mock.ANY,
            overwrite=True,
        )


@pytest.mark.parametrize("artifact_path", [None, "dir"])
def test_list_artifacts(artifact_repo, artifact_path):
    with mock.patch(
        "mlflow.store.artifact.databricks_sdk_artifact_repo.DatabricksSdkArtifactRepository.files_api"
    ) as mock_files_api:
        mock_files_api.list_directory_contents.return_value = [
            DirectoryEntry(
                path=join_non_empty(
                    "/Volumes/catalog/schema/volume/run_id/artifacts", artifact_path, "file"
                ),
                is_directory=False,
                file_size=123,
            ),
            DirectoryEntry(
                path=join_non_empty(
                    "/Volumes/catalog/schema/volume/run_id/artifacts", artifact_path, "dir"
                ),
                is_directory=True,
            ),
        ]
        artifacts = artifact_repo.list_artifacts(artifact_path)
        assert artifacts == [
            FileInfo(join_non_empty(artifact_path, "dir"), True, None),
            FileInfo(join_non_empty(artifact_path, "file"), False, 123),
        ]


@pytest.mark.parametrize("remote_file_path", ["file", "dir/file"])
def test_download_file(artifact_repo, remote_file_path, tmp_path):
    with mock.patch(
        "mlflow.store.artifact.databricks_sdk_artifact_repo.DatabricksSdkArtifactRepository.files_api"
    ) as mock_files_api:
        output_path = tmp_path.joinpath("output_path")
        mock_files_api.download.return_value.contents.read.side_effect = [b"test", b""]
        artifact_repo._download_file(remote_file_path, output_path)
        assert mock_files_api.download.call_count == 1
