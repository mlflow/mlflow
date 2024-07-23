import posixpath
from unittest import mock

import pytest

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
    mock_response = mock.MagicMock()
    mock_response.status_code = 204

    with mock.patch("mlflow.store.artifact.uc_volume_artifact_repo.http_request") as mock_request:
        tmp_file = tmp_path.joinpath("local_file")
        tmp_file.touch()
        artifact_repo.log_artifact(tmp_file, artifact_path)
        mock_request.assert_called_once()
        rel_path = join_non_empty(artifact_path, tmp_file.name)
        endpoint = mock_request.call_args.kwargs["endpoint"]
        file_path = join_non_empty(artifact_repo.root_path, rel_path)
        assert endpoint == f"/api/2.0/fs/files{file_path}"


@pytest.mark.parametrize("artifact_path", [None, "dir"])
def test_log_artifacts(artifact_repo, artifact_path, tmp_path):
    mock_response = mock.MagicMock()
    mock_response.status_code = 204

    with mock.patch(
        "mlflow.store.artifact.uc_volume_artifact_repo.http_request", return_value=mock_response
    ) as mock_request:
        file1 = tmp_path.joinpath("local_file1")
        file1.touch()
        subdir = tmp_path.joinpath("dir")
        subdir.mkdir()
        file2 = subdir.joinpath("local_file2")
        file2.touch()
        artifact_repo.log_artifacts(tmp_path, artifact_path)
        assert mock_request.call_count == 2

        rel_path_1 = join_non_empty(artifact_path, file1.name)
        rel_path_2 = join_non_empty(artifact_path, subdir.name, file2.name)

        endpoints = [c.kwargs["endpoint"] for c in mock_request.call_args_list]
        file_path1 = join_non_empty(artifact_repo.root_path, rel_path_1)
        file_path2 = join_non_empty(artifact_repo.root_path, rel_path_2)
        assert endpoints == [f"/api/2.0/fs/files{file_path1}", f"/api/2.0/fs/files{file_path2}"]


@pytest.mark.parametrize("artifact_path", [None, "dir"])
def test_list_artifacts(artifact_repo, artifact_path):
    mock_response = mock.MagicMock()
    mock_response.status_code = 200

    mock_response.json.return_value = {
        "contents": [
            {
                "path": join_non_empty(artifact_repo.root_path, artifact_path, "directory"),
                "is_directory": True,
                "last_modified": 0,
                "name": "directory",
            },
            {
                "path": join_non_empty(artifact_repo.root_path, artifact_path, "file"),
                "is_directory": False,
                "file_size": 1,
                "last_modified": 0,
                "name": "file",
            },
        ],
    }
    with mock.patch(
        "mlflow.store.artifact.uc_volume_artifact_repo.http_request", return_value=mock_response
    ) as mock_request:
        artifacts = artifact_repo.list_artifacts(artifact_path)
        assert artifacts == [
            FileInfo(join_non_empty(artifact_path, "directory"), True, None),
            FileInfo(join_non_empty(artifact_path, "file"), False, 1),
        ]
        mock_request.assert_called_once()
        endpoint = mock_request.call_args.kwargs["endpoint"]
        directory_path = join_non_empty(artifact_repo.root_path, artifact_path)
        assert endpoint == f"/api/2.0/fs/directories{directory_path}"


@pytest.mark.parametrize("artifact_path", [None, "dir"])
def test_list_artifacts_pagination(artifact_repo, artifact_path):
    first_mock_response = mock.MagicMock()
    first_mock_response.status_code = 200
    first_mock_response.json.return_value = {
        "contents": [
            {
                "path": join_non_empty(artifact_repo.root_path, artifact_path, "directory"),
                "is_directory": True,
                "last_modified": 0,
                "name": "directory",
            },
            {
                "path": join_non_empty(artifact_repo.root_path, artifact_path, "file"),
                "is_directory": False,
                "file_size": 1,
                "last_modified": 0,
                "name": "file",
            },
        ],
        "next_page_token": "token",
    }

    second_mock_response = mock.MagicMock()
    second_mock_response.status_code = 200
    second_mock_response.json.return_value = {
        "contents": [
            {
                "path": join_non_empty(artifact_repo.root_path, artifact_path, "another_directory"),
                "is_directory": True,
                "last_modified": 0,
                "name": "another_directory",
            },
            {
                "path": join_non_empty(artifact_repo.root_path, artifact_path, "another_file"),
                "is_directory": False,
                "file_size": 1,
                "last_modified": 0,
                "name": "another_file",
            },
        ]
    }

    with mock.patch(
        "mlflow.store.artifact.uc_volume_artifact_repo.http_request",
        side_effect=[first_mock_response, second_mock_response],
    ) as mock_request:
        artifacts = artifact_repo.list_artifacts(artifact_path)
        assert artifacts == [
            FileInfo(join_non_empty(artifact_path, "another_directory"), True, None),
            FileInfo(join_non_empty(artifact_path, "another_file"), False, 1),
            FileInfo(join_non_empty(artifact_path, "directory"), True, None),
            FileInfo(join_non_empty(artifact_path, "file"), False, 1),
        ]
        assert mock_request.call_count == 2

        endpoints = [c.kwargs["endpoint"] for c in mock_request.call_args_list]
        directory_path = join_non_empty(artifact_repo.root_path, artifact_path)
        assert endpoints == [
            f"/api/2.0/fs/directories{directory_path}",
            f"/api/2.0/fs/directories{directory_path}",
        ]


@pytest.mark.parametrize("remote_file_path", ["file", "dir/file"])
def test_download_file(artifact_repo, remote_file_path, tmp_path):
    mock_response = mock.MagicMock()
    mock_response.status_code = 200
    mock_response.__enter__.return_value.iter_content.return_value = iter([b"content"])
    with mock.patch(
        "mlflow.store.artifact.uc_volume_artifact_repo.http_request", return_value=mock_response
    ) as mock_request:
        output_path = tmp_path.joinpath("output_path")
        artifact_repo._download_file(remote_file_path, output_path)
        mock_request.assert_called_once()
        assert output_path.read_bytes() == b"content"

        endpoint = mock_request.call_args.kwargs["endpoint"]
        file_path = join_non_empty(artifact_repo.root_path, remote_file_path)
        assert endpoint == f"/api/2.0/fs/files{file_path}"
