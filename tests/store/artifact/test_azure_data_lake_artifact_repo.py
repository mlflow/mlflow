import json
import os
import posixpath
from unittest import mock
from unittest.mock import ANY

import pytest
import requests
from azure.core.credentials import AzureSasCredential
from azure.storage.filedatalake import (
    DataLakeDirectoryClient,
    DataLakeFileClient,
    DataLakeServiceClient,
    FileSystemClient,
    PathProperties,
)

from mlflow.exceptions import MlflowException, MlflowTraceDataCorrupted
from mlflow.protos.databricks_artifacts_pb2 import ArtifactCredentialInfo
from mlflow.protos.service_pb2 import FileInfo
from mlflow.store.artifact.artifact_repo import try_read_trace_data
from mlflow.store.artifact.azure_data_lake_artifact_repo import (
    AzureDataLakeArtifactRepository,
    _parse_abfss_uri,
)

TEST_ROOT_PATH = "some/path"
TEST_DATA_LAKE_URI_BASE = "abfss://filesystem@account.dfs.core.windows.net"
TEST_DATA_LAKE_URI = posixpath.join(TEST_DATA_LAKE_URI_BASE, TEST_ROOT_PATH)
TEST_CREDENTIAL = mock.Mock()

ADLS_REPOSITORY_PACKAGE = "mlflow.store.artifact.azure_data_lake_artifact_repo"
ADLS_ARTIFACT_REPOSITORY = f"{ADLS_REPOSITORY_PACKAGE}.AzureDataLakeArtifactRepository"


class MockPathList:
    def __init__(self, items, next_marker=None):
        self.items = items
        self.next_marker = next_marker

    def __iter__(self):
        return iter(self.items)


@pytest.fixture
def mock_data_lake_client():
    mock_adls_client = mock.MagicMock(autospec=DataLakeServiceClient)
    with mock.patch(
        "mlflow.store.artifact.azure_data_lake_artifact_repo._get_data_lake_client",
        return_value=mock_adls_client,
    ):
        yield mock_adls_client


@pytest.fixture
def mock_filesystem_client(mock_data_lake_client):
    mock_fs_client = mock.MagicMock(autospec=FileSystemClient)
    mock_data_lake_client.get_file_system_client.return_value = mock_fs_client
    return mock_fs_client


@pytest.fixture
def mock_directory_client(mock_filesystem_client):
    mock_directory_client = mock.MagicMock(autospec=DataLakeDirectoryClient)
    mock_filesystem_client.get_directory_client.return_value = mock_directory_client
    return mock_directory_client


@pytest.fixture
def mock_file_client(mock_directory_client):
    mock_file_client = mock.MagicMock(autospec=DataLakeFileClient)
    mock_directory_client.get_file_client.return_value = mock_file_client
    return mock_file_client


@pytest.mark.parametrize(
    ("uri", "filesystem", "account", "region_suffix", "path"),
    [
        (
            "abfss://filesystem@acct.dfs.core.windows.net/path",
            "filesystem",
            "acct",
            "dfs.core.windows.net",
            "path",
        ),
        (
            "abfss://filesystem@acct.dfs.core.windows.net",
            "filesystem",
            "acct",
            "dfs.core.windows.net",
            "",
        ),
        (
            "abfss://filesystem@acct.dfs.core.windows.net/",
            "filesystem",
            "acct",
            "dfs.core.windows.net",
            "",
        ),
        (
            "abfss://filesystem@acct.dfs.core.windows.net/a/b",
            "filesystem",
            "acct",
            "dfs.core.windows.net",
            "a/b",
        ),
        (
            "abfss://filesystem@acct.dfs.core.chinacloudapi.cn/a/b",
            "filesystem",
            "acct",
            "dfs.core.chinacloudapi.cn",
            "a/b",
        ),
        (
            "abfss://filesystem@acct.privatelink.dfs.core.windows.net/a/b",
            "filesystem",
            "acct",
            "privatelink.dfs.core.windows.net",
            "a/b",
        ),
        (
            "abfss://filesystem@acct.dfs.core.usgovcloudapi.net/a/b",
            "filesystem",
            "acct",
            "dfs.core.usgovcloudapi.net",
            "a/b",
        ),
    ],
)
def test_parse_valid_abfss_uri(uri, filesystem, account, region_suffix, path):
    assert _parse_abfss_uri(uri) == (filesystem, account, region_suffix, path)


@pytest.mark.parametrize(
    "uri",
    [
        "abfss://filesystem@acct/path",
        "abfss://acct.dfs.core.windows.net/path",
        "abfss://@acct.dfs.core.windows.net/path",
    ],
)
def test_parse_invalid_abfss_uri(uri):
    with pytest.raises(MlflowException, match="ABFSS URI must be of the form"):
        _parse_abfss_uri(uri)


def test_parse_invalid_abfss_uri_bad_scheme():
    with pytest.raises(MlflowException, match="Not an ABFSS URI"):
        _parse_abfss_uri("abfs://cont@acct.dfs.core.windows.net/path")


def test_list_artifacts_empty(mock_data_lake_client):
    repo = AzureDataLakeArtifactRepository(TEST_DATA_LAKE_URI, credential=TEST_CREDENTIAL)
    mock_data_lake_client.get_file_system_client.get_paths.return_value = MockPathList([])
    assert repo.list_artifacts() == []


def test_list_artifacts_single_file(mock_data_lake_client):
    repo = AzureDataLakeArtifactRepository(TEST_DATA_LAKE_URI, credential=TEST_CREDENTIAL)

    # Evaluate single file
    path_props = PathProperties(name=posixpath.join(TEST_DATA_LAKE_URI, "file"), content_length=42)
    mock_data_lake_client.get_file_system_client.get_paths.return_value = MockPathList([path_props])
    assert repo.list_artifacts("file") == []


def test_list_artifacts(mock_filesystem_client):
    repo = AzureDataLakeArtifactRepository(TEST_DATA_LAKE_URI, credential=TEST_CREDENTIAL)

    # Create some files to return
    dir_prefix = PathProperties(is_directory=True, name=posixpath.join(TEST_ROOT_PATH, "dir"))
    path_props = PathProperties(content_length=42, name=posixpath.join(TEST_ROOT_PATH, "file"))
    mock_filesystem_client.get_paths.return_value = MockPathList([dir_prefix, path_props])

    artifacts = repo.list_artifacts()
    mock_filesystem_client.get_paths.assert_called_once_with(path=TEST_ROOT_PATH, recursive=False)
    assert artifacts[0].path == "dir"
    assert artifacts[0].is_dir is True
    assert artifacts[0].file_size is None
    assert artifacts[1].path == "file"
    assert artifacts[1].is_dir is False
    assert artifacts[1].file_size == 42

    mock_filesystem_client.reset_mock()
    repo.list_artifacts(path="nonexistent-dir")
    mock_filesystem_client.get_paths.assert_called_once_with(
        path=posixpath.join(TEST_ROOT_PATH, "nonexistent-dir"), recursive=False
    )


@pytest.mark.parametrize(
    "contents",
    ["", "B"],
)
def test_log_artifact(mock_filesystem_client, mock_directory_client, tmp_path, contents):
    file_name = "b.txt"
    repo = AzureDataLakeArtifactRepository(TEST_DATA_LAKE_URI, credential=TEST_CREDENTIAL)

    parentd = tmp_path.joinpath("data")
    parentd.mkdir()
    subd = parentd.joinpath("subdir")
    subd.mkdir()
    subd.joinpath("b.txt").write_text(contents)

    repo.log_artifact(subd.joinpath("b.txt"))

    mock_filesystem_client.get_directory_client.assert_called_once_with(TEST_ROOT_PATH)
    mock_directory_client.get_file_client.assert_called_once_with(file_name)

    if contents == "":
        mock_directory_client.get_file_client(file_name).create_file.assert_called()
    else:
        mock_directory_client.get_file_client(file_name).upload_data.assert_called()


def test_log_artifacts(mock_filesystem_client, mock_directory_client, tmp_path):
    fake_sas_token = "fake_session_token"
    repo = AzureDataLakeArtifactRepository(
        TEST_DATA_LAKE_URI, credential=AzureSasCredential(fake_sas_token)
    )

    parentd = tmp_path.joinpath("data")
    parentd.mkdir()
    subd = parentd.joinpath("subdir")
    subd.mkdir()
    parentd.joinpath("a.txt").write_text("A")
    subd.joinpath("b.txt").write_text("B")
    subd.joinpath("empty-file.txt").write_text("")

    repo.log_artifacts(parentd)

    called_directories = [
        call[0][0] for call in mock_filesystem_client.get_directory_client.call_args_list
    ]
    assert len(called_directories) == 3
    assert sorted(called_directories) == [
        posixpath.join(TEST_ROOT_PATH, "."),
        posixpath.join(TEST_ROOT_PATH, "subdir"),
        posixpath.join(TEST_ROOT_PATH, "subdir"),
    ]

    uploaded_filenames = [
        call[0][0] for call in mock_directory_client.get_file_client.call_args_list
    ]
    assert len(uploaded_filenames) == 3
    assert set(uploaded_filenames) == {"a.txt", "b.txt", "empty-file.txt"}

    mock_directory_client.get_file_client("a.txt").upload_data.assert_called()
    mock_directory_client.get_file_client("b.txt").upload_data.assert_called()
    mock_directory_client.get_file_client("subdir/empty-file.txt").create_file.assert_called()


def test_log_artifacts_in_parallel_when_necessary(tmp_path, monkeypatch):
    fake_sas_token = "fake_session_token"
    repo = AzureDataLakeArtifactRepository(
        TEST_DATA_LAKE_URI, credential=AzureSasCredential(fake_sas_token)
    )

    parentd = tmp_path.joinpath("data")
    parentd.mkdir()
    parentd.joinpath("a.txt").write_text("ABCDE")

    monkeypatch.setenv("MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE", "0")
    with (
        mock.patch(
            f"{ADLS_ARTIFACT_REPOSITORY}._multipart_upload", return_value=None
        ) as multipart_upload_mock,
        mock.patch(f"{ADLS_ARTIFACT_REPOSITORY}.log_artifact", return_value=None),
    ):
        repo.log_artifacts(parentd)
        multipart_upload_mock.assert_called_with(
            ArtifactCredentialInfo(
                signed_uri="https://account.dfs.core.windows.net/filesystem/some/path/"
                + "./a.txt?fake_session_token"
            ),
            ANY,
            "./a.txt",
        )


@pytest.mark.parametrize(
    ("file_size", "is_parallel_download"),
    [(None, False), (100, False), (500 * 1024**2 - 1, False), (500 * 1024**2, True)],
)
def test_download_file_in_parallel_when_necessary(file_size, is_parallel_download):
    repo = AzureDataLakeArtifactRepository(TEST_DATA_LAKE_URI, credential=TEST_CREDENTIAL)
    remote_file_path = "file_1.txt"
    list_artifacts_result = (
        [FileInfo(path=remote_file_path, is_dir=False, file_size=file_size)] if file_size else []
    )
    with (
        mock.patch(
            f"{ADLS_ARTIFACT_REPOSITORY}.list_artifacts",
            return_value=list_artifacts_result,
        ),
        mock.patch(
            f"{ADLS_ARTIFACT_REPOSITORY}._download_from_cloud", return_value=None
        ) as download_mock,
        mock.patch(
            f"{ADLS_ARTIFACT_REPOSITORY}._parallelized_download_from_cloud", return_value=None
        ) as parallel_download_mock,
    ):
        repo.download_artifacts("")
        if is_parallel_download:
            parallel_download_mock.assert_called_with(file_size, remote_file_path, ANY)
        else:
            download_mock.assert_called()


def test_download_file_artifact(mock_directory_client, mock_file_client, tmp_path):
    repo = AzureDataLakeArtifactRepository(TEST_DATA_LAKE_URI, credential=TEST_CREDENTIAL)

    def create_file(file):
        local_path = os.path.basename(file.name)
        f = tmp_path.joinpath(local_path)
        f.write_text("hello world!")

    mock_file_client.download_file().readinto.side_effect = create_file
    repo.download_artifacts("test.txt")
    assert os.path.exists(os.path.join(tmp_path, "test.txt"))
    mock_directory_client.get_file_client.assert_called_once_with("test.txt")


def test_download_directory_artifact(mock_filesystem_client, mock_file_client, tmp_path):
    repo = AzureDataLakeArtifactRepository(TEST_DATA_LAKE_URI, credential=TEST_CREDENTIAL)

    file_path_1 = "file_1"
    file_path_2 = "file_2"

    path_props_1 = PathProperties(
        content_length=42, name=posixpath.join(TEST_ROOT_PATH, file_path_1)
    )
    path_props_2 = PathProperties(
        content_length=42, name=posixpath.join(TEST_ROOT_PATH, file_path_2)
    )

    dir_name = "dir"
    dir_path = posixpath.join(TEST_ROOT_PATH, dir_name)
    dir_props = PathProperties(is_directory=True, name=dir_path)
    dir_file_name = "subdir_file"
    dir_file_props = PathProperties(content_length=42, name=posixpath.join(dir_path, dir_file_name))

    def get_mock_listing(*args, **kwargs):
        """
        Produces a mock listing that only contains content if the
        specified prefix is the artifact root. This allows us to mock
        `list_artifacts` during the `_download_artifacts_into` subroutine
        without recursively listing the same artifacts at every level of the
        directory traversal.
        """

        path_arg = posixpath.abspath(kwargs["path"])
        if path_arg == posixpath.abspath(TEST_ROOT_PATH):
            return MockPathList([path_props_1, path_props_2, dir_props])
        elif path_arg == posixpath.abspath(dir_path):
            return MockPathList([dir_file_props])
        else:
            return MockPathList([])

    def create_file(buffer):
        buffer.write(b"hello world!")

    mock_filesystem_client.get_paths.side_effect = get_mock_listing
    mock_file_client.download_file().readinto.side_effect = create_file

    # Ensure that the root directory can be downloaded successfully
    dest_dir = tmp_path.joinpath("download_dir")
    dest_dir.mkdir()
    repo.download_artifacts(artifact_path="", dst_path=dest_dir)
    # Ensure that the `mkfile` side effect copied all of the download artifacts into `tmp_path`
    dir_contents = os.listdir(dest_dir)
    assert file_path_1 in dir_contents
    assert file_path_2 in dir_contents
    assert dir_name in dir_contents
    subdir_contents = os.listdir(dest_dir.joinpath(dir_name))
    assert dir_file_name in subdir_contents


def test_refresh_credentials():
    dl_client = mock.MagicMock()
    with mock.patch(
        f"{ADLS_REPOSITORY_PACKAGE}._get_data_lake_client", return_value=dl_client
    ) as get_data_lake_client_mock:
        fs_client = mock.MagicMock()
        dl_client.get_file_system_client.return_value = fs_client
        resp = requests.Response()
        resp.status_code = 401
        err = requests.HTTPError(response=resp)
        fs_client.get_directory_client.side_effect = err

        second_credential = AzureSasCredential("new_fake_token")

        def credential_refresh():
            return {"credential": second_credential}

        first_credential = AzureSasCredential("fake_token")
        repo = AzureDataLakeArtifactRepository(
            TEST_DATA_LAKE_URI,
            credential=first_credential,
            credential_refresh_def=credential_refresh,
        )

        get_data_lake_client_mock.assert_called_with(account_url=ANY, credential=first_credential)

        try:
            repo._download_from_cloud("test.txt", "local_path")
        except requests.HTTPError as e:
            assert e == err

        get_data_lake_client_mock.assert_called_with(account_url=ANY, credential=second_credential)


def test_trace_data(mock_data_lake_client, tmp_path):
    repo = AzureDataLakeArtifactRepository(TEST_DATA_LAKE_URI, credential=TEST_CREDENTIAL)
    with pytest.raises(MlflowException, match=r"Trace data not found for path="):
        repo.download_trace_data()
    trace_data_path = tmp_path.joinpath("traces.json")
    trace_data_path.write_text("invalid data")
    with (
        mock.patch(
            "mlflow.store.artifact.artifact_repo.try_read_trace_data",
            side_effect=lambda x: try_read_trace_data(trace_data_path),
        ),
        pytest.raises(MlflowTraceDataCorrupted, match=r"Trace data is corrupted for path="),
    ):
        repo.download_trace_data()

    mock_trace_data = {"spans": [], "request": {"test": 1}, "response": {"test": 2}}
    trace_data_path.write_text(json.dumps(mock_trace_data))
    with mock.patch(
        "mlflow.store.artifact.artifact_repo.try_read_trace_data",
        side_effect=lambda x: try_read_trace_data(trace_data_path),
    ):
        assert repo.download_trace_data() == mock_trace_data
