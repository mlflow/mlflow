import os
import posixpath
import pytest
from unittest import mock

from azure.storage.blob import BlobServiceClient, BlobPrefix, BlobProperties

from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.azure_blob_artifact_repo import AzureBlobArtifactRepository


TEST_ROOT_PATH = "some/path"
TEST_BLOB_CONTAINER_ROOT = "wasbs://container@account.blob.core.windows.net/"
TEST_URI = os.path.join(TEST_BLOB_CONTAINER_ROOT, TEST_ROOT_PATH)


class MockBlobList:
    def __init__(self, items, next_marker=None):
        self.items = items
        self.next_marker = next_marker

    def __iter__(self):
        return iter(self.items)


@pytest.fixture
def mock_client():
    # Make sure that our environment variable aren't set to actually access Azure
    old_access_key = os.environ.get("AZURE_STORAGE_ACCESS_KEY")
    if old_access_key is not None:
        del os.environ["AZURE_STORAGE_ACCESS_KEY"]
    old_conn_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if old_conn_string is not None:
        del os.environ["AZURE_STORAGE_CONNECTION_STRING"]

    yield mock.MagicMock(autospec=BlobServiceClient)

    if old_access_key is not None:
        os.environ["AZURE_STORAGE_ACCESS_KEY"] = old_access_key
    if old_conn_string is not None:
        os.environ["AZURE_STORAGE_CONNECTION_STRING"] = old_conn_string


def test_artifact_uri_factory(mock_client):
    # pylint: disable=unused-argument
    # We pass in the mock_client here to clear Azure environment variables, but we don't use it;
    # We do need to set up a fake access key for the code to run though
    os.environ["AZURE_STORAGE_ACCESS_KEY"] = ""
    repo = get_artifact_repository(TEST_URI)
    assert isinstance(repo, AzureBlobArtifactRepository)
    del os.environ["AZURE_STORAGE_ACCESS_KEY"]


@mock.patch("azure.identity.DefaultAzureCredential")
def test_default_az_cred_if_no_env_vars(mock_default_azure_credential, mock_client):
    # pylint: disable=unused-argument
    # We pass in the mock_client here to clear Azure environment variables, but we don't use it
    AzureBlobArtifactRepository(TEST_URI)
    assert mock_default_azure_credential.call_count == 1


def test_parse_global_wasbs_uri():
    parse = AzureBlobArtifactRepository.parse_wasbs_uri
    global_api_suffix = "blob.core.windows.net"

    global_wasb_with_short_path = "wasbs://cont@acct.blob.core.windows.net/path"
    assert parse(global_wasb_with_short_path) == ("cont", "acct", "path", global_api_suffix)

    global_wasb_without_path = "wasbs://cont@acct.blob.core.windows.net"
    assert parse(global_wasb_without_path) == ("cont", "acct", "", global_api_suffix)

    global_wasb_without_path2 = "wasbs://cont@acct.blob.core.windows.net/"
    assert parse(global_wasb_without_path2) == ("cont", "acct", "", global_api_suffix)

    global_wasb_with_multi_path = "wasbs://cont@acct.blob.core.windows.net/a/b"
    assert parse(global_wasb_with_multi_path) == ("cont", "acct", "a/b", global_api_suffix)

    with pytest.raises(Exception, match="WASBS URI must be of the form"):
        parse("wasbs://cont@acct.blob.core.evil.net/path")
    with pytest.raises(Exception, match="WASBS URI must be of the form"):
        parse("wasbs://cont@acct/path")
    with pytest.raises(Exception, match="WASBS URI must be of the form"):
        parse("wasbs://acct.blob.core.windows.net/path")
    with pytest.raises(Exception, match="WASBS URI must be of the form"):
        parse("wasbs://@acct.blob.core.windows.net/path")
    with pytest.raises(Exception, match="WASBS URI must be of the form"):
        parse("wasbs://cont@acctxblob.core.windows.net/path")
    with pytest.raises(Exception, match="Not a WASBS URI"):
        parse("wasb://cont@acct.blob.core.windows.net/path")


def test_parse_cn_wasbs_uri():
    parse = AzureBlobArtifactRepository.parse_wasbs_uri
    cn_api_suffix = "blob.core.chinacloudapi.cn"

    cn_wasb_with_short_path = "wasbs://cont@acct.blob.core.chinacloudapi.cn/path"
    assert parse(cn_wasb_with_short_path) == ("cont", "acct", "path", cn_api_suffix)

    cn_wasb_without_path = "wasbs://cont@acct.blob.core.chinacloudapi.cn"
    assert parse(cn_wasb_without_path) == ("cont", "acct", "", cn_api_suffix)

    cn_wasb_without_path2 = "wasbs://cont@acct.blob.core.chinacloudapi.cn/"
    assert parse(cn_wasb_without_path2) == ("cont", "acct", "", cn_api_suffix)

    cn_wasb_with_multi_path = "wasbs://cont@acct.blob.core.chinacloudapi.cn/a/b"
    assert parse(cn_wasb_with_multi_path) == ("cont", "acct", "a/b", cn_api_suffix)

    with pytest.raises(Exception, match="WASBS URI must be of the form"):
        parse("wasbs://cont@acct.blob.core.evil.cn/path")
    with pytest.raises(Exception, match="WASBS URI must be of the form"):
        parse("wasbs://cont@acct/path")
    with pytest.raises(Exception, match="WASBS URI must be of the form"):
        parse("wasbs://acct.blob.core.chinacloudapi.cn/path")
    with pytest.raises(Exception, match="WASBS URI must be of the form"):
        parse("wasbs://@acct.blob.core.chinacloudapi.cn/path")
    with pytest.raises(Exception, match="WASBS URI must be of the form"):
        parse("wasbs://cont@acctxblob.core.chinacloudapi.cn/path")
    with pytest.raises(Exception, match="Not a WASBS URI"):
        parse("wasb://cont@acct.blob.core.chinacloudapi.cn/path")


def test_list_artifacts_empty(mock_client):
    repo = AzureBlobArtifactRepository(TEST_URI, mock_client)
    mock_client.get_container_client().walk_blobs.return_value = MockBlobList([])
    assert repo.list_artifacts() == []


def test_list_artifacts_single_file(mock_client):
    repo = AzureBlobArtifactRepository(TEST_URI, mock_client)

    # Evaluate single file
    blob_props = BlobProperties()
    blob_props.name = posixpath.join(TEST_ROOT_PATH, "file")
    mock_client.get_container_client().walk_blobs.return_value = MockBlobList([blob_props])
    assert repo.list_artifacts("file") == []


@pytest.mark.parametrize("root_path", ["some/path", "some/path/"])
def test_list_artifacts(mock_client, root_path):
    repo = AzureBlobArtifactRepository(
        posixpath.join(TEST_BLOB_CONTAINER_ROOT, root_path), mock_client
    )

    # Create some files to return
    dir_prefix = BlobPrefix()
    dir_prefix.name = posixpath.join(TEST_ROOT_PATH, "dir")

    blob_props = BlobProperties()
    blob_props.size = 42
    blob_props.name = posixpath.join(TEST_ROOT_PATH, "file")

    mock_client.get_container_client().walk_blobs.return_value = MockBlobList(
        [dir_prefix, blob_props]
    )

    artifacts = repo.list_artifacts()
    mock_client.get_container_client().walk_blobs.assert_called_with(name_starts_with="some/path/")
    assert artifacts[0].path == "dir"
    assert artifacts[0].is_dir is True
    assert artifacts[0].file_size is None
    assert artifacts[1].path == "file"
    assert artifacts[1].is_dir is False
    assert artifacts[1].file_size == 42


def test_log_artifact(mock_client, tmpdir):
    repo = AzureBlobArtifactRepository(TEST_URI, mock_client)

    d = tmpdir.mkdir("data")
    f = d.join("test.txt")
    f.write("hello world!")
    fpath = posixpath.join(d.strpath, "test.txt")

    repo.log_artifact(fpath)

    mock_client.get_container_client.assert_called_with("container")
    arg1, arg2 = mock_client.get_container_client().upload_blob.call_args[0]
    assert arg1 == posixpath.join(TEST_ROOT_PATH, "test.txt")
    # arg2 should be a filebuffer
    assert arg2.name == fpath


def test_log_artifacts(mock_client, tmpdir):
    repo = AzureBlobArtifactRepository(TEST_URI, mock_client)

    parentd = tmpdir.mkdir("data")
    subd = parentd.mkdir("subdir")
    parentd.join("a.txt").write("A")
    subd.join("b.txt").write("B")
    subd.join("c.txt").write("C")

    repo.log_artifacts(parentd.strpath)

    mock_client.get_container_client.assert_called_with("container")
    call_list = mock_client.get_container_client().upload_blob.call_args_list

    # Ensure that the order of the calls do not matter
    for call in call_list:
        arg1, arg2 = call[0]
        assert arg1 in [
            posixpath.join(TEST_ROOT_PATH, x) for x in ["a.txt", "subdir/b.txt", "subdir/c.txt"]
        ]
        # arg2 should be a filebuffer
        if arg1.endswith("/a.txt"):
            assert arg2.name == os.path.normpath(parentd.strpath + "/a.txt")
        elif arg1.endswith("/b.txt"):
            assert arg2.name == os.path.normpath(subd.strpath + "/b.txt")
        elif arg1.endswith("/c.txt"):
            assert arg2.name == os.path.normpath(subd.strpath + "/c.txt")
        else:
            # This should be unreachable
            assert False


def test_download_file_artifact(mock_client, tmpdir):
    repo = AzureBlobArtifactRepository(TEST_URI, mock_client)

    mock_client.get_container_client().walk_blobs.return_value = MockBlobList([])

    def create_file(buffer):
        local_path = os.path.basename(buffer.name)
        f = tmpdir.join(local_path)
        f.write("hello world!")

    mock_client.get_container_client().download_blob().readinto.side_effect = create_file

    repo.download_artifacts("test.txt")
    assert os.path.exists(os.path.join(tmpdir.strpath, "test.txt"))
    mock_client.get_container_client().download_blob.assert_called_with(
        posixpath.join(TEST_ROOT_PATH, "test.txt")
    )


def test_download_directory_artifact_succeeds_when_artifact_root_is_not_blob_container_root(
    mock_client, tmpdir
):
    assert TEST_URI is not TEST_BLOB_CONTAINER_ROOT
    repo = AzureBlobArtifactRepository(TEST_URI, mock_client)

    file_path_1 = "file_1"
    file_path_2 = "file_2"

    blob_props_1 = BlobProperties()
    blob_props_1.size = 42
    blob_props_1.name = posixpath.join(TEST_ROOT_PATH, file_path_1)

    blob_props_2 = BlobProperties()
    blob_props_2.size = 42
    blob_props_2.name = posixpath.join(TEST_ROOT_PATH, file_path_2)

    def get_mock_listing(*args, **kwargs):
        """
        Produces a mock listing that only contains content if the
        specified prefix is the artifact root. This allows us to mock
        `list_artifacts` during the `_download_artifacts_into` subroutine
        without recursively listing the same artifacts at every level of the
        directory traversal.
        """
        # pylint: disable=unused-argument
        if posixpath.abspath(kwargs["name_starts_with"]) == posixpath.abspath(TEST_ROOT_PATH):
            return MockBlobList([blob_props_1, blob_props_2])
        else:
            return MockBlobList([])

    def create_file(buffer):
        fname = os.path.basename(buffer.name)
        f = tmpdir.join(fname)
        f.write("hello world!")

    mock_client.get_container_client().walk_blobs.side_effect = get_mock_listing
    mock_client.get_container_client().download_blob().readinto.side_effect = create_file

    # Ensure that the root directory can be downloaded successfully
    repo.download_artifacts("")
    # Ensure that the `mkfile` side effect copied all of the download artifacts into `tmpdir`
    dir_contents = os.listdir(tmpdir.strpath)
    assert file_path_1 in dir_contents
    assert file_path_2 in dir_contents


def test_download_directory_artifact_succeeds_when_artifact_root_is_blob_container_root(
    mock_client, tmpdir
):
    repo = AzureBlobArtifactRepository(TEST_BLOB_CONTAINER_ROOT, mock_client)

    subdir_path = "my_directory"
    dir_prefix = BlobPrefix()
    dir_prefix.name = subdir_path

    file_path_1 = "file_1"
    file_path_2 = "file_2"

    blob_props_1 = BlobProperties()
    blob_props_1.size = 42
    blob_props_1.name = posixpath.join(subdir_path, file_path_1)

    blob_props_2 = BlobProperties()
    blob_props_2.size = 42
    blob_props_2.name = posixpath.join(subdir_path, file_path_2)

    def get_mock_listing(*args, **kwargs):
        """
        Produces a mock listing that only contains content if the specified prefix is the artifact
        root or a relevant subdirectory. This allows us to mock `list_artifacts` during the
        `_download_artifacts_into` subroutine without recursively listing the same artifacts at
        every level of the directory traversal.
        """
        # pylint: disable=unused-argument
        if posixpath.abspath(kwargs["name_starts_with"]) == "/":
            return MockBlobList([dir_prefix])
        if posixpath.abspath(kwargs["name_starts_with"]) == posixpath.abspath(subdir_path):
            return MockBlobList([blob_props_1, blob_props_2])
        else:
            return MockBlobList([])

    def create_file(buffer):
        fname = os.path.basename(buffer.name)
        f = tmpdir.join(fname)
        f.write("hello world!")

    mock_client.get_container_client().walk_blobs.side_effect = get_mock_listing
    mock_client.get_container_client().download_blob().readinto.side_effect = create_file

    # Ensure that the root directory can be downloaded successfully
    repo.download_artifacts("")
    # Ensure that the `mkfile` side effect copied all of the download artifacts into `tmpdir`
    dir_contents = os.listdir(tmpdir.strpath)
    assert file_path_1 in dir_contents
    assert file_path_2 in dir_contents


def test_download_artifact_throws_value_error_when_listed_blobs_do_not_contain_artifact_root_prefix(
    mock_client,
):
    repo = AzureBlobArtifactRepository(TEST_URI, mock_client)

    # Create a "bad blob" with a name that is not prefixed by the root path of the artifact store
    bad_blob_props = BlobProperties()
    bad_blob_props.size = 42
    bad_blob_props.name = "file_path"

    def get_mock_listing(*args, **kwargs):
        """
        Produces a mock listing that only contains content if the
        specified prefix is the artifact root. This allows us to mock
        `list_artifacts` during the `_download_artifacts_into` subroutine
        without recursively listing the same artifacts at every level of the
        directory traversal.
        """
        # pylint: disable=unused-argument
        if posixpath.abspath(kwargs["name_starts_with"]) == posixpath.abspath(TEST_ROOT_PATH):
            # Return a blob that is not prefixed by the root path of the artifact store. This
            # should result in an exception being raised
            return MockBlobList([bad_blob_props])
        else:
            return MockBlobList([])

    mock_client.get_container_client().walk_blobs.side_effect = get_mock_listing

    with pytest.raises(
        MlflowException, match="Azure blob does not begin with the specified artifact path"
    ):
        repo.download_artifacts("")
