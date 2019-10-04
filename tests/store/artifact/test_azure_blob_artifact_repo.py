import os
import posixpath
import mock
import pytest

from azure.storage.blob import Blob, BlobPrefix, BlobProperties, BlockBlobService

from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.azure_blob_artifact_repo import AzureBlobArtifactRepository


TEST_ROOT_PATH = "some/path"
TEST_BLOB_CONTAINER_ROOT = "wasbs://container@account.blob.core.windows.net/"
TEST_URI = os.path.join(TEST_BLOB_CONTAINER_ROOT, TEST_ROOT_PATH)


class MockBlobList(object):
    def __init__(self, items, next_marker=None):
        self.items = items
        self.next_marker = next_marker

    def __iter__(self):
        return iter(self.items)


@pytest.fixture
def mock_client():
    # Make sure that our environment variable aren't set to actually access Azure
    old_access_key = os.environ.get('AZURE_STORAGE_ACCESS_KEY')
    if old_access_key is not None:
        del os.environ['AZURE_STORAGE_ACCESS_KEY']
    old_conn_string = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
    if old_conn_string is not None:
        del os.environ['AZURE_STORAGE_CONNECTION_STRING']

    yield mock.MagicMock(autospec=BlockBlobService)

    if old_access_key is not None:
        os.environ['AZURE_STORAGE_ACCESS_KEY'] = old_access_key
    if old_conn_string is not None:
        os.environ['AZURE_STORAGE_CONNECTION_STRING'] = old_conn_string


def test_artifact_uri_factory(mock_client):
    # pylint: disable=unused-argument
    # We pass in the mock_client here to clear Azure environment variables, but we don't use it;
    # We do need to set up a fake access key for the code to run though
    os.environ['AZURE_STORAGE_ACCESS_KEY'] = ''
    repo = get_artifact_repository(TEST_URI)
    assert isinstance(repo, AzureBlobArtifactRepository)
    del os.environ['AZURE_STORAGE_ACCESS_KEY']


def test_exception_if_no_env_vars(mock_client):
    # pylint: disable=unused-argument
    # We pass in the mock_client here to clear Azure environment variables, but we don't use it
    with pytest.raises(Exception, match="AZURE_STORAGE_CONNECTION_STRING"):
        AzureBlobArtifactRepository(TEST_URI)


def test_parse_wasbs_uri():
    parse = AzureBlobArtifactRepository.parse_wasbs_uri
    assert parse("wasbs://cont@acct.blob.core.windows.net/path") == ("cont", "acct", "path")
    assert parse("wasbs://cont@acct.blob.core.windows.net") == ("cont", "acct", "")
    assert parse("wasbs://cont@acct.blob.core.windows.net/") == ("cont", "acct", "")
    assert parse("wasbs://cont@acct.blob.core.windows.net/a/b") == ("cont", "acct", "a/b")
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


def test_list_artifacts_empty(mock_client):
    repo = AzureBlobArtifactRepository(TEST_URI, mock_client)
    mock_client.list_blobs.return_value = MockBlobList([])
    assert repo.list_artifacts() == []


def test_list_artifacts(mock_client):
    repo = AzureBlobArtifactRepository(TEST_URI, mock_client)

    # Create some files to return
    dir_prefix = BlobPrefix()
    dir_prefix.name = TEST_ROOT_PATH + "/dir"

    blob_props = BlobProperties()
    blob_props.content_length = 42
    blob = Blob(TEST_ROOT_PATH + "/file", props=blob_props)

    mock_client.list_blobs.return_value = MockBlobList([dir_prefix, blob])

    artifacts = repo.list_artifacts()
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
    fpath = d + '/test.txt'
    fpath = fpath.strpath

    repo.log_artifact(fpath)
    mock_client.create_blob_from_path.assert_called_with(
        "container", TEST_ROOT_PATH + "/test.txt", fpath)


def test_log_artifacts(mock_client, tmpdir):
    repo = AzureBlobArtifactRepository(TEST_URI, mock_client)

    parentd = tmpdir.mkdir("data")
    subd = parentd.mkdir("subdir")
    parentd.join("a.txt").write("A")
    subd.join("b.txt").write("B")
    subd.join("c.txt").write("C")

    repo.log_artifacts(parentd.strpath)

    mock_client.create_blob_from_path.assert_has_calls([
        mock.call("container", TEST_ROOT_PATH + "/a.txt",
                  os.path.normpath(parentd.strpath + "/a.txt")),
        mock.call("container", TEST_ROOT_PATH + "/subdir/b.txt",
                  os.path.normpath(subd.strpath + "/b.txt")),
        mock.call("container", TEST_ROOT_PATH + "/subdir/c.txt",
                  os.path.normpath(subd.strpath + "/c.txt")),
    ], any_order=True)


def test_download_file_artifact(mock_client, tmpdir):
    repo = AzureBlobArtifactRepository(TEST_URI, mock_client)

    mock_client.list_blobs.return_value = MockBlobList([])

    def create_file(container, cloud_path, local_path):
        # pylint: disable=unused-argument
        local_path = os.path.basename(local_path)
        f = tmpdir.join(local_path)
        f.write("hello world!")

    mock_client.get_blob_to_path.side_effect = create_file

    repo.download_artifacts("test.txt")
    assert os.path.exists(os.path.join(tmpdir.strpath, "test.txt"))
    mock_client.get_blob_to_path.assert_called_with(
        "container", TEST_ROOT_PATH + "/test.txt", mock.ANY)


def test_download_directory_artifact_succeeds_when_artifact_root_is_not_blob_container_root(
        mock_client, tmpdir):
    assert TEST_URI is not TEST_BLOB_CONTAINER_ROOT
    repo = AzureBlobArtifactRepository(TEST_URI, mock_client)

    file_path_1 = "file_1"
    file_path_2 = "file_2"

    blob_props_1 = BlobProperties()
    blob_props_1.content_length = 42
    blob_1 = Blob(posixpath.join(TEST_ROOT_PATH, file_path_1), props=blob_props_1)

    blob_props_2 = BlobProperties()
    blob_props_2.content_length = 42
    blob_2 = Blob(posixpath.join(TEST_ROOT_PATH, file_path_2), props=blob_props_2)

    def get_mock_listing(*args, **kwargs):
        """
        Produces a mock listing that only contains content if the
        specified prefix is the artifact root. This allows us to mock
        `list_artifacts` during the `_download_artifacts_into` subroutine
        without recursively listing the same artifacts at every level of the
        directory traversal.
        """
        # pylint: disable=unused-argument
        if posixpath.abspath(kwargs["prefix"]) == posixpath.abspath(TEST_ROOT_PATH):
            return MockBlobList([blob_1, blob_2])
        else:
            return MockBlobList([])

    def create_file(container, cloud_path, local_path):
        # pylint: disable=unused-argument
        fname = os.path.basename(local_path)
        f = tmpdir.join(fname)
        f.write("hello world!")

    mock_client.list_blobs.side_effect = get_mock_listing
    mock_client.get_blob_to_path.side_effect = create_file

    # Ensure that the root directory can be downloaded successfully
    repo.download_artifacts("")
    # Ensure that the `mkfile` side effect copied all of the download artifacts into `tmpdir`
    dir_contents = os.listdir(tmpdir.strpath)
    assert file_path_1 in dir_contents
    assert file_path_2 in dir_contents


def test_download_directory_artifact_succeeds_when_artifact_root_is_blob_container_root(
        mock_client, tmpdir):
    repo = AzureBlobArtifactRepository(TEST_BLOB_CONTAINER_ROOT, mock_client)

    subdir_path = "my_directory"
    dir_prefix = BlobPrefix()
    dir_prefix.name = subdir_path

    file_path_1 = "file_1"
    file_path_2 = "file_2"

    blob_props_1 = BlobProperties()
    blob_props_1.content_length = 42
    blob_1 = Blob(os.path.join(subdir_path, file_path_1), props=blob_props_1)

    blob_props_2 = BlobProperties()
    blob_props_2.content_length = 42
    blob_2 = Blob(os.path.join(subdir_path, file_path_2), props=blob_props_2)

    def get_mock_listing(*args, **kwargs):
        """
        Produces a mock listing that only contains content if the specified prefix is the artifact
        root or a relevant subdirectory. This allows us to mock `list_artifacts` during the
        `_download_artifacts_into` subroutine without recursively listing the same artifacts at
        every level of the directory traversal.
        """
        # pylint: disable=unused-argument
        if posixpath.abspath(kwargs["prefix"]) == "/":
            return MockBlobList([dir_prefix])
        if posixpath.abspath(kwargs["prefix"]) == posixpath.abspath(subdir_path):
            return MockBlobList([blob_1, blob_2])
        else:
            return MockBlobList([])

    def create_file(container, cloud_path, local_path):
        # pylint: disable=unused-argument
        fname = os.path.basename(local_path)
        f = tmpdir.join(fname)
        f.write("hello world!")

    mock_client.list_blobs.side_effect = get_mock_listing
    mock_client.get_blob_to_path.side_effect = create_file

    # Ensure that the root directory can be downloaded successfully
    repo.download_artifacts("")
    # Ensure that the `mkfile` side effect copied all of the download artifacts into `tmpdir`
    dir_contents = os.listdir(tmpdir.strpath)
    assert file_path_1 in dir_contents
    assert file_path_2 in dir_contents


def test_download_artifact_throws_value_error_when_listed_blobs_do_not_contain_artifact_root_prefix(
        mock_client):
    repo = AzureBlobArtifactRepository(TEST_URI, mock_client)

    # Create a "bad blob" with a name that is not prefixed by the root path of the artifact store
    bad_blob_props = BlobProperties()
    bad_blob_props.content_length = 42
    bad_blob = Blob("file_path", props=bad_blob_props)

    def get_mock_listing(*args, **kwargs):
        """
        Produces a mock listing that only contains content if the
        specified prefix is the artifact root. This allows us to mock
        `list_artifacts` during the `_download_artifacts_into` subroutine
        without recursively listing the same artifacts at every level of the
        directory traversal.
        """
        # pylint: disable=unused-argument
        if posixpath.abspath(kwargs["prefix"]) == posixpath.abspath(TEST_ROOT_PATH):
            # Return a blob that is not prefixed by the root path of the artifact store. This
            # should result in an exception being raised
            return MockBlobList([bad_blob])
        else:
            return MockBlobList([])

    mock_client.list_blobs.side_effect = get_mock_listing

    with pytest.raises(MlflowException) as exc:
        repo.download_artifacts("")

    assert "Azure blob does not begin with the specified artifact path" in str(exc)
