import os
import mock
import pytest

from azure.storage.blob import Blob, BlobPrefix, BlobProperties, BlockBlobService

from mlflow.store.artifact_repo import ArtifactRepository
from mlflow.store.azure_blob_artifact_repo import AzureBlobArtifactRepository


TEST_ROOT_PATH = "some/path"
TEST_URI = "wasbs://container@account.blob.core.windows.net/" + TEST_ROOT_PATH


class MockBlobList(object):
    def __init__(self, items, next_marker=None):
        self.items = items
        self.next_marker = next_marker

    def __iter__(self):
        return iter(self.items)


@pytest.fixture
def mock_client():
    # Make sure that the environment variable isn't set to actually access Azure
    old_env_var = os.environ.get('AZURE_STORAGE_ACCESS_KEY')
    os.environ['AZURE_STORAGE_ACCESS_KEY'] = ''

    yield mock.MagicMock(autospec=BlockBlobService)

    if old_env_var:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = old_env_var


def test_artifact_uri_factory():
    repo = ArtifactRepository.from_artifact_uri(TEST_URI)
    assert isinstance(repo, AzureBlobArtifactRepository)


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

    subd = tmpdir.mkdir("data").mkdir("subdir")
    subd.join("a.txt").write("A")
    subd.join("b.txt").write("B")
    subd.join("c.txt").write("C")
    subd_path = subd.strpath

    repo.log_artifacts(subd_path)

    mock_client.create_blob_from_path.assert_has_calls([
        mock.call("container", TEST_ROOT_PATH + "/a.txt", subd_path + "/a.txt"),
        mock.call("container", TEST_ROOT_PATH + "/b.txt", subd_path + "/b.txt"),
        mock.call("container", TEST_ROOT_PATH + "/c.txt", subd_path + "/c.txt"),
    ], any_order=True)


def test_download_artifacts(mock_client, tmpdir):
    repo = AzureBlobArtifactRepository(TEST_URI, mock_client)

    mock_client.list_blobs.return_value = MockBlobList([])

    def create_file(container, cloud_path, local_path):
        local_path = local_path.replace(tmpdir.strpath, '')
        f = tmpdir.join(local_path)
        f.write("hello world!")
        return f.strpath

    mock_client.get_blob_to_path.side_effect = create_file

    open(repo._download_artifacts_into("test.txt", tmpdir.strpath)).read()
    mock_client.get_blob_to_path.assert_called_with(
        "container", TEST_ROOT_PATH + "/test.txt", tmpdir.strpath + "/test.txt")
