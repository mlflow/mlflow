import os
import mock
import pytest

from google.cloud.storage import client as gcs_client

from mlflow.store.artifact_repo import ArtifactRepository, GCSArtifactRepository


@pytest.fixture
def gcs_mock():
    # Make sure that the environment variable isn't set to actually make calls
    old_GOOGLE_APPLICATION_CREDENTIALS = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/dev/null'

    yield mock.MagicMock(autospec=gcs_client)

    if old_GOOGLE_APPLICATION_CREDENTIALS:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = old_GOOGLE_APPLICATION_CREDENTIALS

def test_artifact_uri_factory():
    repo = ArtifactRepository.from_artifact_uri("gs://test_bucket/some/path")
    assert isinstance(repo, GCSArtifactRepository)

def test_list_artifacts_empty(gcs_mock):
    repo = GCSArtifactRepository("gs://test_bucket/some/path", gcs_mock)
    gcs_mock.Client.return_value.get_bucket.return_value\
        .list_blobs.return_value = []
    assert repo.list_artifacts() == []

def test_list_artifacts(gcs_mock):
    repo = GCSArtifactRepository("gs://test_bucket/some/path", gcs_mock)
    mockobj = mock.Mock()
    mockobj.configure_mock(
            name='/some/path/mockeryname',
            f='/mockeryname',
            size=1,
    )
    gcs_mock.Client.return_value.get_bucket.return_value\
        .list_blobs.return_value = [mockobj]
    assert repo.list_artifacts()[0].path == mockobj.f
    assert repo.list_artifacts()[0].file_size == mockobj.size

def test_log_artifact(gcs_mock, tmpdir):
    repo = GCSArtifactRepository("gs://test_bucket/some/path", gcs_mock)

    d = tmpdir.mkdir("data")
    f = d.join("test.txt")
    f.write("hello world!")
    fpath = d + '/test.txt'
    fpath = fpath.strpath

    # This will call isfile on the code path being used,
    # thus testing that it's being called with an actually file path
    gcs_mock.Client.return_value.get_bucket.return_value.blob.return_value \
        .upload_from_filename.side_effect = os.path.isfile
    repo.log_artifact(fpath)

    # A redundant check, but verifying signature since it's all a mock object
    gcs_mock.Client().get_bucket.assert_called_with('test_bucket')
    gcs_mock.Client().get_bucket().blob.assert_called_with('some/path/test.txt')
    gcs_mock.Client().get_bucket().blob().upload_from_filename.assert_called_with(fpath)
