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

    gcs_mock.Client.return_value.get_bucket.return_value\
        .upload_from_filename.side_effect = os.path.isfile(fpath)
    repo.log_artifact(fpath)
