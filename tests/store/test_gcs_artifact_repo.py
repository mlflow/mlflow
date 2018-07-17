import os
import mock
import pytest

from google.cloud.storage import client as gcs_client

from mlflow.store.artifact_repo import ArtifactRepository, GCSArtifactRepository


@pytest.fixture
def gcs_mock():
    # Make sure that the environment variable isn't set to actually make calls
    old_G_APP_CREDS = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/dev/null'

    yield mock.MagicMock(autospec=gcs_client)

    if old_G_APP_CREDS:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = old_G_APP_CREDS


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
    gcs_mock.Client.return_value.get_bucket.return_value.blob.return_value\
        .upload_from_filename.side_effect = os.path.isfile
    repo.log_artifact(fpath)

    gcs_mock.Client().get_bucket.assert_called_with('test_bucket')
    gcs_mock.Client().get_bucket().blob\
        .assert_called_with('some/path/test.txt')
    gcs_mock.Client().get_bucket().blob().upload_from_filename\
        .assert_called_with(fpath)


def test_log_artifacts(gcs_mock, tmpdir):
    repo = GCSArtifactRepository("gs://test_bucket/some/path", gcs_mock)

    subd = tmpdir.mkdir("data").mkdir("subdir")
    subd.join("a.txt").write("A")
    subd.join("b.txt").write("B")
    subd.join("c.txt").write("C")

    gcs_mock.Client.return_value.get_bucket.return_value.blob.return_value\
        .upload_from_filename.side_effect = os.path.isfile
    repo.log_artifacts(subd.strpath)

    gcs_mock.Client().get_bucket.assert_called_with('test_bucket')
    gcs_mock.Client().get_bucket().blob().upload_from_filename\
        .assert_has_calls([
            mock.call('%s/a.txt' % subd.strpath),
            mock.call('%s/b.txt' % subd.strpath),
            mock.call('%s/c.txt' % subd.strpath),
        ], any_order=True)


def test_download_artifacts(gcs_mock, tmpdir):
    repo = GCSArtifactRepository("gs://test_bucket/some/path", gcs_mock)

    def mkfile(fname):
        fname = fname.replace(tmpdir.strpath, '')
        f = tmpdir.join(fname)
        f.write("hello world!")
        return f.strpath

    gcs_mock.Client.return_value.get_bucket.return_value.get_blob.return_value\
        .download_to_filename.side_effect = mkfile

    open(repo._download_artifacts_into("test.txt", tmpdir.strpath)).read()
    gcs_mock.Client().get_bucket.assert_called_with('test_bucket')
    gcs_mock.Client().get_bucket().get_blob\
        .assert_called_with('some/path/test.txt')
    gcs_mock.Client().get_bucket().get_blob()\
        .download_to_filename.assert_called_with(tmpdir + "/test.txt")
