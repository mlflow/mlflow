# pylint: disable=redefined-outer-name
import os
import posixpath
import pytest
from unittest import mock

from google.cloud.storage import client as gcs_client
from google.auth.exceptions import DefaultCredentialsError

from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.gcs_artifact_repo import GCSArtifactRepository
from tests.helper_functions import mock_method_chain


@pytest.fixture
def gcs_mock():
    # Make sure that the environment variable isn't set to actually make calls
    old_G_APP_CREDS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/dev/null"

    yield mock.MagicMock(autospec=gcs_client)

    if old_G_APP_CREDS:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = old_G_APP_CREDS


def test_artifact_uri_factory():
    repo = get_artifact_repository("gs://test_bucket/some/path")
    assert isinstance(repo, GCSArtifactRepository)


def test_list_artifacts_empty(gcs_mock):
    repo = GCSArtifactRepository("gs://test_bucket/some/path", gcs_mock)
    gcs_mock.Client.return_value.bucket.return_value.list_blobs.return_value = mock.MagicMock()
    assert repo.list_artifacts() == []


def test_list_artifacts(gcs_mock):
    artifact_root_path = "/experiment_id/run_id/"
    repo = GCSArtifactRepository("gs://test_bucket" + artifact_root_path, gcs_mock)

    # mocked bucket/blob structure
    # gs://test_bucket/experiment_id/run_id/
    #  |- file
    #  |- model
    #     |- model.pb

    # mocking a single blob returned by bucket.list_blobs iterator
    # https://googlecloudplatform.github.io/google-cloud-python/latest/storage/buckets.html#google.cloud.storage.bucket.Bucket.list_blobs

    # list artifacts at artifact root level
    obj_mock = mock.Mock()
    file_path = "file"
    obj_mock.configure_mock(name=artifact_root_path + file_path, size=1)

    dir_mock = mock.Mock()
    dir_name = "model"
    dir_mock.configure_mock(prefixes=(artifact_root_path + dir_name + "/",))

    mock_results = mock.MagicMock()
    mock_results.configure_mock(pages=[dir_mock])
    mock_results.__iter__.return_value = [obj_mock]

    gcs_mock.Client.return_value.bucket.return_value.list_blobs.return_value = mock_results

    artifacts = repo.list_artifacts(path=None)

    assert len(artifacts) == 2
    assert artifacts[0].path == file_path
    assert artifacts[0].is_dir is False
    assert artifacts[0].file_size == obj_mock.size
    assert artifacts[1].path == dir_name
    assert artifacts[1].is_dir is True
    assert artifacts[1].file_size is None


@pytest.mark.parametrize("dir_name", ["model", "model/"])
def test_list_artifacts_with_subdir(gcs_mock, dir_name):
    artifact_root_path = "/experiment_id/run_id/"
    repo = GCSArtifactRepository("gs://test_bucket" + artifact_root_path, gcs_mock)

    # mocked bucket/blob structure
    # gs://test_bucket/experiment_id/run_id/
    #  |- model
    #     |- model.pb
    #     |- variables

    # list artifacts at sub directory level
    obj_mock = mock.Mock()
    file_path = posixpath.join(dir_name, "model.pb")
    obj_mock.configure_mock(name=artifact_root_path + file_path, size=1)

    subdir_mock = mock.Mock()
    subdir_name = posixpath.join(dir_name, "variables")
    subdir_mock.configure_mock(prefixes=(artifact_root_path + subdir_name + "/",))

    mock_results = mock.MagicMock()
    mock_results.configure_mock(pages=[subdir_mock])
    mock_results.__iter__.return_value = [obj_mock]

    gcs_mock.Client.return_value.bucket.return_value.list_blobs.return_value = mock_results

    artifacts = repo.list_artifacts(path=dir_name)
    gcs_mock.Client().bucket().list_blobs.assert_called_with(
        prefix=posixpath.join(artifact_root_path[1:], "model/"), delimiter="/"
    )
    assert len(artifacts) == 2
    assert artifacts[0].path == file_path
    assert artifacts[0].is_dir is False
    assert artifacts[0].file_size == obj_mock.size
    assert artifacts[1].path == subdir_name
    assert artifacts[1].is_dir is True
    assert artifacts[1].file_size is None


def test_log_artifact(gcs_mock, tmpdir):
    repo = GCSArtifactRepository("gs://test_bucket/some/path", gcs_mock)

    d = tmpdir.mkdir("data")
    f = d.join("test.txt")
    f.write("hello world!")
    fpath = d + "/test.txt"
    fpath = fpath.strpath

    # This will call isfile on the code path being used,
    # thus testing that it's being called with an actually file path
    def custom_isfile(*args, **kwargs):
        if args:
            return os.path.isfile(args[0])
        return os.path.isfile(kwargs.get("filename"))

    mock_method_chain(
        gcs_mock,
        [
            "Client",
            "bucket",
            "blob",
            "upload_from_filename",
        ],
        side_effect=custom_isfile,
    )
    repo.log_artifact(fpath)

    gcs_mock.Client().bucket.assert_called_with("test_bucket")
    gcs_mock.Client().bucket().blob.assert_called_with(
        "some/path/test.txt", chunk_size=repo._GCS_UPLOAD_CHUNK_SIZE
    )
    gcs_mock.Client().bucket().blob().upload_from_filename.assert_called_with(
        fpath, timeout=repo._GCS_DEFAULT_TIMEOUT
    )


def test_log_artifacts(gcs_mock, tmpdir):
    repo = GCSArtifactRepository("gs://test_bucket/some/path", gcs_mock)

    subd = tmpdir.mkdir("data").mkdir("subdir")
    subd.join("a.txt").write("A")
    subd.join("b.txt").write("B")
    subd.join("c.txt").write("C")

    def custom_isfile(*args, **kwargs):
        if args:
            return os.path.isfile(args[0])
        return os.path.isfile(kwargs.get("filename"))

    mock_method_chain(
        gcs_mock,
        [
            "Client",
            "bucket",
            "blob",
            "upload_from_filename",
        ],
        side_effect=custom_isfile,
    )
    repo.log_artifacts(subd.strpath)

    gcs_mock.Client().bucket.assert_called_with("test_bucket")
    gcs_mock.Client().bucket().blob().upload_from_filename.assert_has_calls(
        [
            mock.call(
                os.path.normpath("%s/a.txt" % subd.strpath), timeout=repo._GCS_DEFAULT_TIMEOUT
            ),
            mock.call(
                os.path.normpath("%s/b.txt" % subd.strpath), timeout=repo._GCS_DEFAULT_TIMEOUT
            ),
            mock.call(
                os.path.normpath("%s/c.txt" % subd.strpath), timeout=repo._GCS_DEFAULT_TIMEOUT
            ),
        ],
        any_order=True,
    )


def test_download_artifacts_calls_expected_gcs_client_methods(gcs_mock, tmpdir):
    repo = GCSArtifactRepository("gs://test_bucket/some/path", gcs_mock)

    def mkfile(fname, **kwargs):
        # pylint: disable=unused-argument
        fname = os.path.basename(fname)
        f = tmpdir.join(fname)
        f.write("hello world!")

    mock_method_chain(
        gcs_mock,
        [
            "Client",
            "bucket",
            "blob",
            "download_to_filename",
        ],
        side_effect=mkfile,
    )

    repo.download_artifacts("test.txt")
    assert os.path.exists(os.path.join(tmpdir.strpath, "test.txt"))
    gcs_mock.Client().bucket.assert_called_with("test_bucket")
    gcs_mock.Client().bucket().blob.assert_called_with(
        "some/path/test.txt", chunk_size=repo._GCS_DOWNLOAD_CHUNK_SIZE
    )
    download_calls = gcs_mock.Client().bucket().blob().download_to_filename.call_args_list
    assert len(download_calls) == 1
    download_path_arg = download_calls[0][0][0]
    assert "test.txt" in download_path_arg


def test_get_anonymous_bucket(gcs_mock):
    gcs_mock.Client.side_effect = DefaultCredentialsError("Test")
    repo = GCSArtifactRepository("gs://test_bucket", gcs_mock)
    repo._get_bucket("gs://test_bucket")
    anon_call_count = gcs_mock.Client.create_anonymous_client.call_count
    assert anon_call_count == 1
    bucket_call_count = gcs_mock.Client.create_anonymous_client.return_value.bucket.call_count
    assert bucket_call_count == 1


def test_download_artifacts_downloads_expected_content(gcs_mock, tmpdir):
    artifact_root_path = "/experiment_id/run_id/"
    repo = GCSArtifactRepository("gs://test_bucket" + artifact_root_path, gcs_mock)

    obj_mock_1 = mock.Mock()
    file_path_1 = "file1"
    obj_mock_1.configure_mock(name=os.path.join(artifact_root_path, file_path_1), size=1)
    obj_mock_2 = mock.Mock()
    file_path_2 = "file2"
    obj_mock_2.configure_mock(name=os.path.join(artifact_root_path, file_path_2), size=1)
    mock_populated_results = mock.MagicMock()
    mock_populated_results.__iter__.return_value = [obj_mock_1, obj_mock_2]

    mock_empty_results = mock.MagicMock()
    mock_empty_results.__iter__.return_value = []

    def get_mock_listing(prefix, **kwargs):
        """
        Produces a mock listing that only contains content if the
        specified prefix is the artifact root. This allows us to mock
        `list_artifacts` during the `_download_artifacts_into` subroutine
        without recursively listing the same artifacts at every level of the
        directory traversal.
        """
        # pylint: disable=unused-argument
        prefix = os.path.join("/", prefix)
        if os.path.abspath(prefix) == os.path.abspath(artifact_root_path):
            return mock_populated_results
        else:
            return mock_empty_results

    def mkfile(fname, **kwargs):
        # pylint: disable=unused-argument
        fname = os.path.basename(fname)
        f = tmpdir.join(fname)
        f.write("hello world!")

    mock_method_chain(
        gcs_mock,
        [
            "Client",
            "bucket",
            "list_blobs",
        ],
        side_effect=get_mock_listing,
    )
    mock_method_chain(
        gcs_mock,
        [
            "Client",
            "bucket",
            "blob",
            "download_to_filename",
        ],
        side_effect=mkfile,
    )

    # Ensure that the root directory can be downloaded successfully
    repo.download_artifacts("")
    # Ensure that the `mkfile` side effect copied all of the download artifacts into `tmpdir`
    dir_contents = os.listdir(tmpdir.strpath)
    assert file_path_1 in dir_contents
    assert file_path_2 in dir_contents


def test_delete_artifacts(gcs_mock):
    experiment_root_path = "/experiment_id/"
    repo = GCSArtifactRepository("gs://test_bucket" + experiment_root_path, gcs_mock)

    def delete_file():
        del obj_mock.name
        del obj_mock.size
        return obj_mock

    obj_mock = mock.Mock()
    run_id_path = experiment_root_path + "run_id/"
    file_path = "file"
    attrs = {"name": run_id_path + file_path, "size": 1, "delete.side_effect": delete_file}
    obj_mock.configure_mock(**attrs)

    def get_mock_listing(prefix, **kwargs):
        """
        Produces a mock listing that only contains content if the
        specified prefix is the artifact root. This allows us to mock
        `list_artifacts` during the `_download_artifacts_into` subroutine
        without recursively listing the same artifacts at every level of the
        directory traversal.
        """

        # pylint: disable=unused-argument
        if hasattr(obj_mock, "name") and hasattr(obj_mock, "size"):
            mock_results = mock.MagicMock()
            mock_results.__iter__.return_value = [obj_mock]
            return mock_results
        else:
            mock_empty_results = mock.MagicMock()
            mock_empty_results.__iter__.return_value = []
            return mock_empty_results

    mock_method_chain(
        gcs_mock,
        [
            "Client",
            "bucket",
            "list_blobs",
        ],
        side_effect=get_mock_listing,
    )

    artifact_file_names = [obj.path for obj in repo.list_artifacts()]
    assert "run_id/file" in artifact_file_names
    repo.delete_artifacts()
    artifact_file_names = [obj.path for obj in repo.list_artifacts()]
    assert not artifact_file_names
