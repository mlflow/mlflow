# pylint: disable=redefined-outer-name
import os
import posixpath
from unittest import mock

import pytest
import requests
from google.auth.exceptions import DefaultCredentialsError
from google.cloud.storage import client as gcs_client

from mlflow.entities.multipart_upload import MultipartUploadPart
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.gcs_artifact_repo import GCSArtifactRepository, GCSMPUArguments

from tests.helper_functions import mock_method_chain


@pytest.fixture
def mock_client():
    return mock.MagicMock(autospec=gcs_client.Client)


def test_artifact_uri_factory():
    repo = get_artifact_repository("gs://test_bucket/some/path")
    assert isinstance(repo, GCSArtifactRepository)


def test_list_artifacts_empty(mock_client):
    repo = GCSArtifactRepository("gs://test_bucket/some/path", client=mock_client)
    mock_client.bucket.return_value.list_blobs.return_value = mock.MagicMock()
    assert repo.list_artifacts() == []


def test_custom_gcs_client_used():
    mock_client = mock.MagicMock(autospec=gcs_client.Client)
    repo = GCSArtifactRepository("gs://test_bucket/some/path", client=mock_client)
    mock_client.bucket.return_value.list_blobs.return_value = mock.MagicMock()
    repo.list_artifacts()
    mock_client.bucket.assert_called()


def test_list_artifacts(mock_client):
    artifact_root_path = "/experiment_id/run_id/"
    repo = GCSArtifactRepository("gs://test_bucket" + artifact_root_path, client=mock_client)

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

    mock_client.bucket.return_value.list_blobs.return_value = mock_results

    artifacts = repo.list_artifacts(path=None)

    assert len(artifacts) == 2
    assert artifacts[0].path == file_path
    assert artifacts[0].is_dir is False
    assert artifacts[0].file_size == obj_mock.size
    assert artifacts[1].path == dir_name
    assert artifacts[1].is_dir is True
    assert artifacts[1].file_size is None


@pytest.mark.parametrize("dir_name", ["model", "model/"])
def test_list_artifacts_with_subdir(mock_client, dir_name):
    artifact_root_path = "/experiment_id/run_id/"
    repo = GCSArtifactRepository("gs://test_bucket" + artifact_root_path, client=mock_client)

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

    mock_client.bucket.return_value.list_blobs.return_value = mock_results

    artifacts = repo.list_artifacts(path=dir_name)
    mock_client.bucket().list_blobs.assert_called_with(
        prefix=posixpath.join(artifact_root_path[1:], "model/"), delimiter="/"
    )
    assert len(artifacts) == 2
    assert artifacts[0].path == file_path
    assert artifacts[0].is_dir is False
    assert artifacts[0].file_size == obj_mock.size
    assert artifacts[1].path == subdir_name
    assert artifacts[1].is_dir is True
    assert artifacts[1].file_size is None


def test_log_artifact(mock_client, tmp_path):
    repo = GCSArtifactRepository("gs://test_bucket/some/path", mock_client)

    d = tmp_path.joinpath("data")
    d.mkdir()
    f = d.joinpath("test.txt")
    f.write_text("hello world!")
    fpath = d.joinpath("test.txt")
    fpath = str(fpath)

    # This will call isfile on the code path being used,
    # thus testing that it's being called with an actually file path
    def custom_isfile(*args, **kwargs):
        if args:
            return os.path.isfile(args[0])
        return os.path.isfile(kwargs.get("filename"))

    mock_method_chain(
        mock_client,
        [
            "bucket",
            "blob",
            "upload_from_filename",
        ],
        side_effect=custom_isfile,
    )
    repo.log_artifact(fpath)

    mock_client.bucket.assert_called_with("test_bucket")
    mock_client.bucket().blob.assert_called_with(
        "some/path/test.txt", chunk_size=repo._GCS_UPLOAD_CHUNK_SIZE
    )
    mock_client.bucket().blob().upload_from_filename.assert_called_with(
        fpath, timeout=repo._GCS_DEFAULT_TIMEOUT
    )


def test_log_artifacts(mock_client, tmp_path):
    repo = GCSArtifactRepository("gs://test_bucket/some/path", mock_client)

    data = tmp_path.joinpath("data")
    data.mkdir()
    subd = data.joinpath("subdir")
    subd.mkdir()
    subd.joinpath("a.txt").write_text("A")
    subd.joinpath("b.txt").write_text("B")
    subd.joinpath("c.txt").write_text("C")

    def custom_isfile(*args, **kwargs):
        if args:
            return os.path.isfile(args[0])
        return os.path.isfile(kwargs.get("filename"))

    mock_method_chain(
        mock_client,
        [
            "bucket",
            "blob",
            "upload_from_filename",
        ],
        side_effect=custom_isfile,
    )
    repo.log_artifacts(subd)

    mock_client.bucket.assert_called_with("test_bucket")
    mock_client.bucket().blob().upload_from_filename.assert_has_calls(
        [
            mock.call(os.path.normpath(f"{subd}/a.txt"), timeout=repo._GCS_DEFAULT_TIMEOUT),
            mock.call(os.path.normpath(f"{subd}/b.txt"), timeout=repo._GCS_DEFAULT_TIMEOUT),
            mock.call(os.path.normpath(f"{subd}/c.txt"), timeout=repo._GCS_DEFAULT_TIMEOUT),
        ],
        any_order=True,
    )


def test_download_artifacts_calls_expected_gcs_client_methods(mock_client, tmp_path):
    repo = GCSArtifactRepository("gs://test_bucket/some/path", mock_client)

    def mkfile(fname, **kwargs):
        # pylint: disable=unused-argument
        fname = os.path.basename(fname)
        f = tmp_path.joinpath(fname)
        f.write_text("hello world!")

    mock_method_chain(
        mock_client,
        [
            "bucket",
            "blob",
            "download_to_filename",
        ],
        side_effect=mkfile,
    )

    repo.download_artifacts("test.txt")
    assert tmp_path.joinpath("test.txt").exists()
    mock_client.bucket.assert_called_with("test_bucket")
    mock_client.bucket().blob.assert_called_with(
        "some/path/test.txt", chunk_size=repo._GCS_DOWNLOAD_CHUNK_SIZE
    )
    download_calls = mock_client.bucket().blob().download_to_filename.call_args_list
    assert len(download_calls) == 1
    download_path_arg = download_calls[0][0][0]
    assert "test.txt" in download_path_arg


def test_get_anonymous_bucket():
    with mock.patch("google.cloud.storage", autospec=True) as gcs_mock:
        gcs_mock.Client.side_effect = DefaultCredentialsError("Test")
        repo = GCSArtifactRepository("gs://test_bucket")
        repo._get_bucket("gs://test_bucket")
        anon_call_count = gcs_mock.Client.create_anonymous_client.call_count
        assert anon_call_count == 1
        bucket_call_count = gcs_mock.Client.create_anonymous_client.return_value.bucket.call_count
        assert bucket_call_count == 1


def test_download_artifacts_downloads_expected_content(mock_client, tmp_path):
    artifact_root_path = "/experiment_id/run_id/"
    repo = GCSArtifactRepository("gs://test_bucket" + artifact_root_path, mock_client)

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
        f = tmp_path.joinpath(fname)
        f.write_text("hello world!")

    mock_method_chain(
        mock_client,
        [
            "bucket",
            "list_blobs",
        ],
        side_effect=get_mock_listing,
    )
    mock_method_chain(
        mock_client,
        [
            "bucket",
            "blob",
            "download_to_filename",
        ],
        side_effect=mkfile,
    )

    # Ensure that the root directory can be downloaded successfully
    repo.download_artifacts("")
    # Ensure that the `mkfile` side effect copied all of the download artifacts into `tmpdir`
    dir_contents = os.listdir(tmp_path)
    assert file_path_1 in dir_contents
    assert file_path_2 in dir_contents


def test_delete_artifacts(mock_client):
    experiment_root_path = "/experiment_id/"
    repo = GCSArtifactRepository("gs://test_bucket" + experiment_root_path, mock_client)

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
        mock_client,
        [
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


def test_gcs_mpu_arguments():
    artifact_root_path = "/experiment_id/run_id/"
    repo = GCSArtifactRepository("gs://test_bucket" + artifact_root_path, mock_client)
    requests_session = requests.Session()
    mock_blob = mock.MagicMock()
    mock_blob.name = "experiment_id/run_id/file.txt"
    mock_blob.bucket.name = "test_bucket"
    mock_blob.kms_key_name = None
    mock_blob.user_project = None
    mock_blob._get_upload_arguments.return_value = {}, {}, "application/octet-stream"
    mock_blob._get_transport.return_value = requests_session
    mock_blob.client._connection.get_api_base_url_for_mtls.return_value = "gcs_base_url"
    args = repo._gcs_mpu_arguments("file.txt", mock_blob)
    assert args.transport == requests_session
    assert args.url == "gcs_base_url/test_bucket/experiment_id/run_id/file.txt"
    assert args.headers == {}
    assert args.content_type == "application/octet-stream"


def test_create_multipart_upload(mock_client):
    artifact_root_path = "experiment_id/run_id/"
    bucket_name = "test_bucket"
    file_name = "file.txt"
    gcs_base_url = "gcs_base_url"
    repo = GCSArtifactRepository("gs://test_bucket" + artifact_root_path, mock_client)

    gcs_mpu_arguments_patch = mock.patch(
        "mlflow.store.artifact.gcs_artifact_repo.GCSArtifactRepository._gcs_mpu_arguments",
        return_value=GCSMPUArguments(
            requests.Session(),
            f"{gcs_base_url}/{bucket_name}/{artifact_root_path}/{file_name}",
            {},
            "application/octet-stream",
        ),
    )

    # mock the XML API response of initiate multipart upload
    # see https://cloud.google.com/storage/docs/xml-api/post-object-multipart#example
    upload_id = "some_upload_id"
    resp = mock.Mock(status_code=200)
    resp.text = f"""<?xml version="1.0" encoding="UTF-8"?>
<InitiateMultipartUploadResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Bucket>{bucket_name}</Bucket>
  <Key>{file_name}</Key>
  <UploadId>{upload_id}</UploadId>
</InitiateMultipartUploadResult>"""

    with gcs_mpu_arguments_patch, mock.patch(
        "requests.Session.request", return_value=resp
    ) as request_mock:
        create = repo.create_multipart_upload(
            file_name, num_parts=5, artifact_path=artifact_root_path
        )
        request_mock.assert_called_once()
        args, kwargs = request_mock.call_args
        assert args == (
            "POST",
            f"{gcs_base_url}/{bucket_name}/{artifact_root_path}/{file_name}?uploads",
        )
        assert len(create.credentials) == 5
        assert create.upload_id == upload_id
        assert kwargs["data"] is None


def test_complete_multipart_upload(mock_client):
    artifact_root_path = "experiment_id/run_id/"
    bucket_name = "test_bucket"
    file_name = "file.txt"
    gcs_base_url = "gcs_base_url"
    repo = GCSArtifactRepository("gs://test_bucket" + artifact_root_path, mock_client)

    upload_id = "some_upload_id"
    parts = []
    for part_number in range(1, 3):
        parts.append(MultipartUploadPart(part_number=part_number, etag=f"etag_{part_number}"))

    gcs_mpu_arguments_patch = mock.patch(
        "mlflow.store.artifact.gcs_artifact_repo.GCSArtifactRepository._gcs_mpu_arguments",
        return_value=GCSMPUArguments(
            requests.Session(),
            f"{gcs_base_url}/{bucket_name}/{artifact_root_path}/{file_name}",
            {},
            "application/octet-stream",
        ),
    )

    # See https://cloud.google.com/storage/docs/xml-api/post-object-complete
    expected_payload = (
        b"<CompleteMultipartUpload>"
        b"<Part><PartNumber>1</PartNumber>"
        b"<ETag>etag_1</ETag></Part>"
        b"<Part><PartNumber>2</PartNumber>"
        b"<ETag>etag_2</ETag></Part>"
        b"</CompleteMultipartUpload>"
    )

    resp = mock.Mock(status_code=200)
    with gcs_mpu_arguments_patch, mock.patch(
        "requests.Session.request", return_value=resp
    ) as request_mock:
        repo.complete_multipart_upload(file_name, upload_id, parts, artifact_root_path)
        request_mock.assert_called_once()
        args, kwargs = request_mock.call_args
        assert args == (
            "POST",
            f"{gcs_base_url}/{bucket_name}/{artifact_root_path}/{file_name}?uploadId={upload_id}",
        )
        assert kwargs["data"] == expected_payload


def test_abort_multipart_upload(mock_client):
    artifact_root_path = "experiment_id/run_id/"
    bucket_name = "test_bucket"
    file_name = "file.txt"
    gcs_base_url = "gcs_base_url"
    repo = GCSArtifactRepository("gs://test_bucket" + artifact_root_path, mock_client)

    upload_id = "some_upload_id"
    gcs_mpu_arguments_patch = mock.patch(
        "mlflow.store.artifact.gcs_artifact_repo.GCSArtifactRepository._gcs_mpu_arguments",
        return_value=GCSMPUArguments(
            requests.Session(),
            f"{gcs_base_url}/{bucket_name}/{artifact_root_path}/{file_name}",
            {},
            "application/octet-stream",
        ),
    )

    resp = mock.Mock(status_code=204)
    with gcs_mpu_arguments_patch, mock.patch(
        "requests.Session.request", return_value=resp
    ) as request_mock:
        repo.abort_multipart_upload(file_name, upload_id, artifact_root_path)
        request_mock.assert_called_once()
        args, kwargs = request_mock.call_args
        assert args == (
            "DELETE",
            f"{gcs_base_url}/{bucket_name}/{artifact_root_path}/{file_name}?uploadId={upload_id}",
        )
        assert kwargs["data"] is None
