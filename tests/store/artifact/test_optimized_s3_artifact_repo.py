import os
import posixpath
from datetime import datetime
from unittest import mock
from unittest.mock import ANY

import pytest

from mlflow.protos.service_pb2 import FileInfo
from mlflow.store.artifact.optimized_s3_artifact_repo import OptimizedS3ArtifactRepository
from mlflow.store.artifact.s3_artifact_repo import (
    _MAX_CACHE_SECONDS,
    _cached_get_s3_client,
)

from tests.helper_functions import set_boto_credentials  # noqa: F401

S3_REPOSITORY_MODULE = "mlflow.store.artifact.optimized_s3_artifact_repo"
S3_ARTIFACT_REPOSITORY = f"{S3_REPOSITORY_MODULE}.OptimizedS3ArtifactRepository"


@pytest.fixture
def s3_artifact_root(mock_s3_bucket):
    return f"s3://{mock_s3_bucket}"


@pytest.fixture(autouse=True)
def reset_cached_get_s3_client():
    _cached_get_s3_client.cache_clear()


def test_get_s3_client_hits_cache(s3_artifact_root, monkeypatch):
    with mock.patch("boto3.client") as mock_get_s3_client:
        s3_client_mock = mock.Mock()
        mock_get_s3_client.return_value = s3_client_mock
        s3_client_mock.get_bucket_location.return_value = {"LocationConstraint": "us-west-2"}

        # pylint: disable=no-value-for-parameter
        repo = OptimizedS3ArtifactRepository(posixpath.join(s3_artifact_root, "some/path"))

        # We get the s3 client once during initialization to get the bucket region name
        cache_info = _cached_get_s3_client.cache_info()
        assert cache_info.hits == 0
        assert cache_info.misses == 1
        assert cache_info.currsize == 1

        # When the s3 client is fetched via class method, it is called with the region name
        repo._get_s3_client()
        cache_info = _cached_get_s3_client.cache_info()
        assert cache_info.hits == 0
        assert cache_info.misses == 2
        assert cache_info.currsize == 2

        # A second fetch via class method leads to cache hit
        repo._get_s3_client()
        cache_info = _cached_get_s3_client.cache_info()
        assert cache_info.hits == 1
        assert cache_info.misses == 2
        assert cache_info.currsize == 2

        monkeypatch.setenv("MLFLOW_EXPERIMENTAL_S3_SIGNATURE_VERSION", "s3v2")
        repo._get_s3_client()
        cache_info = _cached_get_s3_client.cache_info()
        assert cache_info.hits == 1
        assert cache_info.misses == 3
        assert cache_info.currsize == 3

        with mock.patch(
            "mlflow.store.artifact.s3_artifact_repo._get_utcnow_timestamp",
            return_value=datetime.utcnow().timestamp() + _MAX_CACHE_SECONDS,
        ):
            repo._get_s3_client()
        cache_info = _cached_get_s3_client.cache_info()
        assert cache_info.hits == 1
        assert cache_info.misses == 4
        assert cache_info.currsize == 4


@pytest.mark.parametrize(
    ("ignore_tls_env", "verify"), [("0", None), ("1", False), ("true", False), ("false", None)]
)
def test_get_s3_client_verify_param_set_correctly(
    s3_artifact_root, ignore_tls_env, verify, monkeypatch
):
    monkeypatch.setenv("MLFLOW_S3_IGNORE_TLS", ignore_tls_env)
    with mock.patch("boto3.client") as mock_get_s3_client:
        repo = OptimizedS3ArtifactRepository(posixpath.join(s3_artifact_root, "some/path"))
        repo._get_s3_client()
        mock_get_s3_client.assert_called_with(
            "s3",
            config=ANY,
            endpoint_url=ANY,
            verify=verify,
            aws_access_key_id=None,
            aws_secret_access_key=None,
            aws_session_token=None,
            region_name=ANY,
        )


def test_get_s3_client_region_name_set_correctly(s3_artifact_root):
    region_name = "us_random_region_42"
    with mock.patch("boto3.client") as mock_get_s3_client:
        s3_client_mock = mock.Mock()
        mock_get_s3_client.return_value = s3_client_mock
        s3_client_mock.get_bucket_location.return_value = {"LocationConstraint": region_name}

        repo = OptimizedS3ArtifactRepository(posixpath.join(s3_artifact_root, "some/path"))
        repo._get_s3_client()

        mock_get_s3_client.assert_called_with(
            "s3",
            config=ANY,
            endpoint_url=ANY,
            verify=None,
            aws_access_key_id=None,
            aws_secret_access_key=None,
            aws_session_token=None,
            region_name=region_name,
        )


def test_s3_client_config_set_correctly(s3_artifact_root):
    repo = OptimizedS3ArtifactRepository(posixpath.join(s3_artifact_root, "some/path"))
    s3_client = repo._get_s3_client()
    assert s3_client.meta.config.s3.get("addressing_style") == "path"


def test_s3_creds_passed_to_client(s3_artifact_root):
    with mock.patch("boto3.client") as mock_get_s3_client:
        repo = OptimizedS3ArtifactRepository(
            s3_artifact_root,
            access_key_id="my-id",
            secret_access_key="my-key",
            session_token="my-session-token",
        )
        repo._get_s3_client()
        mock_get_s3_client.assert_called_with(
            "s3",
            config=ANY,
            endpoint_url=ANY,
            verify=None,
            aws_access_key_id="my-id",
            aws_secret_access_key="my-key",
            aws_session_token="my-session-token",
            region_name=ANY,
        )


def test_log_artifacts_in_parallel_when_necessary(s3_artifact_root, mock_s3_bucket, tmp_path):
    repo = OptimizedS3ArtifactRepository(posixpath.join(s3_artifact_root, "some/path"))

    file_a_name = "a.txt"
    file_a_text = "A"
    file_a_path = os.path.join(tmp_path, file_a_name)
    with open(file_a_path, "w") as f:
        f.write(file_a_text)

    with mock.patch(f"{S3_REPOSITORY_MODULE}._MULTIPART_UPLOAD_CHUNK_SIZE", 0), mock.patch(
        f"{S3_ARTIFACT_REPOSITORY}._multipart_upload", return_value=None
    ) as multipart_upload_mock:
        repo.log_artifacts(tmp_path)
        multipart_upload_mock.assert_called_once_with(ANY, ANY, mock_s3_bucket, "some/path/a.txt")


@pytest.mark.parametrize(
    ("file_size", "is_parallel_download"),
    [(None, False), (100, False), (499_999_999, False), (500_000_000, True)],
)
def test_download_file_in_parallel_when_necessary(
    s3_artifact_root, file_size, is_parallel_download
):
    repo = OptimizedS3ArtifactRepository(posixpath.join(s3_artifact_root, "some/path"))
    remote_file_path = "file_1.txt"
    list_artifacts_result = (
        [FileInfo(path=remote_file_path, is_dir=False, file_size=file_size)] if file_size else []
    )
    with mock.patch(
        f"{S3_ARTIFACT_REPOSITORY}.list_artifacts",
        return_value=list_artifacts_result,
    ), mock.patch(
        f"{S3_ARTIFACT_REPOSITORY}._download_from_cloud", return_value=None
    ) as download_mock, mock.patch(
        f"{S3_ARTIFACT_REPOSITORY}._parallelized_download_from_cloud", return_value=None
    ) as parallel_download_mock:
        repo.download_artifacts("")
        if is_parallel_download:
            parallel_download_mock.assert_called_with(file_size, remote_file_path, ANY)
        else:
            download_mock.assert_called()
