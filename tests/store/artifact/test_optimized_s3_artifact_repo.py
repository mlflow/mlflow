import os
import posixpath
import threading
import uuid
from datetime import datetime, timezone
from unittest import mock
from unittest.mock import ANY

import boto3
import pytest
import requests
from opentelemetry import context as otel_context

from mlflow.protos.service_pb2 import FileInfo
from mlflow.store.artifact.b2_artifact_repo import B2ArtifactRepository
from mlflow.store.artifact.optimized_s3_artifact_repo import OptimizedS3ArtifactRepository
from mlflow.store.artifact.s3_artifact_repo import (
    _MAX_CACHE_SECONDS,
    _cached_get_s3_client,
)
from mlflow.tracing.export.async_export_queue import Task

from tests.helper_functions import set_boto_credentials  # noqa: F401

S3_REPOSITORY_MODULE = "mlflow.store.artifact.optimized_s3_artifact_repo"
S3_ARTIFACT_REPOSITORY = f"{S3_REPOSITORY_MODULE}.OptimizedS3ArtifactRepository"
DEFAULT_REGION_NAME = "us_random_region"


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
        s3_client_mock.head_bucket.return_value = {"BucketRegion": "us-west-2"}

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
            return_value=datetime.now(timezone.utc).timestamp() + _MAX_CACHE_SECONDS,
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
        s3_client_mock = mock.Mock()
        s3_client_mock.head_bucket.return_value = {
            "ResponseMetadata": {
                "HTTPHeaders": {"x-amz-bucket-region": DEFAULT_REGION_NAME},
            }
        }
        mock_get_s3_client.return_value = s3_client_mock
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


@pytest.mark.parametrize("client_throws", [True, False])
def test_get_s3_client_region_name_set_correctly(s3_artifact_root, client_throws):
    region_name = "us_random_region_42"
    with mock.patch("boto3.client") as mock_get_s3_client:
        from botocore.exceptions import ClientError

        s3_client_mock = mock.Mock()
        mock_get_s3_client.return_value = s3_client_mock
        if client_throws:
            error = ClientError(
                {
                    "Error": {"Code": "403", "Message": "Forbidden"},
                    "ResponseMetadata": {
                        "HTTPHeaders": {"x-amz-bucket-region": region_name},
                    },
                },
                "head_bucket",
            )
            s3_client_mock.head_bucket.side_effect = error
        else:
            s3_client_mock.head_bucket.return_value = {"BucketRegion": region_name}

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


def test_get_s3_client_region_name_set_correctly_with_non_throwing_response(s3_artifact_root):
    region_name = "us_random_region_42"
    with mock.patch("boto3.client") as mock_get_s3_client:
        s3_client_mock = mock.Mock()
        mock_get_s3_client.return_value = s3_client_mock
        s3_client_mock.head_bucket.return_value = {
            "ResponseMetadata": {
                "HTTPHeaders": {"x-amz-bucket-region": region_name},
            }
        }
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


def test_get_region_name_client_error_without_region_header():
    from botocore.exceptions import ClientError

    from mlflow.exceptions import MlflowException

    with mock.patch("boto3.client") as mock_get_s3_client:
        s3_client_mock = mock.Mock()
        mock_get_s3_client.return_value = s3_client_mock
        error = ClientError(
            {
                "Error": {"Code": "403", "Message": "Forbidden"},
                "ResponseMetadata": {"HTTPHeaders": {}},
            },
            "head_bucket",
        )
        s3_client_mock.head_bucket.side_effect = error

        with pytest.raises(MlflowException, match="AWS_DEFAULT_REGION"):
            OptimizedS3ArtifactRepository("s3://mock-bucket/some/path")

        s3_client_mock.head_bucket.assert_called_once()


def test_s3_client_config_set_correctly(s3_artifact_root):
    repo = OptimizedS3ArtifactRepository(posixpath.join(s3_artifact_root, "some/path"))
    s3_client = repo._get_s3_client()
    assert s3_client.meta.config.s3.get("addressing_style") == "auto"


def test_s3_creds_passed_to_client(s3_artifact_root):
    with mock.patch("boto3.client") as mock_get_s3_client:
        s3_client_mock = mock.Mock()
        s3_client_mock.head_bucket.return_value = {
            "ResponseMetadata": {
                "HTTPHeaders": {"x-amz-bucket-region": DEFAULT_REGION_NAME},
            }
        }
        mock_get_s3_client.return_value = s3_client_mock
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


def test_log_artifacts_in_parallel_when_necessary(
    s3_artifact_root, mock_s3_bucket, tmp_path, monkeypatch
):
    repo = OptimizedS3ArtifactRepository(posixpath.join(s3_artifact_root, "some/path"))

    file_a_name = "a.txt"
    file_a_text = "A"
    file_a_path = os.path.join(tmp_path, file_a_name)
    with open(file_a_path, "w") as f:
        f.write(file_a_text)

    monkeypatch.setenv("MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE", "0")
    with mock.patch(
        f"{S3_ARTIFACT_REPOSITORY}._multipart_upload", return_value=None
    ) as multipart_upload_mock:
        repo.log_artifacts(tmp_path)
        multipart_upload_mock.assert_called_once_with(ANY, ANY, mock_s3_bucket, "some/path/a.txt")


@pytest.mark.parametrize(
    ("repository_extra_args", "environment_extra_args", "expected_extra_args"),
    [
        (
            {"ServerSideEncryption": "AES256"},
            None,
            {"ServerSideEncryption": "AES256"},
        ),
        (
            {"ServerSideEncryption": "AES256"},
            (
                '{"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": "key-id", '
                '"SSEKMSEncryptionContext": "context", "BucketKeyEnabled": true}'
            ),
            {
                "ServerSideEncryption": "aws:kms",
                "SSEKMSKeyId": "key-id",
                "SSEKMSEncryptionContext": "context",
                "BucketKeyEnabled": True,
            },
        ),
    ],
)
def test_multipart_upload_passes_s3_upload_encryption_args(
    s3_artifact_root,
    tmp_path,
    monkeypatch,
    repository_extra_args,
    environment_extra_args,
    expected_extra_args,
):
    if environment_extra_args:
        monkeypatch.setenv("MLFLOW_S3_UPLOAD_EXTRA_ARGS", environment_extra_args)
    s3_client = mock.Mock()
    s3_client.create_multipart_upload.return_value = {"UploadId": "upload-id"}
    local_file = tmp_path / "artifact"
    local_file.touch()
    repo = OptimizedS3ArtifactRepository(
        posixpath.join(s3_artifact_root, "path"),
        s3_upload_extra_args=repository_extra_args,
    )

    repo._multipart_upload(s3_client, local_file, "bucket", "path/artifact")

    s3_client.create_multipart_upload.assert_called_once_with(
        Bucket="bucket",
        Key="path/artifact",
        **expected_extra_args,
    )


def test_multipart_upload_does_not_pass_non_encryption_upload_args(
    s3_artifact_root, tmp_path, monkeypatch
):
    monkeypatch.setenv(
        "MLFLOW_S3_UPLOAD_EXTRA_ARGS",
        '{"ServerSideEncryption": "AES256", "ACL": "bucket-owner-full-control"}',
    )
    s3_client = mock.Mock()
    s3_client.create_multipart_upload.return_value = {"UploadId": "upload-id"}
    local_file = tmp_path / "artifact"
    local_file.touch()
    repo = OptimizedS3ArtifactRepository(posixpath.join(s3_artifact_root, "path"))

    repo._multipart_upload(s3_client, local_file, "bucket", "path/artifact")

    s3_client.create_multipart_upload.assert_called_once_with(
        Bucket="bucket",
        Key="path/artifact",
        ServerSideEncryption="AES256",
    )


@pytest.mark.parametrize(
    ("file_size", "is_parallel_download"),
    [(None, False), (100, False), (500 * 1024**2 - 1, False), (500 * 1024**2, True)],
)
def test_download_file_in_parallel_when_necessary(
    s3_artifact_root, file_size, is_parallel_download
):
    repo = OptimizedS3ArtifactRepository(posixpath.join(s3_artifact_root, "some/path"))
    remote_file_path = "file_1.txt"
    list_artifacts_result = [FileInfo(path=remote_file_path, is_dir=False, file_size=file_size)]
    with (
        mock.patch(
            f"{S3_ARTIFACT_REPOSITORY}.list_artifacts",
            return_value=list_artifacts_result,
        ),
        mock.patch(
            f"{S3_ARTIFACT_REPOSITORY}._download_from_cloud", return_value=None
        ) as download_mock,
        mock.patch(
            f"{S3_ARTIFACT_REPOSITORY}._parallelized_download_from_cloud", return_value=None
        ) as parallel_download_mock,
    ):
        repo.download_artifacts("")
        if is_parallel_download:
            parallel_download_mock.assert_called_with(file_size, remote_file_path, ANY)
        else:
            download_mock.assert_called()


def test_refresh_credentials():
    with (
        mock.patch(
            "mlflow.store.artifact.optimized_s3_artifact_repo._get_s3_client"
        ) as mock_get_s3_client,
        mock.patch(
            "mlflow.store.artifact.optimized_s3_artifact_repo.OptimizedS3ArtifactRepository._get_region_name"
        ) as mock_get_region_name,
    ):
        s3_client_mock = mock.Mock()
        mock_get_s3_client.return_value = s3_client_mock
        resp = requests.Response()
        resp.status_code = 401
        err = requests.HTTPError(response=resp)
        s3_client_mock.download_file.side_effect = err
        mock_get_region_name.return_value = "us-west-2"

        def credential_refresh_def():
            return {
                "access_key_id": "my-id-2",
                "secret_access_key": "my-key-2",
                "session_token": "my-session-2",
                "s3_upload_extra_args": {},
            }

        repo = OptimizedS3ArtifactRepository(
            "s3://my_bucket/my_path",
            access_key_id="my-id-1",
            secret_access_key="my-key-1",
            session_token="my-session-1",
            credential_refresh_def=credential_refresh_def,
        )
        with pytest.raises(requests.HTTPError, match=r".*", check=lambda e: e == err):
            repo._download_from_cloud("file_1.txt", "local_path")

        mock_get_s3_client.assert_any_call(
            addressing_style=None,
            access_key_id="my-id-1",
            secret_access_key="my-key-1",
            session_token="my-session-1",
            region_name="us-west-2",
            s3_endpoint_url=None,
        )

        mock_get_s3_client.assert_any_call(
            addressing_style=None,
            access_key_id="my-id-2",
            secret_access_key="my-key-2",
            session_token="my-session-2",
            region_name="us-west-2",
            s3_endpoint_url=None,
        )


def test_trace_uploads_put_objects_with_instrumentation_suppressed(mock_s3_bucket, monkeypatch):
    monkeypatch.setenv("MLFLOW_S3_UPLOAD_EXTRA_ARGS", '{"Metadata": {"origin": "mlflow"}}')
    monkeypatch.setenv("MLFLOW_S3_EXPECTED_BUCKET_OWNER", "123456789012")
    repo = B2ArtifactRepository(
        f"b2://{mock_s3_bucket}@s3.us-west-004.backblazeb2.com/some/path",
        s3_upload_extra_args={"ServerSideEncryption": "AES256"},
    )
    # B2's own client targets Backblaze; this one is served by moto.
    s3_client = boto3.client("s3")
    attachment_id = str(uuid.uuid4())
    # Trace export runs uploads in a Task that suppresses instrumentation through a contextvar.
    # Record whether each PutObject still sees it, and the bucket owner it sends.
    calls = []
    s3_client.meta.events.register(
        "before-call.s3.PutObject",
        lambda params, **kwargs: calls.append((
            otel_context.get_value(otel_context._SUPPRESS_INSTRUMENTATION_KEY),
            params["headers"].get("x-amz-expected-bucket-owner"),
        )),
    )

    with mock.patch.object(repo, "_get_s3_client", return_value=s3_client) as mock_get_s3_client:
        Task(handler=repo.upload_trace_data, args=('{"spans": []}',)).handle()
        Task(handler=repo.upload_attachment, args=(attachment_id, b"\x89PNG data")).handle()
        assert repo.download_trace_data() == {"spans": []}
        assert repo.download_trace_attachment(attachment_id) == b"\x89PNG data"

    mock_get_s3_client.assert_called()
    assert calls == [(True, "123456789012"), (True, "123456789012")]
    trace_data = s3_client.get_object(Bucket=mock_s3_bucket, Key="some/path/traces.json")
    assert trace_data["ContentType"] == "application/json"
    assert trace_data["Metadata"] == {"origin": "mlflow"}
    assert trace_data["ServerSideEncryption"] == "AES256"
    attachment = s3_client.get_object(
        Bucket=mock_s3_bucket, Key=f"some/path/attachments/{attachment_id}"
    )
    assert attachment["Metadata"] == {"origin": "mlflow"}
    assert attachment["ServerSideEncryption"] == "AES256"


def test_upload_attachment_uses_multipart_on_calling_thread(mock_s3_bucket, monkeypatch):
    # moto stores multipart bodies with aws-chunked checksum framing unless this is set.
    monkeypatch.setenv("AWS_REQUEST_CHECKSUM_CALCULATION", "when_required")
    repo = B2ArtifactRepository(f"b2://{mock_s3_bucket}@s3.us-west-004.backblazeb2.com/some/path")
    s3_client = boto3.client("s3")
    calls = {}
    for op in ("CreateMultipartUpload", "UploadPart", "CompleteMultipartUpload"):
        s3_client.meta.events.register(
            f"before-call.s3.{op}",
            lambda params, op=op, **kwargs: calls.setdefault(op, []).append((
                threading.current_thread() is threading.main_thread(),
                otel_context.get_value(otel_context._SUPPRESS_INSTRUMENTATION_KEY),
            )),
        )
    attachment_id = str(uuid.uuid4())
    body = os.urandom(8 * 1024 * 1024 + 1024)

    with mock.patch.object(repo, "_get_s3_client", return_value=s3_client):
        Task(handler=repo.upload_attachment, args=(attachment_id, body)).handle()
        assert repo.download_trace_attachment(attachment_id) == body

    assert calls == {
        "CreateMultipartUpload": [(True, True)],
        "UploadPart": [(True, True)] * 2,
        "CompleteMultipartUpload": [(True, True)],
    }


def test_upload_trace_data_drops_extra_args_put_object_rejects(
    s3_artifact_root, mock_s3_bucket, monkeypatch
):
    # s3transfer consumes these for multipart and strips them before PutObject.
    monkeypatch.setenv(
        "MLFLOW_S3_UPLOAD_EXTRA_ARGS",
        '{"ChecksumType": "FULL_OBJECT", "MpuObjectSize": 13, "Metadata": {"origin": "mlflow"}}',
    )
    repo = OptimizedS3ArtifactRepository(posixpath.join(s3_artifact_root, "some/path"))

    repo.upload_trace_data('{"spans": []}')

    obj = repo._get_s3_client().get_object(Bucket=mock_s3_bucket, Key="some/path/traces.json")
    assert obj["Metadata"] == {"origin": "mlflow"}


def test_upload_trace_data_retries_with_refreshed_credentials():
    expired_client = mock.Mock(**{"upload_fileobj.side_effect": Exception("ExpiredToken")})
    refreshed_client = mock.Mock()
    with mock.patch.object(
        OptimizedS3ArtifactRepository, "_get_region_name", return_value="us-west-2"
    ) as mock_get_region_name:
        repo = OptimizedS3ArtifactRepository(
            "s3://my_bucket/my_path",
            credential_refresh_def=lambda: {
                "access_key_id": "my-id-2",
                "secret_access_key": "my-key-2",
                "session_token": "my-session-2",
                "s3_upload_extra_args": {},
            },
        )
    mock_get_region_name.assert_called_once()

    with mock.patch.object(
        repo, "_get_s3_client", side_effect=[expired_client, refreshed_client]
    ) as mock_get_s3_client:
        repo.upload_trace_data('{"spans": []}')

    assert mock_get_s3_client.call_count == 2
    expired_client.upload_fileobj.assert_called_once()
    refreshed_client.upload_fileobj.assert_called_once()
    args, kwargs = refreshed_client.upload_fileobj.call_args
    assert args[0].read() == b'{"spans": []}'
    assert args[1:] == ("my_bucket", "my_path/traces.json")
    assert kwargs["ExtraArgs"] == {"ContentType": "application/json"}
    assert kwargs["Config"].use_threads is False
