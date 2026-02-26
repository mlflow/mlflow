import json
import os
import posixpath
import tarfile
from datetime import datetime, timezone
from unittest import mock
from unittest.mock import ANY

import botocore.exceptions
import pytest
import requests

from mlflow.entities.multipart_upload import MultipartUploadPart
from mlflow.exceptions import MlflowException, MlflowTraceDataCorrupted
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.optimized_s3_artifact_repo import OptimizedS3ArtifactRepository
from mlflow.store.artifact.s3_artifact_repo import (
    _MAX_CACHE_SECONDS,
    S3ArtifactRepository,
    _cached_get_s3_client,
)

from tests.helper_functions import set_boto_credentials  # noqa: F401


@pytest.fixture
def s3_artifact_root(mock_s3_bucket):
    return f"s3://{mock_s3_bucket}"


@pytest.fixture(params=[True, False])
def s3_artifact_repo(s3_artifact_root, request):
    if request.param:
        return OptimizedS3ArtifactRepository(posixpath.join(s3_artifact_root, "some/path"))
    return S3ArtifactRepository(posixpath.join(s3_artifact_root, "some/path"))


@pytest.fixture(autouse=True)
def reset_cached_get_s3_client():
    _cached_get_s3_client.cache_clear()


def teardown_function():
    if "MLFLOW_S3_UPLOAD_EXTRA_ARGS" in os.environ:
        del os.environ["MLFLOW_S3_UPLOAD_EXTRA_ARGS"]


def test_file_artifact_is_logged_and_downloaded_successfully(s3_artifact_repo, tmp_path):
    file_name = "test.txt"
    file_path = tmp_path / file_name
    file_text = "Hello world!"
    file_path.write_text(file_text)

    s3_artifact_repo.log_artifact(file_path)
    with open(s3_artifact_repo.download_artifacts(file_name)) as f:
        assert f.read() == file_text


def test_file_artifact_is_logged_with_content_metadata(
    s3_artifact_repo, s3_artifact_root, tmp_path
):
    file_name = "test.txt"
    file_path = tmp_path / file_name
    file_text = "Hello world!"
    file_path.write_text(file_text)

    s3_artifact_repo.log_artifact(file_path)

    bucket, _ = s3_artifact_repo.parse_s3_compliant_uri(s3_artifact_root)
    s3_client = s3_artifact_repo._get_s3_client()
    response = s3_client.head_object(Bucket=bucket, Key="some/path/test.txt")
    assert response.get("ContentType") == "text/plain"
    assert response.get("ContentEncoding") == "aws-chunked"


def test_get_s3_client_hits_cache(s3_artifact_root, monkeypatch):
    repo = get_artifact_repository(posixpath.join(s3_artifact_root, "some/path"))
    repo._get_s3_client()
    cache_info = _cached_get_s3_client.cache_info()
    assert cache_info.hits == 0
    assert cache_info.misses == 1
    assert cache_info.currsize == 1

    repo._get_s3_client()
    cache_info = _cached_get_s3_client.cache_info()
    assert cache_info.hits == 1
    assert cache_info.misses == 1
    assert cache_info.currsize == 1

    monkeypatch.setenv("MLFLOW_EXPERIMENTAL_S3_SIGNATURE_VERSION", "s3v2")
    repo._get_s3_client()
    cache_info = _cached_get_s3_client.cache_info()
    assert cache_info.hits == 1
    assert cache_info.misses == 2
    assert cache_info.currsize == 2

    with mock.patch(
        "mlflow.store.artifact.s3_artifact_repo._get_utcnow_timestamp",
        return_value=datetime.now(timezone.utc).timestamp() + _MAX_CACHE_SECONDS,
    ):
        repo._get_s3_client()
    cache_info = _cached_get_s3_client.cache_info()
    assert cache_info.hits == 1
    assert cache_info.misses == 3
    assert cache_info.currsize == 3


@pytest.mark.parametrize(
    ("ignore_tls_env", "verify"), [("0", None), ("1", False), ("true", False), ("false", None)]
)
def test_get_s3_client_verify_param_set_correctly(
    s3_artifact_root, ignore_tls_env, verify, monkeypatch
):
    monkeypatch.setenv("MLFLOW_S3_IGNORE_TLS", ignore_tls_env)
    with mock.patch("boto3.client") as mock_get_s3_client:
        repo = get_artifact_repository(posixpath.join(s3_artifact_root, "some/path"))
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


def test_s3_client_config_set_correctly(s3_artifact_root):
    repo = get_artifact_repository(posixpath.join(s3_artifact_root, "some/path"))
    s3_client = repo._get_s3_client()
    assert s3_client.meta.config.s3.get("addressing_style") == "auto"


def test_s3_creds_passed_to_client(s3_artifact_root):
    with mock.patch("boto3.client") as mock_get_s3_client:
        repo = S3ArtifactRepository(
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


def test_file_artifacts_are_logged_with_content_metadata_in_batch(
    s3_artifact_repo, s3_artifact_root, tmp_path
):
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "nested").mkdir(parents=True)
    subdir_path = str(subdir)
    path_a = subdir / "a.txt"
    path_b = subdir / "b.tar.gz"
    path_c = subdir / "nested" / "c.csv"

    path_a.write_text("A")
    with tarfile.open(path_b, "w:gz") as f:
        f.add(path_a)
    path_c.write_text("col1,col2\n1,3\n2,4\n")

    s3_artifact_repo.log_artifacts(subdir_path)

    bucket, _ = s3_artifact_repo.parse_s3_compliant_uri(s3_artifact_root)
    s3_client = s3_artifact_repo._get_s3_client()

    response_a = s3_client.head_object(Bucket=bucket, Key="some/path/a.txt")
    assert response_a.get("ContentType") == "text/plain"
    assert response_a.get("ContentEncoding") == "aws-chunked"

    response_b = s3_client.head_object(Bucket=bucket, Key="some/path/b.tar.gz")
    assert response_b.get("ContentType") == "application/x-tar"
    assert response_b.get("ContentEncoding") == "gzip,aws-chunked"

    response_c = s3_client.head_object(Bucket=bucket, Key="some/path/nested/c.csv")
    assert response_c.get("ContentType") == "text/csv"
    assert response_c.get("ContentEncoding") == "aws-chunked"


def test_file_and_directories_artifacts_are_logged_and_downloaded_successfully_in_batch(
    s3_artifact_repo, tmp_path
):
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "nested").mkdir(parents=True)
    (subdir / "a.txt").write_text("A")
    (subdir / "b.txt").write_text("B")
    (subdir / "nested" / "c.txt").write_text("C")

    s3_artifact_repo.log_artifacts(str(subdir))

    # Download individual files and verify correctness of their contents
    with open(s3_artifact_repo.download_artifacts("a.txt")) as f:
        assert f.read() == "A"
    with open(s3_artifact_repo.download_artifacts("b.txt")) as f:
        assert f.read() == "B"
    with open(s3_artifact_repo.download_artifacts("nested/c.txt")) as f:
        assert f.read() == "C"

    # Download the nested directory and verify correctness of its contents
    downloaded_dir = s3_artifact_repo.download_artifacts("nested")
    assert os.path.basename(downloaded_dir) == "nested"
    with open(os.path.join(downloaded_dir, "c.txt")) as f:
        assert f.read() == "C"

    # Download the root directory and verify correctness of its contents
    downloaded_dir = s3_artifact_repo.download_artifacts("")
    dir_contents = os.listdir(downloaded_dir)
    assert "nested" in dir_contents
    assert os.path.isdir(os.path.join(downloaded_dir, "nested"))
    assert "a.txt" in dir_contents
    assert "b.txt" in dir_contents


def test_file_and_directories_artifacts_are_logged_and_listed_successfully_in_batch(
    s3_artifact_repo, tmp_path
):
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "nested").mkdir(parents=True)
    (subdir / "a.txt").write_text("A")
    (subdir / "b.txt").write_text("B")
    (subdir / "nested" / "c.txt").write_text("C")

    s3_artifact_repo.log_artifacts(str(subdir))

    root_artifacts_listing = sorted(
        [(f.path, f.is_dir, f.file_size) for f in s3_artifact_repo.list_artifacts()]
    )
    assert root_artifacts_listing == [
        ("a.txt", False, 1),
        ("b.txt", False, 1),
        ("nested", True, None),
    ]

    nested_artifacts_listing = sorted(
        [(f.path, f.is_dir, f.file_size) for f in s3_artifact_repo.list_artifacts("nested")]
    )
    assert nested_artifacts_listing == [("nested/c.txt", False, 1)]


def test_download_directory_artifact_succeeds_when_artifact_root_is_s3_bucket_root(
    s3_artifact_root, tmp_path
):
    file_a_name = "a.txt"
    file_a_text = "A"
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "nested").mkdir(parents=True)
    (subdir / "nested" / file_a_name).write_text(file_a_text)

    repo = get_artifact_repository(s3_artifact_root)
    repo.log_artifacts(str(subdir))

    downloaded_dir_path = repo.download_artifacts("nested")
    assert file_a_name in os.listdir(downloaded_dir_path)
    with open(os.path.join(downloaded_dir_path, file_a_name)) as f:
        assert f.read() == file_a_text


def test_download_file_artifact_succeeds_when_artifact_root_is_s3_bucket_root(
    s3_artifact_root, tmp_path
):
    file_a_name = "a.txt"
    file_a_text = "A"
    file_a_path = tmp_path / file_a_name
    file_a_path.write_text(file_a_text)

    repo = get_artifact_repository(s3_artifact_root)
    repo.log_artifact(file_a_path)

    downloaded_file_path = repo.download_artifacts(file_a_name)
    with open(downloaded_file_path) as f:
        assert f.read() == file_a_text


def test_get_s3_file_upload_extra_args():
    os.environ.setdefault(
        "MLFLOW_S3_UPLOAD_EXTRA_ARGS",
        '{"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": "123456"}',
    )

    parsed_args = S3ArtifactRepository.get_s3_file_upload_extra_args()

    assert parsed_args == {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": "123456"}


def test_get_s3_file_upload_extra_args_env_var_not_present():
    parsed_args = S3ArtifactRepository.get_s3_file_upload_extra_args()

    assert parsed_args is None


def test_get_s3_file_upload_extra_args_invalid_json():
    os.environ.setdefault(
        "MLFLOW_S3_UPLOAD_EXTRA_ARGS", '"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": "123456"}'
    )

    with pytest.raises(json.decoder.JSONDecodeError, match=r".+"):
        S3ArtifactRepository.get_s3_file_upload_extra_args()


def test_delete_artifacts(s3_artifact_repo, tmp_path):
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    nested_path = subdir / "nested"
    nested_path.mkdir()
    path_a = subdir / "a.txt"

    path_a.write_text("A")
    with tarfile.open(str(subdir / "b.tar.gz"), "w:gz") as f:
        f.add(str(path_a))
    (nested_path / "c.csv").write_text("col1,col2\n1,3\n2,4\n")

    s3_artifact_repo.log_artifacts(str(subdir))

    # confirm that artifacts are present
    artifact_file_names = [obj.path for obj in s3_artifact_repo.list_artifacts()]
    assert "a.txt" in artifact_file_names
    assert "b.tar.gz" in artifact_file_names
    assert "nested" in artifact_file_names

    s3_artifact_repo.delete_artifacts()
    assert s3_artifact_repo.list_artifacts() == []


def test_delete_artifacts_single_object(s3_artifact_repo, tmp_path):
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    path_a = subdir / "a.txt"
    path_a.write_text("A")

    s3_artifact_repo.log_artifacts(str(subdir))

    # confirm that artifact is present
    artifact_file_names = [obj.path for obj in s3_artifact_repo.list_artifacts()]
    assert "a.txt" in artifact_file_names

    s3_artifact_repo.delete_artifacts(artifact_path="a.txt")
    assert s3_artifact_repo.list_artifacts() == []


@pytest.mark.parametrize("artifact_path", ["subdir", "subdir/"])
def test_list_and_delete_artifacts_path(s3_artifact_repo, tmp_path, artifact_path):
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    path_a = subdir / "a.txt"
    path_a.write_text("A")

    s3_artifact_repo.log_artifacts(str(subdir), artifact_path.rstrip("/"))

    # confirm that artifact is present
    artifact_file_names = [obj.path for obj in s3_artifact_repo.list_artifacts(artifact_path)]
    assert "subdir/a.txt" in artifact_file_names

    s3_artifact_repo.delete_artifacts(artifact_path=artifact_path)
    assert s3_artifact_repo.list_artifacts(artifact_path) == []
    assert s3_artifact_repo.list_artifacts() == []


@pytest.mark.parametrize(
    ("boto_error_code", "expected_mlflow_error"),
    [
        ("AccessDenied", "PERMISSION_DENIED"),
        ("NoSuchBucket", "RESOURCE_DOES_NOT_EXIST"),
        ("NoSuchKey", "RESOURCE_DOES_NOT_EXIST"),
        ("InvalidAccessKeyId", "UNAUTHENTICATED"),
        ("SignatureDoesNotMatch", "UNAUTHENTICATED"),
    ],
)
def test_list_artifacts_error_handling(s3_artifact_root, boto_error_code, expected_mlflow_error):
    artifact_path = "some/path/"
    s3_repo = S3ArtifactRepository(posixpath.join(s3_artifact_root, artifact_path))

    with mock.patch.object(s3_repo, "_get_s3_client") as mock_client:
        mock_paginator = mock.Mock()
        boto_error_message = "Error message from the client"
        mock_paginator.paginate.side_effect = botocore.exceptions.ClientError(
            {"Error": {"Code": boto_error_code, "Message": boto_error_message}}, "ListObjectsV2"
        )
        mock_client.return_value.get_paginator.return_value = mock_paginator

        with pytest.raises(
            MlflowException, match=f"Failed to list artifacts in {s3_repo.artifact_uri}:"
        ) as exc_info:
            s3_repo.list_artifacts(artifact_path)
        assert exc_info.value.error_code == expected_mlflow_error
        assert boto_error_message in exc_info.value.message


def test_delete_artifacts_pagination(s3_artifact_repo, tmp_path):
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    # The maximum number of objects that can be listed in a single call is 1000
    # https://docs.aws.amazon.com/AmazonS3/latest/API/API_ListObjectsV2.html
    for i in range(1100):
        (subdir / f"{i}.txt").write_text("A")

    s3_artifact_repo.log_artifacts(str(subdir))

    # confirm that artifacts are present
    artifact_file_names = [obj.path for obj in s3_artifact_repo.list_artifacts()]
    for i in range(1100):
        assert f"{i}.txt" in artifact_file_names

    s3_artifact_repo.delete_artifacts()
    assert s3_artifact_repo.list_artifacts() == []


def test_create_multipart_upload(s3_artifact_root):
    repo = get_artifact_repository(posixpath.join(s3_artifact_root, "some/path"))
    create = repo.create_multipart_upload("local_file")

    # confirm that a mpu is created with the correct upload_id
    bucket, _ = repo.parse_s3_compliant_uri(s3_artifact_root)
    s3_client = repo._get_s3_client()
    response = s3_client.list_multipart_uploads(Bucket=bucket)
    uploads = response.get("Uploads")
    assert len(uploads) == 1
    assert uploads[0]["UploadId"] == create.upload_id


def test_complete_multipart_upload(s3_artifact_root):
    repo = get_artifact_repository(posixpath.join(s3_artifact_root, "some/path"))
    local_file = "local_file"
    create = repo.create_multipart_upload(local_file, 2)

    # cannot complete invalid upload
    fake_parts = [
        MultipartUploadPart(part_number=1, etag="fake_etag1"),
        MultipartUploadPart(part_number=2, etag="fake_etag2"),
    ]
    with pytest.raises(botocore.exceptions.ClientError, match=r"InvalidPart"):
        repo.complete_multipart_upload(local_file, create.upload_id, fake_parts)

    # can complete valid upload
    parts = []
    data = b"0" * 5 * 1024 * 1024
    for credential in create.credentials:
        url = credential.url
        response = requests.put(url, data=data)
        parts.append(
            MultipartUploadPart(part_number=credential.part_number, etag=response.headers["ETag"])
        )

    repo.complete_multipart_upload(local_file, create.upload_id, parts)

    # verify upload is completed
    bucket, _ = repo.parse_s3_compliant_uri(s3_artifact_root)
    s3_client = repo._get_s3_client()
    response = s3_client.list_multipart_uploads(Bucket=bucket)
    assert response.get("Uploads") is None


def test_abort_multipart_upload(s3_artifact_root):
    repo = get_artifact_repository(posixpath.join(s3_artifact_root, "some/path"))
    local_file = "local_file"
    create = repo.create_multipart_upload(local_file, 2)

    # cannot abort a non-existing upload
    with pytest.raises(botocore.exceptions.ClientError, match=r"NoSuchUpload"):
        repo.abort_multipart_upload(local_file, "fake_upload_id")

    # can abort the created upload
    repo.abort_multipart_upload(local_file, create.upload_id)

    # verify upload is aborted
    bucket, _ = repo.parse_s3_compliant_uri(s3_artifact_root)
    s3_client = repo._get_s3_client()
    response = s3_client.list_multipart_uploads(Bucket=bucket)
    assert response.get("Uploads") is None


def test_trace_data(s3_artifact_root):
    repo = get_artifact_repository(s3_artifact_root)
    # s3 download_file raises exception directly if the file doesn't exist
    with pytest.raises(Exception, match=r"Trace data not found"):
        repo.download_trace_data()
    repo.upload_trace_data("invalid data")
    with pytest.raises(MlflowTraceDataCorrupted, match=r"Trace data is corrupted for path="):
        repo.download_trace_data()

    mock_trace_data = {"spans": [], "request": {"test": 1}, "response": {"test": 2}}
    repo.upload_trace_data(json.dumps(mock_trace_data))
    assert repo.download_trace_data() == mock_trace_data


def test_bucket_ownership_verification_with_env_var(s3_artifact_repo, tmp_path, monkeypatch):
    file_name = "test.txt"
    file_path = tmp_path / file_name
    file_path.touch()

    monkeypatch.setenv("MLFLOW_S3_EXPECTED_BUCKET_OWNER", "123456789012")
    repo_with_owner = S3ArtifactRepository(s3_artifact_repo.artifact_uri)
    assert repo_with_owner._bucket_owner_params == {"ExpectedBucketOwner": "123456789012"}

    mock_s3 = mock.Mock()

    with mock.patch.object(repo_with_owner, "_get_s3_client", return_value=mock_s3):
        repo_with_owner.log_artifact(file_path)

    mock_s3.upload_file.assert_called_once()
    call_kwargs = mock_s3.upload_file.call_args[1]
    assert "ExtraArgs" in call_kwargs
    assert call_kwargs["ExtraArgs"]["ExpectedBucketOwner"] == "123456789012"


def test_bucket_ownership_verification_without_env_var(s3_artifact_root, tmp_path, monkeypatch):
    file_name = "test.txt"
    file_path = tmp_path / file_name
    file_path.touch()

    monkeypatch.delenv("MLFLOW_S3_EXPECTED_BUCKET_OWNER", raising=False)
    s3_artifact_repo = S3ArtifactRepository(s3_artifact_root)
    assert s3_artifact_repo._bucket_owner_params == {}

    mock_s3 = mock.Mock()

    with mock.patch.object(s3_artifact_repo, "_get_s3_client", return_value=mock_s3):
        s3_artifact_repo.log_artifact(file_path)

    mock_s3.upload_file.assert_called_once()
    call_kwargs = mock_s3.upload_file.call_args[1]
    assert "ExpectedBucketOwner" not in call_kwargs.get("ExtraArgs", {})


def test_bucket_takeover_scenario(s3_artifact_root, tmp_path, monkeypatch):
    """
    Test the bucket takeover scenario where:
    1. A user creates and uses a bucket (e.g., `my-mlflow-artifacts`)
    2. The bucket is deleted
    3. An attacker creates a new bucket with the same name
    4. MLflow continues to use the same bucket URI, unknowingly sending
       artifacts to the attacker's bucket

    This test verifies that when MLFLOW_S3_EXPECTED_BUCKET_OWNER is set, operations
    will fail if the bucket owner doesn't match, preventing the takeover attack.
    """
    file_name = "sensitive_data.txt"
    file_path = tmp_path / file_name
    file_text = "Sensitive information"
    file_path.write_text(file_text)

    monkeypatch.setenv("MLFLOW_S3_EXPECTED_BUCKET_OWNER", "123456789012")
    repo_with_owner = S3ArtifactRepository(s3_artifact_root)

    mock_s3 = mock.Mock()
    mock_s3.upload_file.side_effect = botocore.exceptions.ClientError(
        {
            "Error": {
                "Code": "AccessDenied",
                "Message": "The bucket owner does not match the expected bucket owner",
            }
        },
        "PutObject",
    )

    with mock.patch.object(repo_with_owner, "_get_s3_client", return_value=mock_s3):
        with pytest.raises(
            botocore.exceptions.ClientError,
            match=r"The bucket owner does not match the expected bucket owner",
        ):
            repo_with_owner.log_artifact(file_path)


def test_list_artifacts_with_bucket_owner(s3_artifact_root, tmp_path, monkeypatch):
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    path_a = subdir / "a.txt"
    path_a.touch()

    monkeypatch.setenv("MLFLOW_S3_EXPECTED_BUCKET_OWNER", "123456789012")
    repo_with_owner = S3ArtifactRepository(s3_artifact_root)
    repo_with_owner.log_artifacts(str(subdir))

    mock_s3 = mock.Mock()
    mock_paginator = mock.Mock()
    mock_s3.get_paginator.return_value = mock_paginator
    mock_paginator.paginate.return_value = [{"Contents": [], "CommonPrefixes": []}]

    with mock.patch.object(repo_with_owner, "_get_s3_client", return_value=mock_s3):
        repo_with_owner.list_artifacts()

    mock_paginator.paginate.assert_called_once()
    call_kwargs = mock_paginator.paginate.call_args[1]
    assert "ExpectedBucketOwner" in call_kwargs
    assert call_kwargs["ExpectedBucketOwner"] == "123456789012"


def test_multipart_upload_with_bucket_owner(s3_artifact_root, monkeypatch):
    monkeypatch.setenv("MLFLOW_S3_EXPECTED_BUCKET_OWNER", "123456789012")
    repo_with_owner = S3ArtifactRepository(s3_artifact_root)

    mock_s3 = mock.Mock()
    mock_s3.create_multipart_upload.return_value = {"UploadId": "test-upload-id"}
    mock_s3.generate_presigned_url.return_value = "https://example.com/presigned"

    with mock.patch.object(repo_with_owner, "_get_s3_client", return_value=mock_s3):
        repo_with_owner.create_multipart_upload("local_file", num_parts=2)

    mock_s3.create_multipart_upload.assert_called_once()
    call_kwargs = mock_s3.create_multipart_upload.call_args[1]
    assert "ExpectedBucketOwner" in call_kwargs
    assert call_kwargs["ExpectedBucketOwner"] == "123456789012"
    presigned_calls = mock_s3.generate_presigned_url.call_args_list
    for call in presigned_calls:
        params = call[1]["Params"]
        assert "ExpectedBucketOwner" in params
        assert params["ExpectedBucketOwner"] == "123456789012"


def test_delete_artifacts_with_bucket_owner(s3_artifact_root, tmp_path, monkeypatch):
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    path_a = subdir / "a.txt"
    path_a.touch()

    monkeypatch.setenv("MLFLOW_S3_EXPECTED_BUCKET_OWNER", "123456789012")
    repo_with_owner = S3ArtifactRepository(s3_artifact_root)
    repo_with_owner.log_artifacts(str(subdir))

    mock_s3 = mock.Mock()
    mock_paginator = mock.Mock()
    mock_s3.get_paginator.return_value = mock_paginator
    mock_paginator.paginate.return_value = [
        {"Contents": [{"Key": "some/path/a.txt"}], "CommonPrefixes": []}
    ]

    with mock.patch.object(repo_with_owner, "_get_s3_client", return_value=mock_s3):
        repo_with_owner.delete_artifacts()

    mock_paginator.paginate.assert_called_once()
    paginate_call_kwargs = mock_paginator.paginate.call_args[1]
    assert "ExpectedBucketOwner" in paginate_call_kwargs
    assert paginate_call_kwargs["ExpectedBucketOwner"] == "123456789012"
    mock_s3.delete_objects.assert_called_once()
    delete_call_kwargs = mock_s3.delete_objects.call_args[1]
    assert "ExpectedBucketOwner" in delete_call_kwargs
    assert delete_call_kwargs["ExpectedBucketOwner"] == "123456789012"


def test_create_presigned_upload_url(s3_artifact_root):
    repo = get_artifact_repository(posixpath.join(s3_artifact_root, "some/path"))

    # Generate a presigned upload URL
    presigned_response = repo.create_presigned_upload_url("model.pkl")

    # Verify response structure
    assert presigned_response.presigned_url is not None
    assert isinstance(presigned_response.presigned_url, str)
    assert "X-Amz-Algorithm" in presigned_response.presigned_url
    assert "X-Amz-Credential" in presigned_response.presigned_url
    assert "X-Amz-Signature" in presigned_response.presigned_url
    # Verify the key contains the artifact path
    assert "model.pkl" in presigned_response.presigned_url

    # Verify headers include Content-Type
    assert "Content-Type" in presigned_response.headers
    # .pkl has no standard MIME type, so it should fallback to application/octet-stream
    assert presigned_response.headers["Content-Type"] == "application/octet-stream"


def test_create_presigned_upload_url_with_known_content_type(s3_artifact_root):
    repo = get_artifact_repository(posixpath.join(s3_artifact_root, "some/path"))

    presigned_response = repo.create_presigned_upload_url("data.json")

    assert presigned_response.presigned_url is not None
    assert presigned_response.headers["Content-Type"] == "application/json"


def test_create_presigned_upload_url_nested_path(s3_artifact_root):
    repo = get_artifact_repository(posixpath.join(s3_artifact_root, "some/path"))

    presigned_response = repo.create_presigned_upload_url("models/subdir/model.pkl")

    assert presigned_response.presigned_url is not None
    # Verify the presigned URL references the full nested path
    assert "models" in presigned_response.presigned_url


def test_create_presigned_upload_url_custom_expiration(s3_artifact_root):
    repo = get_artifact_repository(posixpath.join(s3_artifact_root, "some/path"))

    presigned_response = repo.create_presigned_upload_url("model.pkl", expiration=60)

    assert presigned_response.presigned_url is not None
    assert "X-Amz-Expires=60" in presigned_response.presigned_url


def test_create_presigned_upload_url_default_expiration(s3_artifact_root):
    repo = get_artifact_repository(posixpath.join(s3_artifact_root, "some/path"))

    presigned_response = repo.create_presigned_upload_url("model.pkl")

    assert presigned_response.presigned_url is not None
    assert "X-Amz-Expires=900" in presigned_response.presigned_url


def test_create_presigned_upload_url_upload_works(s3_artifact_root, tmp_path):
    repo = get_artifact_repository(posixpath.join(s3_artifact_root, "some/path"))

    # Create a test file
    file_name = "test_upload.txt"
    file_content = "Hello, presigned upload!"
    file_path = tmp_path / file_name
    file_path.write_text(file_content)

    # Get presigned upload URL
    presigned_response = repo.create_presigned_upload_url(file_name)

    # Upload using the presigned URL
    with open(file_path, "rb") as f:
        resp = requests.put(
            presigned_response.presigned_url,
            data=f,
            headers=presigned_response.headers,
        )
    assert resp.status_code == 200

    # Verify the file was uploaded correctly by downloading it
    with open(repo.download_artifacts(file_name)) as f:
        assert f.read() == file_content


def test_create_presigned_upload_url_with_extra_args(s3_artifact_root, monkeypatch):
    monkeypatch.setenv(
        "MLFLOW_S3_UPLOAD_EXTRA_ARGS",
        '{"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": "my-key-id"}',
    )
    repo = S3ArtifactRepository(posixpath.join(s3_artifact_root, "some/path"))

    presigned_response = repo.create_presigned_upload_url("model.pkl")

    assert presigned_response.presigned_url is not None
    # Verify headers include the extra args mapped to HTTP headers
    assert presigned_response.headers.get("x-amz-server-side-encryption") == "aws:kms"
    assert presigned_response.headers.get("x-amz-server-side-encryption-aws-kms-key-id") == "my-key-id"
    # Content-Type should still be present
    assert "Content-Type" in presigned_response.headers


def test_create_presigned_upload_url_without_extra_args(s3_artifact_root, monkeypatch):
    monkeypatch.delenv("MLFLOW_S3_UPLOAD_EXTRA_ARGS", raising=False)
    repo = S3ArtifactRepository(posixpath.join(s3_artifact_root, "some/path"))

    presigned_response = repo.create_presigned_upload_url("model.pkl")

    # Only Content-Type should be in headers
    assert presigned_response.headers == {"Content-Type": "application/octet-stream"}


def test_create_presigned_upload_url_with_bucket_owner(s3_artifact_root, monkeypatch):
    monkeypatch.setenv("MLFLOW_S3_EXPECTED_BUCKET_OWNER", "123456789012")
    repo = S3ArtifactRepository(posixpath.join(s3_artifact_root, "some/path"))

    mock_s3 = mock.Mock()
    mock_s3.generate_presigned_url.return_value = "https://example.com/presigned"

    with mock.patch.object(repo, "_get_s3_client", return_value=mock_s3):
        repo.create_presigned_upload_url("model.pkl")

    mock_s3.generate_presigned_url.assert_called_once()
    call_kwargs = mock_s3.generate_presigned_url.call_args
    params = call_kwargs[1]["Params"]
    assert "ExpectedBucketOwner" in params
    assert params["ExpectedBucketOwner"] == "123456789012"


def test_create_presigned_upload_url_to_dict(s3_artifact_root):
    repo = get_artifact_repository(posixpath.join(s3_artifact_root, "some/path"))

    presigned_response = repo.create_presigned_upload_url("model.pkl")
    response_dict = presigned_response.to_dict()

    assert "presigned_url" in response_dict
    assert "headers" in response_dict
    assert response_dict["presigned_url"] == presigned_response.presigned_url
    assert response_dict["headers"] == presigned_response.headers


def test_create_presigned_upload_url_to_proto(s3_artifact_root):
    repo = get_artifact_repository(posixpath.join(s3_artifact_root, "some/path"))

    presigned_response = repo.create_presigned_upload_url("model.pkl")
    proto = presigned_response.to_proto()

    assert proto.presigned_url == presigned_response.presigned_url
    assert dict(proto.headers) == presigned_response.headers


def test_create_presigned_upload_url_from_dict():
    from mlflow.entities.presigned_upload import CreatePresignedUploadResponse

    d = {
        "presigned_url": "https://example.com/presigned",
        "headers": {"Content-Type": "application/octet-stream"},
    }
    response = CreatePresignedUploadResponse.from_dict(d)
    assert response.presigned_url == "https://example.com/presigned"
    assert response.headers == {"Content-Type": "application/octet-stream"}


def test_create_presigned_upload_url_from_dict_no_headers():
    from mlflow.entities.presigned_upload import CreatePresignedUploadResponse

    d = {"presigned_url": "https://example.com/presigned"}
    response = CreatePresignedUploadResponse.from_dict(d)
    assert response.presigned_url == "https://example.com/presigned"
    assert response.headers == {}


def test_create_presigned_upload_url_from_proto():
    from mlflow.entities.presigned_upload import CreatePresignedUploadResponse
    from mlflow.protos.service_pb2 import CreatePresignedUploadUrl

    proto = CreatePresignedUploadUrl.Response()
    proto.presigned_url = "https://example.com/presigned"
    proto.headers["Content-Type"] = "application/octet-stream"

    response = CreatePresignedUploadResponse.from_proto(proto)
    assert response.presigned_url == "https://example.com/presigned"
    assert response.headers == {"Content-Type": "application/octet-stream"}


def test_get_download_presigned_url(s3_artifact_root, tmp_path):
    repo = get_artifact_repository(posixpath.join(s3_artifact_root, "some/path"))

    # Create and log a test file
    file_name = "test_download.txt"
    file_path = tmp_path / file_name
    file_text = "Hello, presigned download!"
    file_path.write_text(file_text)
    repo.log_artifact(file_path)

    # Get presigned URL
    presigned_response = repo.get_download_presigned_url(file_name)

    # Verify the response structure
    assert presigned_response.url is not None
    assert isinstance(presigned_response.url, str)
    assert presigned_response.headers == {}
    assert presigned_response.file_size == len(file_text)

    # Verify the URL contains expected S3 presigned URL components
    assert "X-Amz-Algorithm" in presigned_response.url
    assert "X-Amz-Credential" in presigned_response.url
    assert "X-Amz-Signature" in presigned_response.url
    assert file_name in presigned_response.url

    # Verify the presigned URL can be used to download the file
    response = requests.get(presigned_response.url)
    assert response.status_code == 200
    assert response.text == file_text


def test_get_download_presigned_url_nested_path(s3_artifact_root, tmp_path):
    repo = get_artifact_repository(posixpath.join(s3_artifact_root, "some/path"))

    # Create a nested directory structure
    nested_dir = tmp_path / "nested" / "subdir"
    nested_dir.mkdir(parents=True)
    file_name = "nested_file.txt"
    file_path = nested_dir / file_name
    file_text = "Nested content"
    file_path.write_text(file_text)

    # Log artifacts preserving directory structure
    repo.log_artifacts(tmp_path / "nested")

    # Get presigned URL for nested file
    presigned_response = repo.get_download_presigned_url("subdir/nested_file.txt")

    # Verify the URL works
    response = requests.get(presigned_response.url)
    assert response.status_code == 200
    assert response.text == file_text


def test_get_download_presigned_url_custom_expiration(s3_artifact_root, tmp_path):
    repo = get_artifact_repository(posixpath.join(s3_artifact_root, "some/path"))

    # Create and log a test file
    file_name = "expiration_test.txt"
    file_path = tmp_path / file_name
    file_path.write_text("test")
    repo.log_artifact(file_path)

    # Get presigned URL with custom expiration (60 seconds)
    presigned_response = repo.get_download_presigned_url(file_name, expiration=60)

    # Verify the URL is generated (we can't easily verify the exact expiration time
    # in the URL, but we can verify it's a valid presigned URL)
    assert presigned_response.url is not None
    assert "X-Amz-Expires=60" in presigned_response.url


def test_get_download_presigned_url_to_dict(s3_artifact_root, tmp_path):
    repo = get_artifact_repository(posixpath.join(s3_artifact_root, "some/path"))

    # Create and log a test file
    file_name = "dict_test.txt"
    file_content = "test"
    file_path = tmp_path / file_name
    file_path.write_text(file_content)
    repo.log_artifact(file_path)

    # Get presigned URL and convert to dict
    presigned_response = repo.get_download_presigned_url(file_name)
    response_dict = presigned_response.to_dict()

    # Verify dict structure
    assert "url" in response_dict
    assert "headers" in response_dict
    assert "file_size" in response_dict
    assert response_dict["url"] == presigned_response.url
    assert response_dict["headers"] == {}
    assert response_dict["file_size"] == len(file_content)


def test_get_download_presigned_url_returns_file_size(s3_artifact_root, tmp_path):
    repo = get_artifact_repository(posixpath.join(s3_artifact_root, "some/path"))

    # Create and log a test file with known content
    file_name = "size_test.txt"
    file_content = "This is test content for file size verification"
    file_path = tmp_path / file_name
    file_path.write_text(file_content)
    repo.log_artifact(file_path)

    # Get presigned URL
    presigned_response = repo.get_download_presigned_url(file_name)

    # Verify file size matches
    assert presigned_response.file_size == len(file_content)
