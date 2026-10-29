from unittest import mock

import click
import pytest

from mlflow.store.artifact.artifact_repo import MultipartDownloadMixin, MultipartUploadMixin
from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.utils.server_cli_utils import artifacts_presigned_config_validation


class _PresignedArtifactRepository(MultipartUploadMixin, MultipartDownloadMixin):
    def create_multipart_upload(self, local_file, num_parts, artifact_path=None):
        raise NotImplementedError

    def complete_multipart_upload(self, local_file, upload_id, parts, artifact_path=None):
        raise NotImplementedError

    def abort_multipart_upload(self, local_file, upload_id, artifact_path=None):
        raise NotImplementedError

    def get_download_presigned_url(self, artifact_path, expiration=300):
        raise NotImplementedError


def test_presigned_only_requires_artifact_serving():
    with pytest.raises(click.UsageError, match="requires artifact serving"):
        artifacts_presigned_config_validation(
            True,
            serve_artifacts=False,
            artifacts_destination="s3://bucket",
        )


def test_presigned_only_requires_supported_artifact_destination():
    with (
        mock.patch(
            "mlflow.store.artifact.artifact_repository_registry.get_artifact_repository",
            return_value=mock.Mock(spec=[]),
        ) as mock_get_artifact_repository,
        pytest.raises(click.UsageError, match="presigned uploads and presigned downloads"),
    ):
        artifacts_presigned_config_validation(
            True,
            serve_artifacts=True,
            artifacts_destination="./mlartifacts",
        )

    mock_get_artifact_repository.assert_called_once_with("./mlartifacts")


def test_presigned_only_accepts_supported_artifact_destination():
    with mock.patch(
        "mlflow.store.artifact.artifact_repository_registry.get_artifact_repository",
        return_value=_PresignedArtifactRepository(),
    ) as mock_get_artifact_repository:
        artifacts_presigned_config_validation(
            True,
            serve_artifacts=True,
            artifacts_destination="s3://bucket",
        )

    mock_get_artifact_repository.assert_called_once_with("s3://bucket")


def test_presigned_only_rejects_sse_customer_key(monkeypatch):
    monkeypatch.setenv(
        "MLFLOW_S3_UPLOAD_EXTRA_ARGS",
        '{"SSECustomerAlgorithm": "AES256", "SSECustomerKey": "customer-key"}',
    )
    repo = S3ArtifactRepository("s3://bucket")
    with (
        mock.patch(
            "mlflow.store.artifact.artifact_repository_registry.get_artifact_repository",
            return_value=repo,
        ) as get_repo,
        pytest.raises(click.UsageError, match="cannot be used with SSECustomerKey"),
    ):
        artifacts_presigned_config_validation(
            True, serve_artifacts=True, artifacts_destination="s3://bucket"
        )
    get_repo.assert_called_once_with("s3://bucket")


@pytest.mark.parametrize(
    "flag",
    ["MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD", "MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD"],
)
def test_presigned_multipart_flags_reject_sse_customer_key(monkeypatch, flag):
    monkeypatch.setenv(flag, "true")
    monkeypatch.setenv("MLFLOW_S3_UPLOAD_EXTRA_ARGS", '{"SSECustomerKey": "customer-key"}')
    repo = S3ArtifactRepository("s3://bucket")
    with (
        mock.patch(
            "mlflow.store.artifact.artifact_repository_registry.get_artifact_repository",
            return_value=repo,
        ) as get_repo,
        pytest.raises(click.UsageError, match="Presigned multipart transfers cannot be used"),
    ):
        artifacts_presigned_config_validation(
            False, serve_artifacts=True, artifacts_destination="s3://bucket"
        )
    get_repo.assert_called_once_with("s3://bucket")
