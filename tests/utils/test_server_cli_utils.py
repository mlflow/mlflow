from unittest import mock

import click
import pytest

from mlflow.store.artifact.artifact_repo import MultipartDownloadMixin, MultipartUploadMixin
from mlflow.utils.server_cli_utils import artifacts_only_presigned_config_validation


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
        artifacts_only_presigned_config_validation(
            True,
            serve_artifacts=False,
            artifacts_only=False,
            artifacts_destination="s3://bucket",
        )


def test_presigned_only_requires_supported_artifact_destination():
    with (
        mock.patch(
            "mlflow.store.artifact.artifact_repository_registry.get_artifact_repository",
            return_value=mock.Mock(spec=[]),
        ),
        pytest.raises(click.UsageError, match="presigned uploads and presigned downloads"),
    ):
        artifacts_only_presigned_config_validation(
            True,
            serve_artifacts=True,
            artifacts_only=False,
            artifacts_destination="./mlartifacts",
        )


def test_presigned_only_accepts_supported_artifact_destination():
    with mock.patch(
        "mlflow.store.artifact.artifact_repository_registry.get_artifact_repository",
        return_value=_PresignedArtifactRepository(),
    ):
        artifacts_only_presigned_config_validation(
            True,
            serve_artifacts=True,
            artifacts_only=False,
            artifacts_destination="s3://bucket",
        )
