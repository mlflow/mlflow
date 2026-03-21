import posixpath
from unittest import mock
from unittest.mock import ANY

import pytest

from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.s3_artifact_repo import _cached_get_s3_client

from tests.helper_functions import set_boto_credentials  # noqa: F401


@pytest.fixture
def r2_artifact_root():
    return "r2://mock-r2-bucket@account.r2.cloudflarestorage.com"


@pytest.fixture(autouse=True)
def reset_cached_get_s3_client():
    _cached_get_s3_client.cache_clear()


def test_parse_r2_uri(r2_artifact_root):
    with mock.patch("boto3.client") as _:
        artifact_uri = posixpath.join(r2_artifact_root, "some/path")
        repo = get_artifact_repository(artifact_uri)
        parsed_bucket, parsed_path = repo.parse_s3_compliant_uri(artifact_uri)
        assert parsed_bucket == "mock-r2-bucket"
        assert parsed_path == "some/path"


def test_s3_client_config_set_correctly(r2_artifact_root):
    with mock.patch(
        "mlflow.store.artifact.r2_artifact_repo.R2ArtifactRepository._get_region_name"
    ) as mock_method:
        mock_method.return_value = None

        artifact_uri = posixpath.join(r2_artifact_root, "some/path")
        repo = get_artifact_repository(artifact_uri)

        s3_client = repo._get_s3_client()
        assert s3_client.meta.config.s3.get("addressing_style") == "virtual"


def test_convert_r2_uri_to_s3_endpoint_url(r2_artifact_root):
    with mock.patch("boto3.client") as _:
        artifact_uri = posixpath.join(r2_artifact_root, "some/path")
        repo = get_artifact_repository(artifact_uri)

        s3_endpoint_url = repo.convert_r2_uri_to_s3_endpoint_url(r2_artifact_root)
        assert s3_endpoint_url == "https://account.r2.cloudflarestorage.com"


def test_s3_endpoint_url_is_used_to_get_s3_client(r2_artifact_root):
    with mock.patch("boto3.client") as mock_get_s3_client:
        artifact_uri = posixpath.join(r2_artifact_root, "some/path")
        repo = get_artifact_repository(artifact_uri)
        repo._get_s3_client()
        mock_get_s3_client.assert_called_with(
            "s3",
            config=ANY,
            endpoint_url="https://account.r2.cloudflarestorage.com",
            verify=None,
            aws_access_key_id=None,
            aws_secret_access_key=None,
            aws_session_token=None,
            region_name=ANY,
        )


def test_get_r2_client_region_name_set_correctly(r2_artifact_root):
    region_name = "us_random_region_42"
    with mock.patch("boto3.client") as mock_get_s3_client:
        s3_client_mock = mock.Mock()
        mock_get_s3_client.return_value = s3_client_mock
        s3_client_mock.get_bucket_location.return_value = {"LocationConstraint": region_name}

        artifact_uri = posixpath.join(r2_artifact_root, "some/path")
        repo = get_artifact_repository(artifact_uri)
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
