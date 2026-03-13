import posixpath
from unittest import mock
from unittest.mock import ANY

import pytest

from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.b2_artifact_repo import _B2_USER_AGENT, _add_b2_user_agent
from mlflow.store.artifact.s3_artifact_repo import _cached_get_s3_client

from tests.helper_functions import set_boto_credentials  # noqa: F401


@pytest.fixture
def b2_artifact_root():
    return "b2://mock-b2-bucket@s3.us-west-004.backblazeb2.com"


@pytest.fixture(autouse=True)
def reset_cached_get_s3_client():
    _cached_get_s3_client.cache_clear()


def test_parse_b2_uri(b2_artifact_root):
    with mock.patch("boto3.client"):
        artifact_uri = posixpath.join(b2_artifact_root, "some/path")
        repo = get_artifact_repository(artifact_uri)
        parsed_bucket, parsed_path = repo.parse_s3_compliant_uri(artifact_uri)
        assert parsed_bucket == "mock-b2-bucket"
        assert parsed_path == "some/path"


def test_s3_client_config_set_correctly(b2_artifact_root):
    artifact_uri = posixpath.join(b2_artifact_root, "some/path")
    repo = get_artifact_repository(artifact_uri)

    s3_client = repo._get_s3_client()
    assert s3_client.meta.config.s3.get("addressing_style") == "path"


def test_b2_user_agent_event_registered(b2_artifact_root):
    artifact_uri = posixpath.join(b2_artifact_root, "some/path")
    repo = get_artifact_repository(artifact_uri)

    s3_client = repo._get_s3_client()
    assert "b2-user-agent" in s3_client.meta.events._emitter._unique_id_handlers


def test_b2_user_agent_appended_to_request():
    request = mock.Mock()
    request.headers = {"User-Agent": "Boto3/1.0"}
    _add_b2_user_agent(request)
    assert _B2_USER_AGENT in request.headers["User-Agent"]


def test_convert_b2_uri_to_s3_endpoint_url(b2_artifact_root):
    with mock.patch("boto3.client"):
        artifact_uri = posixpath.join(b2_artifact_root, "some/path")
        repo = get_artifact_repository(artifact_uri)

        s3_endpoint_url = repo.convert_b2_uri_to_s3_endpoint_url(b2_artifact_root)
        assert s3_endpoint_url == "https://s3.us-west-004.backblazeb2.com"


def test_s3_endpoint_url_is_used_to_get_s3_client(b2_artifact_root):
    with mock.patch("boto3.client") as mock_get_s3_client:
        artifact_uri = posixpath.join(b2_artifact_root, "some/path")
        repo = get_artifact_repository(artifact_uri)
        repo._get_s3_client()
        mock_get_s3_client.assert_called_with(
            "s3",
            config=ANY,
            endpoint_url="https://s3.us-west-004.backblazeb2.com",
            verify=None,
            aws_access_key_id=None,
            aws_secret_access_key=None,
            aws_session_token=None,
            region_name="us-west-004",
        )


def test_region_parsed_from_endpoint_url(b2_artifact_root):
    with mock.patch("boto3.client"):
        artifact_uri = posixpath.join(b2_artifact_root, "some/path")
        repo = get_artifact_repository(artifact_uri)
        assert repo._region_name == "us-west-004"


def test_region_parse_fails_for_invalid_endpoint():
    invalid_uri = "b2://bucket@invalid-host.example.com/path"
    with pytest.raises(Exception, match="Unable to parse region from B2 endpoint"):
        get_artifact_repository(invalid_uri)
