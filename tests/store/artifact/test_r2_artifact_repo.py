import posixpath
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
    artifact_uri = posixpath.join(r2_artifact_root, "some/path")
    repo = get_artifact_repository(artifact_uri)
    parsed_bucket, parsed_path = repo.parse_s3_compliant_uri(artifact_uri)
    assert parsed_bucket == "mock-r2-bucket"
    assert parsed_path == "some/path"


def test_s3_client_config_set_correctly(r2_artifact_root):
    artifact_uri = posixpath.join(r2_artifact_root, "some/path")
    repo = get_artifact_repository(artifact_uri)

    s3_client = repo._get_s3_client()
    assert s3_client.meta.config.s3.get("addressing_style") == "virtual"
