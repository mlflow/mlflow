import pytest

from mlflow.store.artifact.s3_artifact_repo import _cached_get_s3_client


@pytest.fixture(autouse=True)
def reset_cached_get_s3_client():
    # `_cached_get_s3_client` is a process-global lru_cache whose key does not include the
    # resolved AWS credentials. Under pytest-xdist a worker runs many modules, so a boto3 S3
    # client cached by an earlier test (potentially with no credentials in the environment)
    # would be reused here and raise `NoCredentialsError`. Clearing the cache before each test
    # forces a fresh client that picks up the fake credentials from `set_boto_credentials`.
    _cached_get_s3_client.cache_clear()
