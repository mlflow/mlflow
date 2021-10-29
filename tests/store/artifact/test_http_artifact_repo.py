import pytest

from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.http_artifact_repo import HttpArtifactRepository


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_artifact_uri_factory(scheme):
    repo = get_artifact_repository(f"{scheme}://mlflow.com")
    assert isinstance(repo, HttpArtifactRepository)
