from unittest import mock

import pytest

from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.uc_volumes_artifact_repo import UCVolumesRestArtifactRepository
from mlflow.utils.rest_utils import MlflowHostCreds


@pytest.fixture(autouse=True)
def mock_get_host_creds():
    with mock.patch(
        "mlflow.store.artifact.uc_volumes_artifact_repo._get_host_creds_factory",
        return_value=lambda: MlflowHostCreds(host="http://localhost"),
    ) as mock_creds_factory:
        yield

        mock_creds_factory.assert_called_once()


@pytest.mark.parametrize(
    "artifact_uri",
    [
        "dbfs:/Volumes/some/path",
        "/Volumes/some/path",
        "/volumes/some/path",
        "/Volume/some/path",
        "/volume/some/path",
    ],
)
@pytest.mark.usefixtures(mock_get_host_creds.__name__)
def test_get_artifact_repository(artifact_uri):
    repo = get_artifact_repository(artifact_uri)
    assert isinstance(repo, UCVolumesRestArtifactRepository)
