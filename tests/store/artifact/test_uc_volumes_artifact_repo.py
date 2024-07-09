from unittest import mock

import pytest

from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.uc_volumes_artifact_repo import UCVolumesRestArtifactRepository
from mlflow.utils.rest_utils import MlflowHostCreds


@pytest.fixture
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
        "dbfs:/Volumes/catalog/schema/volume",
        "dbfs://profile@databricks/Volumes/catalog/schema/volume",
    ],
)
@pytest.mark.usefixtures(mock_get_host_creds.__name__)
def test_get_artifact_repository(artifact_uri):
    repo = get_artifact_repository(artifact_uri)
    assert isinstance(repo, UCVolumesRestArtifactRepository)


@pytest.mark.parametrize(
    "artifact_uri",
    [
        "dbfs:/Volumes/catalog",
        "dbfs:/Volumes/catalog/schema",
        "dbfs://profile@databricks/Volumes/catalog",
        "dbfs://profile@databricks/Volumes/catalog/schema",
    ],
)
def test_get_artifact_repository_invalid_uri(artifact_uri):
    with pytest.raises(MlflowException, match="Artifact URI must be of the form"):
        get_artifact_repository(artifact_uri)
