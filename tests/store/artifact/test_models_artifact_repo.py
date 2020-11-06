import pytest
from unittest import mock
from unittest.mock import Mock

from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.databricks_model_artifact_repo import DatabricksModelArtifactRepository
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.tracking import MlflowClient

# pylint: disable=unused-import
from tests.store.artifact.test_databricks_model_artifact_repo import host_creds_mock

DATABRICKS_MODEL_ARTIFACT_REPOSITORY_PACKAGE = (
    "mlflow.store.artifact.databricks_model_artifact_repo"
)
DATABRICKS_MODEL_ARTIFACT_REPOSITORY = (
    DATABRICKS_MODEL_ARTIFACT_REPOSITORY_PACKAGE + ".DatabricksModelArtifactRepository"
)


@pytest.fixture
def mock_get_model_version_download_uri(artifact_location):
    with mock.patch.object(
        MlflowClient, "get_model_version_download_uri", return_value=artifact_location
    ) as mockval:
        yield mockval.return_value


@pytest.mark.parametrize(
    "artifact_location", ["dbfs:/databricks/mlflow-registry/12345/models/keras-model"]
)
def test_models_artifact_repo_init_with_version_uri_and_db_profile(
    host_creds_mock, mock_get_model_version_download_uri,
):  # pylint: disable=unused-argument
    model_uri = "models://profile@databricks/MyModel/12"
    with mock.patch(DATABRICKS_MODEL_ARTIFACT_REPOSITORY, autospec=True):
        models_repo = ModelsArtifactRepository(model_uri)
        assert models_repo.artifact_uri == model_uri
        assert isinstance(models_repo.repo, DatabricksModelArtifactRepository)


@pytest.mark.parametrize(
    "artifact_location", ["dbfs:/databricks/mlflow-registry/12345/models/keras-model"]
)
def test_models_artifact_repo_init_with_version_uri_and_db_profile_from_context(
    host_creds_mock, mock_get_model_version_download_uri,
):  # pylint: disable=unused-argument
    model_uri = "models:/MyModel/12"
    with mock.patch(DATABRICKS_MODEL_ARTIFACT_REPOSITORY, autospec=True), mock.patch(
        DATABRICKS_MODEL_ARTIFACT_REPOSITORY_PACKAGE + ".is_databricks_profile", return_value=True
    ), mock.patch(
        DATABRICKS_MODEL_ARTIFACT_REPOSITORY_PACKAGE + ".mlflow.get_registry_uri",
        return_value="databricks://scope:key",
    ):
        models_repo = ModelsArtifactRepository(model_uri)
        assert models_repo.artifact_uri == model_uri
        assert isinstance(models_repo.repo, DatabricksModelArtifactRepository)


@pytest.mark.parametrize(
    "artifact_location", ["dbfs:/databricks/mlflow-registry/12345/models/keras-model"]
)
def test_models_artifact_repo_init_with_version_uri_and_bad_db_profile_from_context(
    host_creds_mock, mock_get_model_version_download_uri,
):  # pylint: disable=unused-argument
    model_uri = "models:/MyModel/12"
    with mock.patch(DATABRICKS_MODEL_ARTIFACT_REPOSITORY, autospec=True), mock.patch(
        DATABRICKS_MODEL_ARTIFACT_REPOSITORY_PACKAGE + ".is_databricks_profile", return_value=True
    ), mock.patch(
        DATABRICKS_MODEL_ARTIFACT_REPOSITORY_PACKAGE + ".mlflow.get_registry_uri",
        return_value="databricks://scope:key:invalid",
    ):
        with pytest.raises(MlflowException) as ex:
            ModelsArtifactRepository(model_uri)
        assert "Key prefixes cannot contain" in ex.value.message


@pytest.mark.parametrize(
    "artifact_location", ["dbfs:/databricks/mlflow-registry/12345/models/keras-model"]
)
def test_models_artifact_repo_init_with_stage_uri(
    host_creds_mock, mock_get_model_version_download_uri, artifact_location
):  # pylint: disable=unused-argument
    model_uri = "models://profile@databricks/MyModel/12"
    model_version_detailed = ModelVersion(
        "MyModel",
        "10",
        "2345671890",
        "234567890",
        "some description",
        "UserID",
        "Production",
        "source",
        "run12345",
    )
    get_latest_versions_patch = mock.patch.object(
        MlflowClient, "get_latest_versions", return_value=[model_version_detailed]
    )
    with get_latest_versions_patch:
        models_repo = ModelsArtifactRepository(model_uri)
        assert models_repo.artifact_uri == model_uri
        assert isinstance(models_repo.repo, DatabricksModelArtifactRepository)
        assert models_repo.repo.artifact_uri == model_uri


@pytest.mark.parametrize("artifact_location", ["s3://blah_bucket/"])
def test_models_artifact_repo_uses_repo_download_artifacts(
    mock_get_model_version_download_uri,
):  # pylint: disable=unused-argument
    """
    ``ModelsArtifactRepository`` should delegate `download_artifacts` to its
    ``self.repo.download_artifacts`` function.
    """
    model_uri = "models:/MyModel/12"
    models_repo = ModelsArtifactRepository(model_uri)
    models_repo.repo = Mock()
    models_repo.download_artifacts("artifact_path", "dst_path")
    models_repo.repo.download_artifacts.assert_called_once()
