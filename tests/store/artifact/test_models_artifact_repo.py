import pytest
from unittest import mock
from unittest.mock import Mock

from mlflow.entities.model_registry import ModelVersion
from mlflow.store.artifact.databricks_models_artifact_repo import DatabricksModelsArtifactRepository
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.tracking import MlflowClient

MODELS_ARTIFACT_REPOSITORY_PACKAGE = "mlflow.store.artifact.models_artifact_repo"
MODELS_ARTIFACT_REPOSITORY = MODELS_ARTIFACT_REPOSITORY_PACKAGE + ".ModelsArtifactRepository"


@pytest.mark.parametrize(
    "uri_with_profile",
    [
        "models://profile@databricks/MyModel/12",
        "models://profile@databricks/MyModel/Staging",
        "models://profile@databricks/MyModel/Production",
    ],
)
def test_models_artifact_repo_init_with_uri_containing_profile(uri_with_profile):
    with mock.patch(
        MODELS_ARTIFACT_REPOSITORY_PACKAGE + ".DatabricksModelsArtifactRepository", autospec=True
    ) as mock_repo:
        models_repo = ModelsArtifactRepository(uri_with_profile)
        assert models_repo.artifact_uri == uri_with_profile
        assert isinstance(models_repo.repo, DatabricksModelsArtifactRepository)
        mock_repo.assert_called_once_with(uri_with_profile)


@pytest.mark.parametrize(
    "uri_without_profile",
    ["models:/MyModel/12", "models:/MyModel/Staging", "models:/MyModel/Production"],
)
def test_models_artifact_repo_init_with_db_profile_inferred_from_context(uri_without_profile):
    with mock.patch(
        MODELS_ARTIFACT_REPOSITORY_PACKAGE + ".DatabricksModelsArtifactRepository", autospec=True
    ) as mock_repo, mock.patch(
        "mlflow.store.artifact.utils.models.mlflow.get_registry_uri",
        return_value="databricks://getRegistryUriDefault",
    ):
        models_repo = ModelsArtifactRepository(uri_without_profile)
        assert models_repo.artifact_uri == uri_without_profile
        assert isinstance(models_repo.repo, DatabricksModelsArtifactRepository)
        mock_repo.assert_called_once_with(uri_without_profile)


def test_models_artifact_repo_init_with_version_uri_and_not_using_databricks_registry():
    non_databricks_uri = "non_databricks_uri"
    artifact_location = "s3://blah_bucket/"
    with mock.patch.object(
        MlflowClient, "get_model_version_download_uri", return_value=artifact_location
    ), mock.patch(
        "mlflow.store.artifact.utils.models.mlflow.get_registry_uri",
        return_value=non_databricks_uri,
    ), mock.patch(
        "mlflow.store.artifact.artifact_repository_registry.get_artifact_repository"
    ) as get_repo_mock:
        get_repo_mock.return_value = None
        model_uri = "models:/MyModel/12"
        ModelsArtifactRepository(model_uri)
        get_repo_mock.assert_called_once_with(artifact_location)


def test_models_artifact_repo_init_with_stage_uri_and_not_using_databricks_registry():
    model_uri = "models:/MyModel/Staging"
    artifact_location = "s3://blah_bucket/"
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
    get_model_version_download_uri_patch = mock.patch.object(
        MlflowClient, "get_model_version_download_uri", return_value=artifact_location
    )
    with get_latest_versions_patch, get_model_version_download_uri_patch, mock.patch(
        "mlflow.store.artifact.artifact_repository_registry.get_artifact_repository"
    ) as get_repo_mock:
        get_repo_mock.return_value = None
        ModelsArtifactRepository(model_uri)
        get_repo_mock.assert_called_once_with(artifact_location)


def test_models_artifact_repo_uses_repo_download_artifacts():
    """
    ``ModelsArtifactRepository`` should delegate `download_artifacts` to its
    ``self.repo.download_artifacts`` function.
    """
    artifact_location = "s3://blah_bucket/"
    with mock.patch.object(
        MlflowClient, "get_model_version_download_uri", return_value=artifact_location
    ):
        model_uri = "models:/MyModel/12"
        models_repo = ModelsArtifactRepository(model_uri)
        models_repo.repo = Mock()
        models_repo.download_artifacts("artifact_path", "dst_path")
        models_repo.repo.download_artifacts.assert_called_once()
