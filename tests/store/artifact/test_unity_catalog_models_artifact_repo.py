import pytest
from unittest import mock
from unittest.mock import Mock

from mlflow.exceptions import MlflowException
from mlflow.store.artifact.unity_catalog_models_artifact_repo import UnityCatalogModelsArtifactRepository
from mlflow import MlflowClient
from mlflow.utils.uri import _DATABRICKS_UNITY_CATALOG_SCHEME

MODELS_ARTIFACT_REPOSITORY_PACKAGE = "mlflow.store.artifact.unity_catalog_models_artifact_repo"
MODELS_ARTIFACT_REPOSITORY = MODELS_ARTIFACT_REPOSITORY_PACKAGE + ".UnityCatalogModelsArtifactRepository"


@pytest.mark.parametrize(
    "uri_with_profile",
    [
        "models://profile@databricks/MyModel/12"
    ],
)
def test_uc_models_artifact_repo_init_with_uri_containing_profile(uri_with_profile):
    models_repo = UnityCatalogModelsArtifactRepository(uri_with_profile, _DATABRICKS_UNITY_CATALOG_SCHEME)
    assert models_repo.artifact_uri == uri_with_profile
    assert models_repo.client._registry_uri == f"{_DATABRICKS_UNITY_CATALOG_SCHEME}://profile"


def test_uc_models_artifact_repo_init_with_db_profile_inferred_from_context():
    uri_without_profile = "models:/MyModel/12"
    profile_in_registry_uri = "some_profile"
    registry_uri = f"databricks-uc://{profile_in_registry_uri}"
    models_repo = UnityCatalogModelsArtifactRepository(artifact_uri=uri_without_profile, registry_uri=registry_uri)
    assert models_repo.artifact_uri == uri_without_profile
    assert models_repo.client._registry_uri == registry_uri



def test_uc_models_artifact_repo_init_not_using_databricks_registry_raises():
    non_databricks_uri = "non_databricks_uri"
    model_uri = "models:/MyModel/12"
    with pytest.raises(MlflowException, match="Attempted to instantiate an artifact repo to access models in the Unity Catalog with non-Unity Catalog registry URI"):
        UnityCatalogModelsArtifactRepository(model_uri, non_databricks_uri)


def test_uc_models_artifact_repo_with_stage_uri_raises():
    model_uri = "models:/MyModel/Staging"
    with pytest.raises(MlflowException):
        UnityCatalogModelsArtifactRepository(model_uri, _DATABRICKS_UNITY_CATALOG_SCHEME)


def test_uc_models_artifact_repo_uses_repo_download_artifacts():
    """
    ``ModelsArtifactRepository`` should delegate `download_artifacts` to its
    ``self.repo.download_artifacts`` function.
    """
    artifact_location = "s3://blah_bucket/"
    with mock.patch.object(
            MlflowClient, "get_model_version_download_uri", return_value=artifact_location
    ):
        model_uri = "models:/MyModel/12"
        models_repo = UnityCatalogModelsArtifactRepository(model_uri, _DATABRICKS_UNITY_CATALOG_SCHEME)
        models_repo.repo = Mock()
        models_repo.download_artifacts("artifact_path", "dst_path")
        models_repo.repo.download_artifacts.assert_called_once()

