import json
from unittest import mock
from unittest.mock import ANY

from requests import Response

from mlflow import MlflowClient
from mlflow.store.artifact.optimized_s3_artifact_repo import OptimizedS3ArtifactRepository
from mlflow.store.artifact.unity_catalog_oss_models_artifact_repo import (
    UnityCatalogOSSModelsArtifactRepository,
)
from mlflow.utils.uri import _OSS_UNITY_CATALOG_SCHEME

MODELS_ARTIFACT_REPOSITORY_PACKAGE = "mlflow.store.artifact.unity_catalog_models_artifact_repo"
MODELS_ARTIFACT_REPOSITORY = (
    MODELS_ARTIFACT_REPOSITORY_PACKAGE + ".UnityCatalogModelsArtifactRepository"
)


def _mock_temporary_creds_response(temporary_creds):
    mock_response = mock.MagicMock(autospec=Response)
    mock_response.status_code = 200
    mock_response.text = json.dumps(temporary_creds)
    return mock_response


def test_uc_models_artifact_repo_init_with_uri_containing_profile():
    uri_with_profile = "models://profile@uc/MyModel/12"
    models_repo = UnityCatalogOSSModelsArtifactRepository(
        uri_with_profile, f"{_OSS_UNITY_CATALOG_SCHEME}:databricks"
    )
    assert models_repo.artifact_uri == uri_with_profile
    assert models_repo.client._registry_uri == f"{_OSS_UNITY_CATALOG_SCHEME}://profile"


def test_uc_models_artifact_repo_scoped_token_oss(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "my-host")
    monkeypatch.setenv("DATABRICKS_TOKEN", "my-token")
    artifact_location = "s3://blah_bucket/"
    fake_key_id = "fake_key_id"
    fake_secret_access_key = "fake_secret_access_key"
    fake_session_token = "fake_session_token"
    temporary_creds = {
        "aws_temp_credentials": {
            "access_key_id": fake_key_id,
            "secret_access_key": fake_secret_access_key,
            "session_token": fake_session_token,
        }
    }
    fake_local_path = "/tmp/fake_path"
    with (
        mock.patch("mlflow.utils.oss_registry_utils.get_oss_host_creds"),
        mock.patch.object(
            MlflowClient, "get_model_version_download_uri", return_value=artifact_location
        ),
        mock.patch("mlflow.utils.rest_utils.http_request") as request_mock,
        mock.patch(
            "mlflow.store.artifact.optimized_s3_artifact_repo.OptimizedS3ArtifactRepository"
        ) as optimized_s3_artifact_repo_class_mock,
    ):
        mock_s3_repo = mock.MagicMock(autospec=OptimizedS3ArtifactRepository)
        mock_s3_repo.download_artifacts.return_value = fake_local_path
        optimized_s3_artifact_repo_class_mock.return_value = mock_s3_repo
        request_mock.return_value = _mock_temporary_creds_response(temporary_creds)
        models_repo = UnityCatalogOSSModelsArtifactRepository(
            artifact_uri="models:/test-catalog.test-schema.test-model/12",
            registry_uri=f"{_OSS_UNITY_CATALOG_SCHEME}:databricks",
        )
        models_repo._get_artifact_repo()

        request_mock.assert_called_with(
            host_creds=ANY,
            endpoint="/api/2.1/unity-catalog/temporary-model-version-credentials",
            method="POST",
            json={
                "catalog_name": "test-catalog",
                "schema_name": "test-schema",
                "model_name": "test-model",
                "version": 12,
                "operation": "READ_MODEL_VERSION",
            },
            extra_headers=ANY,
        )
