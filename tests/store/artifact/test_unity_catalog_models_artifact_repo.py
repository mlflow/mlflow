import json
from unittest import mock
from unittest.mock import ANY

from google.cloud.storage import Client
import pytest
from requests import Response

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_uc_registry_messages_pb2 import MODEL_VERSION_READ_WRITE
from mlflow.utils.uri import _DATABRICKS_UNITY_CATALOG_SCHEME

from mlflow.store.artifact.s3_artifact_repo import S3ArtifactRepository
from mlflow.store.artifact.azure_data_lake_artifact_repo import AzureDataLakeArtifactRepository
from mlflow.store.artifact.gcs_artifact_repo import GCSArtifactRepository
from mlflow.store.artifact.unity_catalog_models_artifact_repo import (
    UnityCatalogModelsArtifactRepository,
)
from mlflow.store._unity_catalog.registry.rest_store import UcModelRegistryStore

MODELS_ARTIFACT_REPOSITORY_PACKAGE = "mlflow.store.artifact.unity_catalog_models_artifact_repo"
MODELS_ARTIFACT_REPOSITORY = (
    MODELS_ARTIFACT_REPOSITORY_PACKAGE + ".UnityCatalogModelsArtifactRepository"
)


# TODO: remove this mock once the UC model registry store is supported
@pytest.fixture(autouse=True, scope="module")
def mock_get_databricks_unity_catalog_store():
    def get_uc_rest_store(store_uri):
        return UcModelRegistryStore(store_uri)

    with mock.patch(
        "mlflow.tracking._model_registry.utils._get_databricks_uc_rest_store"
    ) as _get_databricks_uc_rest_store_mock:
        _get_databricks_uc_rest_store_mock.side_effect = get_uc_rest_store
        yield _get_databricks_uc_rest_store_mock


def test_uc_models_artifact_repo_init_with_uri_containing_profile():
    uri_with_profile = "models://profile@databricks-uc/MyModel/12"
    models_repo = UnityCatalogModelsArtifactRepository(
        uri_with_profile, _DATABRICKS_UNITY_CATALOG_SCHEME
    )
    assert models_repo.artifact_uri == uri_with_profile
    assert models_repo.client._registry_uri == f"{_DATABRICKS_UNITY_CATALOG_SCHEME}://profile"


def test_uc_models_artifact_repo_init_with_db_profile_inferred_from_context():
    uri_without_profile = "models:/MyModel/12"
    profile_in_registry_uri = "some_profile"
    registry_uri = f"databricks-uc://{profile_in_registry_uri}"
    models_repo = UnityCatalogModelsArtifactRepository(
        artifact_uri=uri_without_profile, registry_uri=registry_uri
    )
    assert models_repo.artifact_uri == uri_without_profile
    assert models_repo.client._registry_uri == registry_uri


def test_uc_models_artifact_repo_init_not_using_databricks_registry_raises():
    non_databricks_uri = "non_databricks_uri"
    model_uri = "models:/MyModel/12"
    with pytest.raises(
        MlflowException,
        match="Attempted to instantiate an artifact repo to access models in the Unity "
        "Catalog with non-Unity Catalog registry URI",
    ):
        UnityCatalogModelsArtifactRepository(model_uri, non_databricks_uri)


def test_uc_models_artifact_repo_with_stage_uri_raises():
    model_uri = "models:/MyModel/Staging"
    with pytest.raises(
        MlflowException, match="staged-based model URIs are unsupported for models in UC"
    ):
        UnityCatalogModelsArtifactRepository(model_uri, _DATABRICKS_UNITY_CATALOG_SCHEME)


def test_uc_models_artifact_uri_with_scope_and_prefix_throws():
    with pytest.raises(
        MlflowException,
        match="Remote model registry access via model URIs of the form "
        "'models://<scope>@<prefix>/<model_name>/<version_or_stage>'",
    ):
        UnityCatalogModelsArtifactRepository(
            "models://scope:prefix@databricks-uc/MyModel/12", _DATABRICKS_UNITY_CATALOG_SCHEME
        )


def _mock_temporary_creds_response(temporary_creds):
    mock_response = mock.MagicMock(autospec=Response)
    mock_response.status_code = 200
    mock_response.text = json.dumps({"credentials": temporary_creds})
    return mock_response


def test_uc_models_artifact_repo_download_artifacts_uses_temporary_creds_aws():
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
    with mock.patch.object(
        MlflowClient, "get_model_version_download_uri", return_value=artifact_location
    ), mock.patch("mlflow.utils.rest_utils.http_request") as request_mock, mock.patch(
        "mlflow.store.artifact.s3_artifact_repo.S3ArtifactRepository"
    ) as s3_artifact_repo_class_mock:
        mock_s3_repo = mock.MagicMock(autospec=S3ArtifactRepository)
        s3_artifact_repo_class_mock.return_value = mock_s3_repo
        request_mock.return_value = _mock_temporary_creds_response(temporary_creds)
        models_repo = UnityCatalogModelsArtifactRepository(
            artifact_uri="models:/MyModel/12", registry_uri=_DATABRICKS_UNITY_CATALOG_SCHEME
        )
        models_repo.download_artifacts("artifact_path", "dst_path")
        s3_artifact_repo_class_mock.assert_called_once_with(
            artifact_uri=artifact_location,
            access_key_id=fake_key_id,
            secret_access_key=fake_secret_access_key,
            session_token=fake_session_token,
        )
        mock_s3_repo.download_artifacts.assert_called_once_with("artifact_path", "dst_path")
        request_mock.assert_called_with(
            host_creds=ANY,
            endpoint="/mlflow/unity-catalog/model-versions/generate-temporary-credentials",
            method="POST",
            json={"name": "MyModel", "version": "12", "operation": MODEL_VERSION_READ_WRITE},
        )


def test_uc_models_artifact_repo_download_artifacts_uses_temporary_creds_azure():
    artifact_location = "abfss://filesystem@account.dfs.core.windows.net"
    fake_sas_token = "fake_session_token"
    temporary_creds = {
        "azure_user_delegation_sas": {
            "sas_token": fake_sas_token,
        },
    }
    with mock.patch.object(
        MlflowClient, "get_model_version_download_uri", return_value=artifact_location
    ), mock.patch("mlflow.utils.rest_utils.http_request") as request_mock, mock.patch(
        "mlflow.store.artifact.azure_data_lake_artifact_repo.AzureDataLakeArtifactRepository"
    ) as adls_artifact_repo_class_mock:
        mock_adls_repo = mock.MagicMock(autospec=AzureDataLakeArtifactRepository)
        adls_artifact_repo_class_mock.return_value = mock_adls_repo
        request_mock.return_value = _mock_temporary_creds_response(temporary_creds)
        models_repo = UnityCatalogModelsArtifactRepository(
            artifact_uri="models:/MyModel/12", registry_uri=_DATABRICKS_UNITY_CATALOG_SCHEME
        )
        models_repo.download_artifacts("artifact_path", "dst_path")
        adls_artifact_repo_class_mock.assert_called_once_with(
            artifact_uri=artifact_location, credential=ANY
        )
        adls_repo_args = adls_artifact_repo_class_mock.call_args_list[0]
        credential = adls_repo_args[1]["credential"]
        assert credential.signature == fake_sas_token
        mock_adls_repo.download_artifacts.assert_called_once_with("artifact_path", "dst_path")
        request_mock.assert_called_with(
            host_creds=ANY,
            endpoint="/mlflow/unity-catalog/model-versions/generate-temporary-credentials",
            method="POST",
            json={"name": "MyModel", "version": "12", "operation": MODEL_VERSION_READ_WRITE},
        )


def test_uc_models_artifact_repo_download_artifacts_uses_temporary_creds_gcp():
    artifact_location = "gs://test_bucket/some/path"
    fake_oauth_token = "fake_session_token"
    temporary_creds = {
        "gcp_oauth_token": {
            "oauth_token": fake_oauth_token,
        },
    }
    with mock.patch.object(
        MlflowClient, "get_model_version_download_uri", return_value=artifact_location
    ), mock.patch("mlflow.utils.rest_utils.http_request") as request_mock, mock.patch(
        "google.cloud.storage.Client"
    ) as gcs_client_class_mock, mock.patch(
        "mlflow.store.artifact.gcs_artifact_repo.GCSArtifactRepository"
    ) as gcs_artifact_repo_class_mock:
        mock_gcs_client = mock.MagicMock(autospec=Client)
        gcs_client_class_mock.return_value = mock_gcs_client
        mock_gcs_repo = mock.MagicMock(autospec=GCSArtifactRepository)
        gcs_artifact_repo_class_mock.return_value = mock_gcs_repo
        request_mock.return_value = _mock_temporary_creds_response(temporary_creds)
        models_repo = UnityCatalogModelsArtifactRepository(
            artifact_uri="models:/MyModel/12", registry_uri=_DATABRICKS_UNITY_CATALOG_SCHEME
        )
        models_repo.download_artifacts("artifact_path", "dst_path")
        gcs_artifact_repo_class_mock.assert_called_once_with(
            artifact_uri=artifact_location, client=ANY
        )
        mock_gcs_repo.download_artifacts.assert_called_once_with("artifact_path", "dst_path")
        gcs_client_args = gcs_client_class_mock.call_args_list[0]
        credentials = gcs_client_args[1]["credentials"]
        assert credentials.token == fake_oauth_token
        request_mock.assert_called_with(
            host_creds=ANY,
            endpoint="/mlflow/unity-catalog/model-versions/generate-temporary-credentials",
            method="POST",
            json={"name": "MyModel", "version": "12", "operation": MODEL_VERSION_READ_WRITE},
        )
