import json
from unittest import mock
from unittest.mock import ANY

import pytest
from google.cloud.storage import Client
from requests import Response

from mlflow import MlflowClient
from mlflow.entities.file_info import FileInfo
from mlflow.exceptions import MlflowException
from mlflow.store._unity_catalog.registry.utils import _ACTIVE_CATALOG_QUERY, _ACTIVE_SCHEMA_QUERY
from mlflow.store.artifact.azure_data_lake_artifact_repo import AzureDataLakeArtifactRepository
from mlflow.store.artifact.gcs_artifact_repo import GCSArtifactRepository
from mlflow.store.artifact.optimized_s3_artifact_repo import OptimizedS3ArtifactRepository
from mlflow.store.artifact.unity_catalog_models_artifact_repo import (
    UnityCatalogModelsArtifactRepository,
)
from mlflow.utils.uri import _DATABRICKS_UNITY_CATALOG_SCHEME

MODELS_ARTIFACT_REPOSITORY_PACKAGE = "mlflow.store.artifact.unity_catalog_models_artifact_repo"
MODELS_ARTIFACT_REPOSITORY = (
    MODELS_ARTIFACT_REPOSITORY_PACKAGE + ".UnityCatalogModelsArtifactRepository"
)


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
    with mock.patch("databricks_cli.configure.provider.get_config"), pytest.raises(
        MlflowException,
        match="If seeing this error while attempting to load a model version by stage, note that "
        "setting stages and loading model versions by stage is unsupported in Unity Catalog",
    ):
        UnityCatalogModelsArtifactRepository(
            artifact_uri=model_uri, registry_uri=_DATABRICKS_UNITY_CATALOG_SCHEME
        )


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
    fake_local_path = "/tmp/fake_path"
    with mock.patch("databricks_cli.configure.provider.get_config"), mock.patch.object(
        MlflowClient, "get_model_version_download_uri", return_value=artifact_location
    ), mock.patch("mlflow.utils.rest_utils.http_request") as request_mock, mock.patch(
        "mlflow.store.artifact.optimized_s3_artifact_repo.OptimizedS3ArtifactRepository"
    ) as optimized_s3_artifact_repo_class_mock:
        mock_s3_repo = mock.MagicMock(autospec=OptimizedS3ArtifactRepository)
        mock_s3_repo.download_artifacts.return_value = fake_local_path
        optimized_s3_artifact_repo_class_mock.return_value = mock_s3_repo
        request_mock.return_value = _mock_temporary_creds_response(temporary_creds)
        models_repo = UnityCatalogModelsArtifactRepository(
            artifact_uri="models:/MyModel/12", registry_uri=_DATABRICKS_UNITY_CATALOG_SCHEME
        )
        assert models_repo.download_artifacts("artifact_path", "dst_path") == fake_local_path
        optimized_s3_artifact_repo_class_mock.assert_called_once_with(
            artifact_uri=artifact_location,
            access_key_id=fake_key_id,
            secret_access_key=fake_secret_access_key,
            session_token=fake_session_token,
        )
        mock_s3_repo.download_artifacts.assert_called_once_with("artifact_path", "dst_path")
        request_mock.assert_called_with(
            host_creds=ANY,
            endpoint="/api/2.0/mlflow/unity-catalog/model-versions/generate-temporary-credentials",
            method="POST",
            json={"name": "MyModel", "version": "12", "operation": "MODEL_VERSION_OPERATION_READ"},
        )


def test_uc_models_artifact_repo_download_artifacts_uses_temporary_creds_azure():
    artifact_location = "abfss://filesystem@account.dfs.core.windows.net"
    fake_sas_token = "fake_session_token"
    temporary_creds = {
        "azure_user_delegation_sas": {
            "sas_token": fake_sas_token,
        },
    }
    fake_local_path = "/tmp/fake_path"
    with mock.patch("databricks_cli.configure.provider.get_config"), mock.patch.object(
        MlflowClient, "get_model_version_download_uri", return_value=artifact_location
    ), mock.patch("mlflow.utils.rest_utils.http_request") as request_mock, mock.patch(
        "mlflow.store.artifact.azure_data_lake_artifact_repo.AzureDataLakeArtifactRepository"
    ) as adls_artifact_repo_class_mock:
        mock_adls_repo = mock.MagicMock(autospec=AzureDataLakeArtifactRepository)
        mock_adls_repo.download_artifacts.return_value = fake_local_path
        adls_artifact_repo_class_mock.return_value = mock_adls_repo
        request_mock.return_value = _mock_temporary_creds_response(temporary_creds)
        models_repo = UnityCatalogModelsArtifactRepository(
            artifact_uri="models:/MyModel/12", registry_uri=_DATABRICKS_UNITY_CATALOG_SCHEME
        )
        assert models_repo.download_artifacts("artifact_path", "dst_path") == fake_local_path
        adls_artifact_repo_class_mock.assert_called_once_with(
            artifact_uri=artifact_location, credential=ANY
        )
        adls_repo_args = adls_artifact_repo_class_mock.call_args_list[0]
        credential = adls_repo_args[1]["credential"]
        assert credential.signature == fake_sas_token
        mock_adls_repo.download_artifacts.assert_called_once_with("artifact_path", "dst_path")
        request_mock.assert_called_with(
            host_creds=ANY,
            endpoint="/api/2.0/mlflow/unity-catalog/model-versions/generate-temporary-credentials",
            method="POST",
            json={"name": "MyModel", "version": "12", "operation": "MODEL_VERSION_OPERATION_READ"},
        )


def test_uc_models_artifact_repo_download_artifacts_uses_temporary_creds_gcp():
    artifact_location = "gs://test_bucket/some/path"
    fake_oauth_token = "fake_session_token"
    temporary_creds = {
        "gcp_oauth_token": {
            "oauth_token": fake_oauth_token,
        },
    }
    fake_local_path = "/tmp/fake_path"
    with mock.patch("databricks_cli.configure.provider.get_config"), mock.patch.object(
        MlflowClient, "get_model_version_download_uri", return_value=artifact_location
    ), mock.patch("mlflow.utils.rest_utils.http_request") as request_mock, mock.patch(
        "google.cloud.storage.Client"
    ) as gcs_client_class_mock, mock.patch(
        "mlflow.store.artifact.gcs_artifact_repo.GCSArtifactRepository"
    ) as gcs_artifact_repo_class_mock:
        mock_gcs_client = mock.MagicMock(autospec=Client)
        gcs_client_class_mock.return_value = mock_gcs_client
        mock_gcs_repo = mock.MagicMock(autospec=GCSArtifactRepository)
        mock_gcs_repo.download_artifacts.return_value = fake_local_path
        gcs_artifact_repo_class_mock.return_value = mock_gcs_repo
        request_mock.return_value = _mock_temporary_creds_response(temporary_creds)
        models_repo = UnityCatalogModelsArtifactRepository(
            artifact_uri="models:/MyModel/12", registry_uri=_DATABRICKS_UNITY_CATALOG_SCHEME
        )
        assert models_repo.download_artifacts("artifact_path", "dst_path") == fake_local_path
        gcs_artifact_repo_class_mock.assert_called_once_with(
            artifact_uri=artifact_location, client=ANY
        )
        mock_gcs_repo.download_artifacts.assert_called_once_with("artifact_path", "dst_path")
        gcs_client_args = gcs_client_class_mock.call_args_list[0]
        credentials = gcs_client_args[1]["credentials"]
        assert credentials.token == fake_oauth_token
        request_mock.assert_called_with(
            host_creds=ANY,
            endpoint="/api/2.0/mlflow/unity-catalog/model-versions/generate-temporary-credentials",
            method="POST",
            json={"name": "MyModel", "version": "12", "operation": "MODEL_VERSION_OPERATION_READ"},
        )


def test_uc_models_artifact_repo_uses_active_catalog_and_schema():
    with mock.patch(
        "mlflow.store.artifact.unity_catalog_models_artifact_repo._get_active_spark_session"
    ) as spark_session_getter:
        spark = mock.MagicMock()
        spark_session_getter.return_value = spark
        sql_mock = mock.MagicMock()
        spark.sql.return_value = sql_mock
        # returns catalog and schema name in order
        sql_mock.collect.side_effect = [[{"catalog": "main"}], [{"schema": "default"}]]

        uri_with_profile = "models:/MyModel/12"
        models_repo = UnityCatalogModelsArtifactRepository(
            uri_with_profile, _DATABRICKS_UNITY_CATALOG_SCHEME
        )
        spark.sql.assert_has_calls(sql_mock(_ACTIVE_CATALOG_QUERY), sql_mock(_ACTIVE_SCHEMA_QUERY))
        assert spark.sql.call_count == 2
        assert models_repo.model_name == "main.default.MyModel"


def test_uc_models_artifact_repo_list_artifacts_uses_temporary_creds():
    artifact_location = "abfss://filesystem@account.dfs.core.windows.net"
    fake_sas_token = "fake_session_token"
    temporary_creds = {
        "azure_user_delegation_sas": {
            "sas_token": fake_sas_token,
        },
    }
    fake_local_path = "/tmp/fake_path"
    with mock.patch("databricks_cli.configure.provider.get_config"), mock.patch.object(
        MlflowClient, "get_model_version_download_uri", return_value=artifact_location
    ), mock.patch("mlflow.utils.rest_utils.http_request") as request_mock, mock.patch(
        "mlflow.store.artifact.azure_data_lake_artifact_repo.AzureDataLakeArtifactRepository"
    ) as adls_artifact_repo_class_mock:
        mock_adls_repo = mock.MagicMock(autospec=AzureDataLakeArtifactRepository)
        fake_fileinfo = FileInfo(fake_local_path, is_dir=False, file_size=1)
        mock_adls_repo.list_artifacts.return_value = [fake_fileinfo]
        adls_artifact_repo_class_mock.return_value = mock_adls_repo
        request_mock.return_value = _mock_temporary_creds_response(temporary_creds)
        models_repo = UnityCatalogModelsArtifactRepository(
            artifact_uri="models:/MyModel/12", registry_uri=_DATABRICKS_UNITY_CATALOG_SCHEME
        )
        assert models_repo.list_artifacts("artifact_path") == [fake_fileinfo]
        adls_artifact_repo_class_mock.assert_called_once_with(
            artifact_uri=artifact_location, credential=ANY
        )
        adls_repo_args = adls_artifact_repo_class_mock.call_args_list[0]
        credential = adls_repo_args[1]["credential"]
        assert credential.signature == fake_sas_token
        mock_adls_repo.list_artifacts.assert_called_once_with(path="artifact_path")
        request_mock.assert_called_with(
            host_creds=ANY,
            endpoint="/api/2.0/mlflow/unity-catalog/model-versions/generate-temporary-credentials",
            method="POST",
            json={"name": "MyModel", "version": "12", "operation": "MODEL_VERSION_OPERATION_READ"},
        )
