from unittest import mock
from unittest.mock import ANY
import requests

from mlflow import MlflowClient
from mlflow.store.artifact.optimized_s3_artifact_repo import OptimizedS3ArtifactRepository
from mlflow.store.artifact.unity_catalog_oss_models_artifact_repo import (
    UnityCatalogOSSModelsArtifactRepository,
)
from mlflow.utils.uri import _OSS_UNITY_CATALOG_SCHEME

from tests.store.artifact.test_unity_catalog_models_artifact_repo import (
    _mock_temporary_creds_response,
)

MODELS_ARTIFACT_REPOSITORY_PACKAGE = "mlflow.store.artifact.unity_catalog_models_artifact_repo"
MODELS_ARTIFACT_REPOSITORY = (
    MODELS_ARTIFACT_REPOSITORY_PACKAGE + ".UnityCatalogModelsArtifactRepository"
)


def test_uc_models_artifact_repo_init_with_uri_containing_profile():
    uri_with_profile = "models://profile@uc/MyModel/12"
    models_repo = UnityCatalogOSSModelsArtifactRepository(
        uri_with_profile, _OSS_UNITY_CATALOG_SCHEME
    )
    assert models_repo.artifact_uri == uri_with_profile
    assert models_repo.client._registry_uri == f"{_OSS_UNITY_CATALOG_SCHEME}://profile"


def test_uc_models_artifact_repo_scoped_token_oss(monkeypatch):
    monkeypatch.setenvs(
        {
            "DATABRICKS_HOST": "my-host",
            "DATABRICKS_TOKEN": "my-token",
        }
    )
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
    with mock.patch("mlflow.utils.oss_utils.get_oss_host_creds"), mock.patch.object(
        MlflowClient, "get_model_version_download_uri", return_value=artifact_location
    ), mock.patch("mlflow.utils.rest_utils.http_request") as request_mock, mock.patch(
        "mlflow.store.artifact.optimized_s3_artifact_repo.OptimizedS3ArtifactRepository"
    ) as optimized_s3_artifact_repo_class_mock:
        mock_s3_repo = mock.MagicMock(autospec=OptimizedS3ArtifactRepository)
        mock_s3_repo.download_artifacts.return_value = fake_local_path
        optimized_s3_artifact_repo_class_mock.return_value = mock_s3_repo
        request_mock.return_value = _mock_temporary_creds_response(temporary_creds)
        models_repo = UnityCatalogOSSModelsArtifactRepository(
            artifact_uri="models:/test-catalog.test-schema.test-model/12",
            registry_uri=_OSS_UNITY_CATALOG_SCHEME,
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

def test_protos_light():
    import mlflow

    mlflow.set_registry_uri("uc:databricks-uc:http://localhost:8081/api/2.1/unity-catalog")
    model = mlflow.MlflowClient().create_registered_model("artjen.rohit.rando_model_1")
    print("Model Details", model)

def test_protos():
    import mlflow

    mlflow.set_registry_uri("uc:databricks-uc:http://localhost:8081/api/2.1/unity-catalog")
    catalog_name = "artjen"
    schema_name = "rohit"

    catalog_payload = {
        "name": catalog_name,
        "comment": "This is a test catalog"
    }

    schema_payload = {
        "name": schema_name,
        "catalog_name": catalog_name,
        "comment": "This is a test schema"
    }

    model_payload = {
      "name": "test-model",
      "catalog_name": "artjen",
      "schema_name": "rohit",
      "comment": "test"
    }

    # catalog_response = requests.post("http://localhost:8081/api/2.1/unity-catalog/catalogs", json=catalog_payload)
    # print(catalog_response.text)
    # schema_response = requests.post("http://localhost:8081/api/2.1/unity-catalog/schemas", json=schema_payload)
    # print(schema_response.text)
    # model_response = requests.post("http://localhost:8081/api/2.1/unity-catalog/models", json=model_payload)
    # print(model_response.text)
    # response = requests.get("http://localhost:8081/api/2.1/unity-catalog/models?catalog_name=artjen&schema_name=rohit")
    # print(response)
    # model = mlflow.MlflowClient().create_registered_model("artjen.rohit.test-model-2")
    # print(model)
    # del_response = requests.delete("http://localhost:8081/api/2.1/unity-catalog/models/artjen.rohit.test-model-2")
    # print(del_response.text)

    # Test Create Registered Model
    model = mlflow.MlflowClient().create_registered_model("artjen.rohit.is_going_to_be_deleted")
    print("Model Details", model)

    # Test Delete Registered Model
    delete_model_version = mlflow.MlflowClient().delete_registered_model(name="artjen.rohit.is_going_to_be_deleted")
    print("Successfully Deleted Model")

    model = mlflow.MlflowClient().create_registered_model("artjen.rohit.testmodel123")
    print("Important Model Details", model)

    update_model = mlflow.MlflowClient().update_registered_model(name="artjen.rohit.testmodel123", description="test")
    print(update_model)

    get_model = mlflow.MlflowClient().get_registered_model("artjen.rohit.testmodel123")
    print(get_model)

    create_model_version_with_artifact("artjen.rohit.testmodel123") #updates model

    update_model_version = mlflow.MlflowClient().update_model_version(name="artjen.rohit.testmodel123", version=1, description="hello")
    print(update_model_version)

    model_ver = mlflow.MlflowClient().get_model_version(name="artjen.rohit.testmodel123", version=1)
    print(model_ver)

    delete_model_version = mlflow.MlflowClient().delete_model_version(name="artjen.rohit.testmodel123", version=1)
    print("Successfully Deleted Model Version!")

    final_del_response = mlflow.MlflowClient().delete_registered_model("artjen.rohit.testmodel123")
    print("Successfully Deleted Model after deleting versions")

def create_model_version_with_artifact(full_name):
    import mlflow
    from sklearn import datasets
    from sklearn.ensemble import RandomForestClassifier

    with mlflow.start_run():
        # Train a sklearn model on the iris dataset
        X, y = datasets.load_iris(return_X_y=True, as_frame=True)
        clf = RandomForestClassifier(max_depth=7)
        clf.fit(X, y)
        # Take the first row of the training dataset as the model input example.
        input_example = X.iloc[[0]]
        # Log the model and register it as a new version in UC.
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            # The signature is automatically inferred from the input example and its predicted output.
            input_example=input_example,
            registered_model_name=full_name,
        )
