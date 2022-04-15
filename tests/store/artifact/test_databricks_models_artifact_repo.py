import json

import pytest
from unittest import mock
from unittest.mock import ANY

from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.databricks_models_artifact_repo import DatabricksModelsArtifactRepository
from mlflow.tracking import MlflowClient

DATABRICKS_MODEL_ARTIFACT_REPOSITORY_PACKAGE = (
    "mlflow.store.artifact.databricks_models_artifact_repo"
)
DATABRICKS_MODEL_ARTIFACT_REPOSITORY = (
    DATABRICKS_MODEL_ARTIFACT_REPOSITORY_PACKAGE + ".DatabricksModelsArtifactRepository"
)
MOCK_MODEL_ROOT_URI_WITH_PROFILE = "models://profile@databricks/MyModel/12"
MOCK_MODEL_ROOT_URI_WITHOUT_PROFILE = "models:/MyModel/12"
MOCK_PROFILE = "databricks://profile"
MOCK_MODEL_NAME = "MyModel"
MOCK_MODEL_VERSION = "12"

REGISTRY_LIST_ARTIFACTS_ENDPOINT = "/api/2.0/mlflow/model-versions/list-artifacts"
REGISTRY_ARTIFACT_PRESIGNED_URI_ENDPOINT = "/api/2.0/mlflow/model-versions/get-signed-download-uri"


@pytest.fixture()
def databricks_model_artifact_repo():
    return DatabricksModelsArtifactRepository(MOCK_MODEL_ROOT_URI_WITH_PROFILE)


class TestDatabricksModelArtifactRepository:
    def test_init_with_version_uri_containing_profile(self):
        repo = DatabricksModelsArtifactRepository(MOCK_MODEL_ROOT_URI_WITH_PROFILE)
        assert repo.artifact_uri == MOCK_MODEL_ROOT_URI_WITH_PROFILE
        assert repo.model_name == MOCK_MODEL_NAME
        assert repo.model_version == MOCK_MODEL_VERSION

    @pytest.mark.parametrize(
        "stage_uri_with_profile",
        [
            "models://profile@databricks/MyModel/Staging",
            "models://profile@databricks/MyModel/Production",
        ],
    )
    def test_init_with_stage_uri_containing_profile(self, stage_uri_with_profile):
        model_version_detailed = ModelVersion(
            MOCK_MODEL_NAME,
            MOCK_MODEL_VERSION,
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
            repo = DatabricksModelsArtifactRepository(stage_uri_with_profile)
            assert repo.artifact_uri == stage_uri_with_profile
            assert repo.model_name == MOCK_MODEL_NAME
            assert repo.model_version == MOCK_MODEL_VERSION
            assert repo.databricks_profile_uri == MOCK_PROFILE

    @pytest.mark.parametrize(
        "invalid_artifact_uri",
        [
            "s3://test",
            "dbfs:/databricks/mlflow/MV-id/models",
            "dbfs://scope:key@notdatabricks/databricks/mlflow-regisry/123/models",
            "models:/MyModel/12",
            "models://scope:key@notdatabricks/MyModel/12",
        ],
    )
    def test_init_with_invalid_artifact_uris(self, invalid_artifact_uri):
        with pytest.raises(
            MlflowException,
            match="A valid databricks profile is required to instantiate this repository",
        ):
            DatabricksModelsArtifactRepository(invalid_artifact_uri)

    def test_init_with_version_uri_and_profile_is_inferred(self):
        # First mock for `is_using_databricks_registry` to pass
        # Second mock to set `databricks_profile_uri` during instantiation
        with mock.patch(
            "mlflow.store.artifact.utils.models.mlflow.get_registry_uri",
            return_value=MOCK_PROFILE,
        ), mock.patch("mlflow.tracking.get_registry_uri", return_value=MOCK_PROFILE):
            repo = DatabricksModelsArtifactRepository(MOCK_MODEL_ROOT_URI_WITHOUT_PROFILE)
            assert repo.artifact_uri == MOCK_MODEL_ROOT_URI_WITHOUT_PROFILE
            assert repo.model_name == MOCK_MODEL_NAME
            assert repo.model_version == MOCK_MODEL_VERSION
            assert repo.databricks_profile_uri == MOCK_PROFILE

    @pytest.mark.parametrize(
        "stage_uri_without_profile",
        ["models:/MyModel/Staging", "models:/MyModel/Production"],
    )
    def test_init_with_stage_uri_and_profile_is_inferred(self, stage_uri_without_profile):
        model_version_detailed = ModelVersion(
            MOCK_MODEL_NAME,
            MOCK_MODEL_VERSION,
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
        with get_latest_versions_patch, mock.patch(
            "mlflow.store.artifact.utils.models.mlflow.get_registry_uri",
            return_value=MOCK_PROFILE,
        ), mock.patch("mlflow.tracking.get_registry_uri", return_value=MOCK_PROFILE):
            repo = DatabricksModelsArtifactRepository(stage_uri_without_profile)
            assert repo.artifact_uri == stage_uri_without_profile
            assert repo.model_name == MOCK_MODEL_NAME
            assert repo.model_version == MOCK_MODEL_VERSION
            assert repo.databricks_profile_uri == MOCK_PROFILE

    @pytest.mark.parametrize(
        "valid_profileless_artifact_uri",
        ["models:/MyModel/12", "models:/MyModel/Staging"],
    )
    def test_init_with_valid_uri_but_no_profile(self, valid_profileless_artifact_uri):
        # Mock for `is_using_databricks_registry` fail when calling `get_registry_uri`
        with mock.patch(
            "mlflow.store.artifact.utils.models.mlflow.get_registry_uri",
            return_value=None,
        ):
            with pytest.raises(
                MlflowException,
                match="A valid databricks profile is required to instantiate this repository",
            ):
                DatabricksModelsArtifactRepository(valid_profileless_artifact_uri)

    def test_list_artifacts(self, databricks_model_artifact_repo):
        status_code = 200

        def _raise_for_status():
            if status_code == 404:
                raise Exception(
                    "404 Client Error: Not Found for url: https://shard-uri/api/2.0/mlflow/model-versions/list-artifacts?name=model&version=1"
                )

        list_artifact_dir_response_mock = mock.MagicMock()
        list_artifact_dir_response_mock.status_code = status_code
        list_artifact_dir_json_mock = {
            "files": [
                {"path": "MLmodel", "is_dir": False, "file_size": 294},
                {"path": "data", "is_dir": True, "file_size": None},
            ]
        }
        list_artifact_dir_response_mock.text = json.dumps(list_artifact_dir_json_mock)
        list_artifact_dir_response_mock.raise_for_status.side_effect = _raise_for_status
        with mock.patch(
            DATABRICKS_MODEL_ARTIFACT_REPOSITORY + "._call_endpoint"
        ) as call_endpoint_mock:
            call_endpoint_mock.return_value = list_artifact_dir_response_mock
            artifacts = databricks_model_artifact_repo.list_artifacts("")
            assert isinstance(artifacts, list)
            assert len(artifacts) == 2
            assert artifacts[0].path == "MLmodel"
            assert artifacts[0].is_dir is False
            assert artifacts[0].file_size == 294
            assert artifacts[1].path == "data"
            assert artifacts[1].is_dir is True
            assert artifacts[1].file_size is None
            call_endpoint_mock.assert_called_once_with(ANY, REGISTRY_LIST_ARTIFACTS_ENDPOINT)

        # errors from API are propagated through to cli response
        list_artifact_dir_bad_response_mock = mock.MagicMock()
        status_code = 404
        list_artifact_dir_bad_response_mock.status_code = status_code
        list_artifact_dir_bad_response_mock.text = "An error occurred"
        list_artifact_dir_bad_response_mock.raise_for_status.side_effect = _raise_for_status
        with mock.patch(
            DATABRICKS_MODEL_ARTIFACT_REPOSITORY + "._call_endpoint"
        ) as call_endpoint_mock:
            call_endpoint_mock.return_value = list_artifact_dir_bad_response_mock
            with pytest.raises(
                MlflowException,
                match=r"API request to list files under path `` failed with status code 404. "
                "Response body: An error occurred",
            ):
                databricks_model_artifact_repo.list_artifacts("")
            call_endpoint_mock.assert_called_once_with(ANY, REGISTRY_LIST_ARTIFACTS_ENDPOINT)

    def test_list_artifacts_for_single_file(self, databricks_model_artifact_repo):
        list_artifact_file_response_mock = mock.MagicMock()
        list_artifact_file_response_mock.status_code = 200
        list_artifact_file_json_mock = {
            "files": [{"path": "MLmodel", "is_dir": False, "file_size": 294}]
        }
        list_artifact_file_response_mock.text = json.dumps(list_artifact_file_json_mock)
        with mock.patch(
            DATABRICKS_MODEL_ARTIFACT_REPOSITORY + "._call_endpoint"
        ) as call_endpoint_mock:
            # Calling list_artifacts() on a path that's a file should return an empty list
            call_endpoint_mock.return_value = list_artifact_file_response_mock
            artifacts = databricks_model_artifact_repo.list_artifacts("MLmodel")
            assert len(artifacts) == 0

    @pytest.mark.parametrize(
        "remote_file_path, local_path",
        [
            ("test_file.txt", ""),
            ("test_file.txt", None),
            ("output/test_file", None),
            ("test_file.txt", ""),
        ],
    )
    def test_download_file(self, databricks_model_artifact_repo, remote_file_path, local_path):
        signed_uri_response_mock = mock.MagicMock()
        signed_uri_response_mock.status_code = 200
        signed_uri_mock = {
            "signed_uri": "https://my-amazing-signed-uri-to-rule-them-all.com/1234-numbers-yay-567"
        }
        signed_uri_response_mock.text = json.dumps(signed_uri_mock)
        with mock.patch(
            DATABRICKS_MODEL_ARTIFACT_REPOSITORY + "._call_endpoint"
        ) as call_endpoint_mock, mock.patch(
            DATABRICKS_MODEL_ARTIFACT_REPOSITORY_PACKAGE + ".download_file_using_http_uri"
        ) as download_mock:
            call_endpoint_mock.return_value = signed_uri_response_mock
            download_mock.return_value = None
            databricks_model_artifact_repo.download_artifacts(remote_file_path, local_path)
            call_endpoint_mock.assert_called_with(ANY, REGISTRY_ARTIFACT_PRESIGNED_URI_ENDPOINT)
            download_mock.assert_called_with(signed_uri_mock["signed_uri"], ANY, ANY)

    def test_download_file_get_request_fail(self, databricks_model_artifact_repo):
        with mock.patch(
            DATABRICKS_MODEL_ARTIFACT_REPOSITORY + "._call_endpoint"
        ) as call_endpoint_mock:
            call_endpoint_mock.side_effect = MlflowException("MOCK ERROR")
            with pytest.raises(MlflowException, match=r".+"):
                databricks_model_artifact_repo.download_artifacts("Something")

    def test_log_artifact_fail(self, databricks_model_artifact_repo):
        with pytest.raises(
            MlflowException, match="This repository does not support logging artifacts"
        ):
            databricks_model_artifact_repo.log_artifact("Some file")

    def test_log_artifacts_fail(self, databricks_model_artifact_repo):
        with pytest.raises(
            MlflowException, match="This repository does not support logging artifacts"
        ):
            databricks_model_artifact_repo.log_artifacts("Some dir")

    def test_delete_artifacts_fail(self, databricks_model_artifact_repo):
        with pytest.raises(
            NotImplementedError,
            match="This artifact repository does not support deleting artifacts",
        ):
            databricks_model_artifact_repo.delete_artifacts()
