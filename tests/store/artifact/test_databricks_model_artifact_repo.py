import pytest
from unittest import mock
import json
from unittest.mock import ANY
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.databricks_model_artifact_repo import DatabricksModelArtifactRepository

DATABRICKS_MODEL_ARTIFACT_REPOSITORY_PACKAGE = (
    "mlflow.store.artifact.databricks_model_artifact_repo"
)
DATABRICKS_MODEL_ARTIFACT_REPOSITORY = (
    DATABRICKS_MODEL_ARTIFACT_REPOSITORY_PACKAGE + ".DatabricksModelArtifactRepository"
)
MOCK_MODEL_ROOT_URI_WITH_PROFILE = "models://profile@databricks/MyModel/12"
MOCK_MODEL_ROOT_URI_WITHOUT_PROFILE = "models:/MyModel/12"
MOCK_MODEL_NAME = "MyModel"
MOCK_MODEL_VERSION = "12"

REGISTRY_LIST_ENDPOINT = "/api/2.0/mlflow/model-versions/list-artifacts"
REGISTRY_GET_PRESIGNED_URI_ENDPOINT = "/api/2.0/mlflow/model-versions/get-signed-download-uri"


@pytest.fixture()
def host_creds_mock():
    with mock.patch(
        DATABRICKS_MODEL_ARTIFACT_REPOSITORY_PACKAGE + ".get_databricks_host_creds"
    ) as get_creds_mock:
        get_creds_mock.return_value = None
        yield


@pytest.fixture()
def databricks_model_artifact_repo(host_creds_mock):  # pylint: disable=unused-argument
    return DatabricksModelArtifactRepository(MOCK_MODEL_ROOT_URI_WITH_PROFILE)


class TestDatabricksModelArtifactRepository(object):
    def test_init_validation_and_cleaning(self):
        with mock.patch(
            DATABRICKS_MODEL_ARTIFACT_REPOSITORY_PACKAGE + ".get_databricks_host_creds"
        ) as get_creds_mock:
            get_creds_mock.return_value = None
            repo = DatabricksModelArtifactRepository(MOCK_MODEL_ROOT_URI_WITH_PROFILE)
            assert repo.artifact_uri == MOCK_MODEL_ROOT_URI_WITH_PROFILE
            assert repo.model_name == MOCK_MODEL_NAME
            assert repo.model_version == MOCK_MODEL_VERSION

        with pytest.raises(MlflowException):
            DatabricksModelArtifactRepository("s3://test")
        with pytest.raises(MlflowException):
            DatabricksModelArtifactRepository("dbfs:/databricks/mlflow/EXP/RUN/artifact")
        with pytest.raises(MlflowException):
            DatabricksModelArtifactRepository(
                "dbfs://scope:key@notdatabricks/databricks/mlflow-tracking/experiment/1/run/2"
            )
        with pytest.raises(MlflowException):
            DatabricksModelArtifactRepository("models:/MyModel/12")
        with pytest.raises(MlflowException):
            DatabricksModelArtifactRepository("models://scope:key@notdatabricks/MyModel/12")

    def test_init_when_profile_is_infered(self):
        with mock.patch(
            DATABRICKS_MODEL_ARTIFACT_REPOSITORY_PACKAGE + ".get_databricks_host_creds"
        ) as get_creds_mock, mock.patch(
            "mlflow.tracking.get_registry_uri", return_value="databricks://getRegistryUriDefault"
        ), mock.patch(
            DATABRICKS_MODEL_ARTIFACT_REPOSITORY_PACKAGE + ".is_databricks_profile",
            return_value=True,
        ):
            get_creds_mock.return_value = None
            repo = DatabricksModelArtifactRepository(MOCK_MODEL_ROOT_URI_WITHOUT_PROFILE)
            assert repo.artifact_uri == MOCK_MODEL_ROOT_URI_WITHOUT_PROFILE
            assert repo.model_name == MOCK_MODEL_NAME
            assert repo.model_version == MOCK_MODEL_VERSION

    def test_list_artifacts(self, databricks_model_artifact_repo):
        list_artifact_dir_response_mock = mock.MagicMock
        list_artifact_dir_response_mock.status_code = 200
        list_artifact_dir_json_mock = {
            "files": [
                {"path": "MLmodel", "is_dir": False, "file_size": 294},
                {"path": "data", "is_dir": True, "file_size": 0},
            ]
        }
        list_artifact_dir_response_mock.text = json.dumps(list_artifact_dir_json_mock)
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
            call_endpoint_mock.assert_called_with(ANY, REGISTRY_LIST_ENDPOINT)

    def test_list_artifacts_for_single_file(self, databricks_model_artifact_repo):
        list_artifact_file_response_mock = mock.MagicMock
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
    def test_databricks_download_file(
        self, databricks_model_artifact_repo, remote_file_path, local_path
    ):
        signed_uri_response_mock = mock.MagicMock
        signed_uri_response_mock.status_code = 200
        signed_uri_mock = {
            "signed_uri": "https://my-amazing-signed-uri-to-rule-them-all.com/1234-numbers-yay-567"
        }
        signed_uri_response_mock.text = json.dumps(signed_uri_mock)
        with mock.patch(
            DATABRICKS_MODEL_ARTIFACT_REPOSITORY + "._call_endpoint"
        ) as call_endpoint_mock, mock.patch(
            DATABRICKS_MODEL_ARTIFACT_REPOSITORY_PACKAGE + ".download_file_using_signed_uri"
        ) as download_mock:
            call_endpoint_mock.return_value = signed_uri_response_mock
            download_mock.return_value = None
            databricks_model_artifact_repo.download_artifacts(remote_file_path, local_path)
            call_endpoint_mock.assert_called_with(ANY, REGISTRY_GET_PRESIGNED_URI_ENDPOINT)
            download_mock.assert_called_with(signed_uri_mock["signed_uri"], ANY, ANY)

    def test_databricks_download_file_get_request_fail(self, databricks_model_artifact_repo):
        with mock.patch(
            DATABRICKS_MODEL_ARTIFACT_REPOSITORY + "._call_endpoint"
        ) as call_endpoint_mock:
            call_endpoint_mock.side_effect = MlflowException("MOCK ERROR")
            with pytest.raises(MlflowException):
                databricks_model_artifact_repo.download_artifacts("Something")
