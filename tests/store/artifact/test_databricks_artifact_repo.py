import os
import time
import shutil

import pytest
import posixpath
from requests.models import Response
from unittest import mock
from unittest.mock import ANY

from mlflow.entities.file_info import FileInfo as FileInfoEntity
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_artifacts_pb2 import (
    GetCredentialsForWrite,
    GetCredentialsForRead,
    ArtifactCredentialType,
    ArtifactCredentialInfo,
)
from mlflow.protos.service_pb2 import ListArtifacts, FileInfo
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.databricks_artifact_repo import (
    DatabricksArtifactRepository,
    _MAX_CREDENTIALS_REQUEST_SIZE,
)

DATABRICKS_ARTIFACT_REPOSITORY_PACKAGE = "mlflow.store.artifact.databricks_artifact_repo"
DATABRICKS_ARTIFACT_REPOSITORY = (
    DATABRICKS_ARTIFACT_REPOSITORY_PACKAGE + ".DatabricksArtifactRepository"
)

MOCK_AZURE_SIGNED_URI = "http://this_is_a_mock_sas_for_azure"
MOCK_AWS_SIGNED_URI = "http://this_is_a_mock_presigned_uri_for_aws?"
MOCK_GCP_SIGNED_URL = "http://this_is_a_mock_signed_url_for_gcp?"
MOCK_RUN_ID = "MOCK-RUN-ID"
MOCK_HEADERS = [
    ArtifactCredentialInfo.HttpHeader(name="Mock-Name1", value="Mock-Value1"),
    ArtifactCredentialInfo.HttpHeader(name="Mock-Name2", value="Mock-Value2"),
]
MOCK_RUN_ROOT_URI = "dbfs:/databricks/mlflow-tracking/MOCK-EXP/MOCK-RUN-ID/artifacts"
MOCK_SUBDIR = "subdir/path"
MOCK_SUBDIR_ROOT_URI = posixpath.join(MOCK_RUN_ROOT_URI, MOCK_SUBDIR)


@pytest.fixture()
def databricks_artifact_repo():
    with mock.patch(
        DATABRICKS_ARTIFACT_REPOSITORY + "._get_run_artifact_root"
    ) as get_run_artifact_root_mock:
        get_run_artifact_root_mock.return_value = MOCK_RUN_ROOT_URI
        return get_artifact_repository(
            "dbfs:/databricks/mlflow-tracking/MOCK-EXP/MOCK-RUN-ID/artifacts"
        )


@pytest.fixture()
def test_file(tmpdir):
    test_file_content = "Hello 🍆🍔".encode("utf-8")
    p = tmpdir.join("test.txt")
    with open(p.strpath, "wb") as f:
        f.write(test_file_content)
    return p


@pytest.fixture()
def test_dir(tmpdir):
    test_file_content = "World 🍆🍔🍆".encode("utf-8")
    with open(tmpdir.mkdir("subdir").join("test.txt").strpath, "wb") as f:
        f.write(test_file_content)
    with open(tmpdir.join("test.txt").strpath, "wb") as f:
        f.write(bytes(test_file_content))
    with open(tmpdir.join("empty-file.txt").strpath, "w"):
        pass
    return tmpdir


class TestDatabricksArtifactRepository:
    def test_init_validation_and_cleaning(self):
        with mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + "._get_run_artifact_root"
        ) as get_run_artifact_root_mock:
            get_run_artifact_root_mock.return_value = MOCK_RUN_ROOT_URI
            # Basic artifact uri
            repo = get_artifact_repository(
                "dbfs:/databricks/mlflow-tracking/MOCK-EXP/MOCK-RUN-ID/artifacts"
            )
            assert (
                repo.artifact_uri == "dbfs:/databricks/mlflow-tracking/"
                "MOCK-EXP/MOCK-RUN-ID/artifacts"
            )
            assert repo.run_id == MOCK_RUN_ID
            assert repo.run_relative_artifact_repo_root_path == ""

            with pytest.raises(MlflowException, match="DBFS URI must be of the form dbfs"):
                DatabricksArtifactRepository("s3://test")
            with pytest.raises(MlflowException, match="Artifact URI incorrect"):
                DatabricksArtifactRepository("dbfs:/databricks/mlflow/EXP/RUN/artifact")
            with pytest.raises(MlflowException, match="DBFS URI must be of the form dbfs"):
                DatabricksArtifactRepository(
                    "dbfs://scope:key@notdatabricks/databricks/mlflow-tracking/experiment/1/run/2"
                )

    @pytest.mark.parametrize(
        "artifact_uri, expected_uri, expected_db_uri",
        [
            (
                "dbfs:/databricks/mlflow-tracking/experiment/1/run/2",
                "dbfs:/databricks/mlflow-tracking/experiment/1/run/2",
                "databricks://getTrackingUriDefault",
            ),  # see test body for the mock
            (
                "dbfs://@databricks/databricks/mlflow-tracking/experiment/1/run/2",
                "dbfs:/databricks/mlflow-tracking/experiment/1/run/2",
                "databricks",
            ),
            (
                "dbfs://someProfile@databricks/databricks/mlflow-tracking/experiment/1/run/2",
                "dbfs:/databricks/mlflow-tracking/experiment/1/run/2",
                "databricks://someProfile",
            ),
            (
                "dbfs://scope:key@databricks/databricks/mlflow-tracking/experiment/1/run/2",
                "dbfs:/databricks/mlflow-tracking/experiment/1/run/2",
                "databricks://scope:key",
            ),
            (
                "dbfs:/databricks/mlflow-tracking/MOCK-EXP/MOCK-RUN-ID/artifacts",
                "dbfs:/databricks/mlflow-tracking/MOCK-EXP/MOCK-RUN-ID/artifacts",
                "databricks://getTrackingUriDefault",
            ),
            (
                "dbfs:/databricks/mlflow-tracking/MOCK-EXP/MOCK-RUN-ID/awesome/path",
                "dbfs:/databricks/mlflow-tracking/MOCK-EXP/MOCK-RUN-ID/awesome/path",
                "databricks://getTrackingUriDefault",
            ),
        ],
    )
    def test_init_artifact_uri(self, artifact_uri, expected_uri, expected_db_uri):
        with mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY_PACKAGE + ".get_databricks_host_creds", return_value=None
        ), mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + "._get_run_artifact_root", return_value="whatever"
        ), mock.patch(
            "mlflow.tracking.get_tracking_uri", return_value="databricks://getTrackingUriDefault"
        ):
            repo = DatabricksArtifactRepository(artifact_uri)
            assert repo.artifact_uri == expected_uri
            assert repo.databricks_profile_uri == expected_db_uri

    @pytest.mark.parametrize(
        "artifact_uri, expected_relative_path",
        [
            ("dbfs:/databricks/mlflow-tracking/MOCK-EXP/MOCK-RUN-ID/artifacts", ""),
            ("dbfs:/databricks/mlflow-tracking/MOCK-EXP/MOCK-RUN-ID/artifacts/arty", "arty"),
            (
                "dbfs://prof@databricks/databricks/mlflow-tracking/MOCK-EXP/MOCK-RUN-ID/artifacts/arty",  # pylint: disable=line-too-long
                "arty",
            ),
            (
                "dbfs:/databricks/mlflow-tracking/MOCK-EXP/MOCK-RUN-ID/awesome/path",
                "../awesome/path",
            ),
        ],
    )
    def test_run_relative_artifact_repo_root_path(self, artifact_uri, expected_relative_path):
        with mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + "._get_run_artifact_root"
        ) as get_run_artifact_root_mock:
            get_run_artifact_root_mock.return_value = MOCK_RUN_ROOT_URI
            # Basic artifact uri
            repo = get_artifact_repository(artifact_uri)
            assert repo.run_id == MOCK_RUN_ID
            assert repo.run_relative_artifact_repo_root_path == expected_relative_path

    def test_extract_run_id(self):
        with mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + "._get_run_artifact_root"
        ) as get_run_artifact_root_mock:
            get_run_artifact_root_mock.return_value = MOCK_RUN_ROOT_URI
            expected_run_id = "RUN_ID"
            repo = get_artifact_repository("dbfs:/databricks/mlflow-tracking/EXP/RUN_ID/artifact")
            assert repo.run_id == expected_run_id
            repo = get_artifact_repository(
                "dbfs:/databricks/mlflow-tracking/EXP_ID/RUN_ID/artifacts"
            )
            assert repo.run_id == expected_run_id
            repo = get_artifact_repository(
                "dbfs:/databricks///mlflow-tracking///EXP_ID///RUN_ID///artifacts/"
            )
            assert repo.run_id == expected_run_id
            repo = get_artifact_repository(
                "dbfs:/databricks///mlflow-tracking//EXP_ID//RUN_ID///artifacts//"
            )
            assert repo.run_id == expected_run_id

    @pytest.mark.parametrize(
        "artifact_path, expected_location",
        [(None, "test.txt"), ("output", "output/test.txt"), ("", "test.txt")],
    )
    def test_log_artifact_azure(
        self, databricks_artifact_repo, test_file, artifact_path, expected_location
    ):
        with mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + "._get_write_credential_infos"
        ) as write_credential_infos_mock, mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + "._azure_upload_file"
        ) as azure_upload_mock:
            mock_credential_info = ArtifactCredentialInfo(
                signed_uri=MOCK_AZURE_SIGNED_URI, type=ArtifactCredentialType.AZURE_SAS_URI
            )
            write_credential_infos_mock.return_value = [mock_credential_info]
            azure_upload_mock.return_value = None
            databricks_artifact_repo.log_artifact(test_file.strpath, artifact_path)
            write_credential_infos_mock.assert_called_with(
                run_id=MOCK_RUN_ID, paths=[expected_location]
            )
            azure_upload_mock.assert_called_with(
                mock_credential_info, test_file.strpath, expected_location
            )

    @pytest.mark.parametrize("artifact_path, expected_location", [(None, "test.txt")])
    def test_log_artifact_azure_with_headers(
        self, databricks_artifact_repo, test_file, artifact_path, expected_location
    ):
        mock_azure_headers = {
            "x-ms-encryption-scope": "test-scope",
            "x-ms-tags": "some-tags",
            "x-ms-blob-type": "some-type",
        }
        filtered_azure_headers = {
            "x-ms-encryption-scope": "test-scope",
            "x-ms-tags": "some-tags",
        }
        mock_response = Response()
        mock_response.status_code = 200
        mock_response.close = lambda: None
        with mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + "._get_write_credential_infos"
        ) as write_credential_infos_mock, mock.patch(
            "mlflow.utils.rest_utils.cloud_storage_http_request"
        ) as request_mock:
            mock_credential_info = ArtifactCredentialInfo(
                signed_uri=MOCK_AZURE_SIGNED_URI,
                type=ArtifactCredentialType.AZURE_SAS_URI,
                headers=[
                    ArtifactCredentialInfo.HttpHeader(name=header_name, value=header_value)
                    for header_name, header_value in mock_azure_headers.items()
                ],
            )
            write_credential_infos_mock.return_value = [mock_credential_info]
            request_mock.return_value = mock_response
            databricks_artifact_repo.log_artifact(test_file.strpath, artifact_path)
            write_credential_infos_mock.assert_called_with(
                run_id=MOCK_RUN_ID, paths=[expected_location]
            )
            request_mock.assert_called_with(
                "put",
                MOCK_AZURE_SIGNED_URI + "?comp=blocklist",
                data=ANY,
                headers=filtered_azure_headers,
            )

    def test_log_artifact_azure_blob_client_sas_error(self, databricks_artifact_repo, test_file):
        with mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + "._get_write_credential_infos"
        ) as write_credential_infos_mock, mock.patch(
            "azure.storage.blob.BlobClient.from_blob_url"
        ) as mock_create_blob_client:
            mock_credential_info = ArtifactCredentialInfo(
                signed_uri=MOCK_AZURE_SIGNED_URI, type=ArtifactCredentialType.AZURE_SAS_URI
            )
            write_credential_infos_mock.return_value = [mock_credential_info]
            mock_create_blob_client.side_effect = MlflowException("MOCK ERROR")
            with pytest.raises(MlflowException, match=r".+"):
                databricks_artifact_repo.log_artifact(test_file.strpath)
            write_credential_infos_mock.assert_called_with(run_id=MOCK_RUN_ID, paths=ANY)

    @pytest.mark.parametrize("artifact_path,expected_location", [(None, "test.txt")])
    def test_log_artifact_aws(
        self, databricks_artifact_repo, test_file, artifact_path, expected_location
    ):
        mock_response = Response()
        mock_response.status_code = 200
        mock_response.close = lambda: None
        with mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + "._get_write_credential_infos"
        ) as write_credential_infos_mock, mock.patch(
            "mlflow.utils.rest_utils.cloud_storage_http_request"
        ) as request_mock:
            mock_credential_info = ArtifactCredentialInfo(
                signed_uri=MOCK_AWS_SIGNED_URI, type=ArtifactCredentialType.AWS_PRESIGNED_URL
            )
            write_credential_infos_mock.return_value = [mock_credential_info]
            request_mock.return_value = mock_response
            databricks_artifact_repo.log_artifact(test_file.strpath, artifact_path)
            write_credential_infos_mock.assert_called_with(
                run_id=MOCK_RUN_ID, paths=[expected_location]
            )
            request_mock.assert_called_with("put", MOCK_AWS_SIGNED_URI, data=ANY, headers={})

    @pytest.mark.parametrize("artifact_path,expected_location", [(None, "test.txt")])
    def test_log_artifact_aws_with_headers(
        self, databricks_artifact_repo, test_file, artifact_path, expected_location
    ):
        expected_headers = {header.name: header.value for header in MOCK_HEADERS}
        mock_response = Response()
        mock_response.status_code = 200
        mock_response.close = lambda: None
        with mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + "._get_write_credential_infos"
        ) as write_credential_infos_mock, mock.patch(
            "mlflow.utils.rest_utils.cloud_storage_http_request"
        ) as request_mock:
            mock_credential_info = ArtifactCredentialInfo(
                signed_uri=MOCK_AWS_SIGNED_URI,
                type=ArtifactCredentialType.AWS_PRESIGNED_URL,
                headers=MOCK_HEADERS,
            )
            write_credential_infos_mock.return_value = [mock_credential_info]
            request_mock.return_value = mock_response
            databricks_artifact_repo.log_artifact(test_file.strpath, artifact_path)
            write_credential_infos_mock.assert_called_with(
                run_id=MOCK_RUN_ID, paths=[expected_location]
            )
            request_mock.assert_called_with(
                "put", MOCK_AWS_SIGNED_URI, data=ANY, headers=expected_headers
            )

    def test_log_artifact_aws_presigned_url_error(self, databricks_artifact_repo, test_file):
        with mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + "._get_write_credential_infos"
        ) as write_credential_infos_mock, mock.patch(
            "mlflow.utils.rest_utils.cloud_storage_http_request"
        ) as request_mock:
            mock_credential_info = ArtifactCredentialInfo(
                signed_uri=MOCK_AWS_SIGNED_URI, type=ArtifactCredentialType.AWS_PRESIGNED_URL
            )
            write_credential_infos_mock.return_value = [mock_credential_info]
            request_mock.side_effect = MlflowException("MOCK ERROR")
            with pytest.raises(MlflowException, match="MOCK ERROR"):
                databricks_artifact_repo.log_artifact(test_file.strpath)
            write_credential_infos_mock.assert_called_with(run_id=MOCK_RUN_ID, paths=ANY)

    @pytest.mark.parametrize("artifact_path,expected_location", [(None, "test.txt")])
    def test_log_artifact_gcp(
        self, databricks_artifact_repo, test_file, artifact_path, expected_location
    ):
        mock_response = Response()
        mock_response.status_code = 200
        mock_response.close = lambda: None
        with mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + "._get_write_credential_infos"
        ) as write_credential_infos_mock, mock.patch(
            "mlflow.utils.rest_utils.cloud_storage_http_request"
        ) as request_mock:
            mock_credential_info = ArtifactCredentialInfo(
                signed_uri=MOCK_GCP_SIGNED_URL, type=ArtifactCredentialType.GCP_SIGNED_URL
            )
            write_credential_infos_mock.return_value = [mock_credential_info]
            request_mock.return_value = mock_response
            databricks_artifact_repo.log_artifact(test_file.strpath, artifact_path)
            write_credential_infos_mock.assert_called_with(
                run_id=MOCK_RUN_ID, paths=[expected_location]
            )
            request_mock.assert_called_with("put", MOCK_GCP_SIGNED_URL, data=ANY, headers={})

    @pytest.mark.parametrize("artifact_path,expected_location", [(None, "test.txt")])
    def test_log_artifact_gcp_with_headers(
        self, databricks_artifact_repo, test_file, artifact_path, expected_location
    ):
        expected_headers = {header.name: header.value for header in MOCK_HEADERS}
        mock_response = Response()
        mock_response.status_code = 200
        mock_response.close = lambda: None
        with mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + "._get_write_credential_infos"
        ) as write_credential_infos_mock, mock.patch(
            "mlflow.utils.rest_utils.cloud_storage_http_request"
        ) as request_mock:
            mock_credential_info = ArtifactCredentialInfo(
                signed_uri=MOCK_GCP_SIGNED_URL,
                type=ArtifactCredentialType.GCP_SIGNED_URL,
                headers=MOCK_HEADERS,
            )
            write_credential_infos_mock.return_value = [mock_credential_info]
            request_mock.return_value = mock_response
            databricks_artifact_repo.log_artifact(test_file.strpath, artifact_path)
            write_credential_infos_mock.assert_called_with(
                run_id=MOCK_RUN_ID, paths=[expected_location]
            )
            request_mock.assert_called_with(
                "put", MOCK_GCP_SIGNED_URL, data=ANY, headers=expected_headers
            )

    def test_log_artifact_gcp_presigned_url_error(self, databricks_artifact_repo, test_file):
        with mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + "._get_write_credential_infos"
        ) as write_credential_infos_mock, mock.patch(
            "mlflow.utils.rest_utils.cloud_storage_http_request"
        ) as request_mock:
            mock_credential_info = ArtifactCredentialInfo(
                signed_uri=MOCK_GCP_SIGNED_URL, type=ArtifactCredentialType.GCP_SIGNED_URL
            )
            write_credential_infos_mock.return_value = [mock_credential_info]
            request_mock.side_effect = MlflowException("MOCK ERROR")
            with pytest.raises(MlflowException, match="MOCK ERROR"):
                databricks_artifact_repo.log_artifact(test_file.strpath)
            write_credential_infos_mock.assert_called_with(run_id=MOCK_RUN_ID, paths=ANY)

    @pytest.mark.parametrize(
        "artifact_path, expected_location",
        [
            (None, posixpath.join(MOCK_SUBDIR, "test.txt")),
            ("test_path", posixpath.join(MOCK_SUBDIR, "test_path/test.txt")),
        ],
    )
    def test_log_artifact_with_relative_path(self, test_file, artifact_path, expected_location):
        with mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + "._get_run_artifact_root"
        ) as get_run_artifact_root_mock, mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + "._get_write_credential_infos"
        ) as write_credential_infos_mock, mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + "._upload_to_cloud"
        ) as upload_mock:
            get_run_artifact_root_mock.return_value = MOCK_RUN_ROOT_URI
            databricks_artifact_repo = get_artifact_repository(MOCK_SUBDIR_ROOT_URI)
            mock_credential_info = ArtifactCredentialInfo(
                signed_uri=MOCK_AZURE_SIGNED_URI, type=ArtifactCredentialType.AZURE_SAS_URI
            )
            write_credential_infos_mock.return_value = [mock_credential_info]
            upload_mock.return_value = None
            databricks_artifact_repo.log_artifact(test_file.strpath, artifact_path)
            write_credential_infos_mock.assert_called_with(
                run_id=MOCK_RUN_ID, paths=[expected_location]
            )
            upload_mock.assert_called_with(
                cloud_credential_info=mock_credential_info,
                src_file_path=test_file.strpath,
                dst_run_relative_artifact_path=expected_location,
            )

    def test_list_artifacts(self, databricks_artifact_repo):
        list_artifact_file_proto_mock = [FileInfo(path="a.txt", is_dir=False, file_size=0)]
        list_artifacts_dir_proto_mock = [
            FileInfo(path="test/a.txt", is_dir=False, file_size=100),
            FileInfo(path="test/dir", is_dir=True, file_size=0),
        ]
        with mock.patch(DATABRICKS_ARTIFACT_REPOSITORY + "._call_endpoint") as call_endpoint_mock:
            list_artifact_response_proto = ListArtifacts.Response(
                root_uri="", files=list_artifacts_dir_proto_mock, next_page_token=None
            )
            call_endpoint_mock.return_value = list_artifact_response_proto
            artifacts = databricks_artifact_repo.list_artifacts("test/")
            assert isinstance(artifacts, list)
            assert isinstance(artifacts[0], FileInfoEntity)
            assert len(artifacts) == 2
            assert artifacts[0].path == "test/a.txt"
            assert artifacts[0].is_dir is False
            assert artifacts[0].file_size == 100
            assert artifacts[1].path == "test/dir"
            assert artifacts[1].is_dir is True
            assert artifacts[1].file_size is None

            # Calling list_artifacts() on a path that's a file should return an empty list
            list_artifact_response_proto = ListArtifacts.Response(
                root_uri="", files=list_artifact_file_proto_mock
            )
            call_endpoint_mock.return_value = list_artifact_response_proto
            artifacts = databricks_artifact_repo.list_artifacts("a.txt")
            assert len(artifacts) == 0

    def test_list_artifacts_with_relative_path(self):
        list_artifact_file_proto_mock = [
            FileInfo(path=posixpath.join(MOCK_SUBDIR, "a.txt"), is_dir=False, file_size=0)
        ]
        list_artifacts_dir_proto_mock = [
            FileInfo(path=posixpath.join(MOCK_SUBDIR, "test/a.txt"), is_dir=False, file_size=100),
            FileInfo(path=posixpath.join(MOCK_SUBDIR, "test/dir"), is_dir=True, file_size=0),
        ]
        with mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + "._get_run_artifact_root"
        ) as get_run_artifact_root_mock, mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY_PACKAGE + ".message_to_json"
        ) as message_mock, mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + "._call_endpoint"
        ) as call_endpoint_mock:
            get_run_artifact_root_mock.return_value = MOCK_RUN_ROOT_URI
            list_artifact_response_proto = ListArtifacts.Response(
                root_uri="", files=list_artifacts_dir_proto_mock, next_page_token=None
            )
            call_endpoint_mock.return_value = list_artifact_response_proto
            message_mock.return_value = None
            databricks_artifact_repo = get_artifact_repository(MOCK_SUBDIR_ROOT_URI)
            artifacts = databricks_artifact_repo.list_artifacts("test")
            assert isinstance(artifacts, list)
            assert isinstance(artifacts[0], FileInfoEntity)
            assert len(artifacts) == 2
            assert artifacts[0].path == "test/a.txt"
            assert artifacts[0].is_dir is False
            assert artifacts[0].file_size == 100
            assert artifacts[1].path == "test/dir"
            assert artifacts[1].is_dir is True
            assert artifacts[1].file_size is None
            message_mock.assert_called_with(
                ListArtifacts(run_id=MOCK_RUN_ID, path=posixpath.join(MOCK_SUBDIR, "test"))
            )

            # Calling list_artifacts() on a relative path that's a file should return an empty list
            list_artifact_response_proto = ListArtifacts.Response(
                root_uri="", files=list_artifact_file_proto_mock, next_page_token=None
            )
            call_endpoint_mock.return_value = list_artifact_response_proto
            artifacts = databricks_artifact_repo.list_artifacts("a.txt")
            assert len(artifacts) == 0

    def test_list_artifacts_handles_pagination(self, databricks_artifact_repo):
        list_artifacts_proto_mock_1 = [
            FileInfo(path="a.txt", is_dir=False, file_size=100),
            FileInfo(path="b", is_dir=True, file_size=0),
        ]
        list_artifacts_proto_mock_2 = [
            FileInfo(path="c.txt", is_dir=False, file_size=100),
            FileInfo(path="d", is_dir=True, file_size=0),
        ]
        list_artifacts_proto_mock_3 = [
            FileInfo(path="e.txt", is_dir=False, file_size=100),
            FileInfo(path="f", is_dir=True, file_size=0),
        ]
        list_artifacts_proto_mock_4 = []
        with mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY_PACKAGE + ".message_to_json"
        ) as message_mock, mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + "._call_endpoint"
        ) as call_endpoint_mock:
            list_artifact_paginated_response_protos = [
                ListArtifacts.Response(
                    root_uri="", files=list_artifacts_proto_mock_1, next_page_token="2"
                ),
                ListArtifacts.Response(
                    root_uri="", files=list_artifacts_proto_mock_2, next_page_token="4"
                ),
                ListArtifacts.Response(
                    root_uri="", files=list_artifacts_proto_mock_3, next_page_token="6"
                ),
                ListArtifacts.Response(
                    root_uri="", files=list_artifacts_proto_mock_4, next_page_token="8"
                ),
            ]
            call_endpoint_mock.side_effect = list_artifact_paginated_response_protos
            message_mock.return_value = None
            artifacts = databricks_artifact_repo.list_artifacts()
            assert set(["a.txt", "b", "c.txt", "d", "e.txt", "f"]) == set(
                [file.path for file in artifacts]
            )
            calls = [
                mock.call(ListArtifacts(run_id=MOCK_RUN_ID, path="")),
                mock.call(ListArtifacts(run_id=MOCK_RUN_ID, path="", page_token="2")),
                mock.call(ListArtifacts(run_id=MOCK_RUN_ID, path="", page_token="4")),
                mock.call(ListArtifacts(run_id=MOCK_RUN_ID, path="", page_token="6")),
            ]
            message_mock.assert_has_calls(calls)

    def test_get_read_credential_infos_handles_pagination(self, databricks_artifact_repo):
        """
        Verifies that the `get_read_credential_infos` method, which is used to resolve read access
        credentials for a collection of artifacts, handles paginated responses properly, issuing
        incremental requests until all pages have been consumed
        """
        credential_infos_mock_1 = [
            ArtifactCredentialInfo(
                signed_uri="http://mock_url_1", type=ArtifactCredentialType.AWS_PRESIGNED_URL
            ),
            ArtifactCredentialInfo(
                signed_uri="http://mock_url_2", type=ArtifactCredentialType.AWS_PRESIGNED_URL
            ),
        ]
        credential_infos_mock_2 = [
            ArtifactCredentialInfo(
                signed_uri="http://mock_url_3", type=ArtifactCredentialType.AWS_PRESIGNED_URL
            )
        ]
        credential_infos_mock_3 = []

        with mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY_PACKAGE + ".message_to_json"
        ) as message_mock, mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + "._call_endpoint"
        ) as call_endpoint_mock:
            get_credentials_for_read_responses = [
                GetCredentialsForRead.Response(
                    credential_infos=credential_infos_mock_1, next_page_token="2"
                ),
                GetCredentialsForRead.Response(
                    credential_infos=credential_infos_mock_2, next_page_token="3"
                ),
                GetCredentialsForRead.Response(credential_infos=credential_infos_mock_3),
            ]
            call_endpoint_mock.side_effect = get_credentials_for_read_responses
            read_credential_infos = databricks_artifact_repo._get_read_credential_infos(
                MOCK_RUN_ID,
                ["testpath"],
            )
            assert read_credential_infos == credential_infos_mock_1 + credential_infos_mock_2
            message_mock.assert_has_calls(
                [
                    mock.call(GetCredentialsForRead(run_id=MOCK_RUN_ID, path=["testpath"])),
                    mock.call(
                        GetCredentialsForRead(run_id=MOCK_RUN_ID, path=["testpath"], page_token="2")
                    ),
                    mock.call(
                        GetCredentialsForRead(run_id=MOCK_RUN_ID, path=["testpath"], page_token="3")
                    ),
                ]
            )
            assert call_endpoint_mock.call_count == 3

    def test_get_read_credential_infos_respects_max_request_size(self, databricks_artifact_repo):
        """
        Verifies that the `_get_read_credential_infos` method, which is used to resolve read access
        credentials for a collection of artifacts, handles paginated responses properly, issuing
        incremental requests until all pages have been consumed
        """
        assert _MAX_CREDENTIALS_REQUEST_SIZE == 2000, (
            "The maximum request size configured by the client should be consistent with the"
            " Databricks backend. Only update this value of the backend limit has changed."
        )

        # Create 3 chunks of paths, two of which have the maximum request size and one of which
        # is smaller than the maximum chunk size. Aggregate and pass these to
        # `_get_read_credential_infos`, validating that this method decomposes the aggregate
        # list into these expected chunks and makes 3 separate requests
        paths_chunk_1 = ["path1"] * _MAX_CREDENTIALS_REQUEST_SIZE
        paths_chunk_2 = ["path2"] * _MAX_CREDENTIALS_REQUEST_SIZE
        paths_chunk_3 = ["path3"] * 5
        credential_infos_mock_1 = [
            ArtifactCredentialInfo(
                signed_uri="http://mock_url_1", type=ArtifactCredentialType.AWS_PRESIGNED_URL
            )
            for _ in range(_MAX_CREDENTIALS_REQUEST_SIZE)
        ]
        credential_infos_mock_2 = [
            ArtifactCredentialInfo(
                signed_uri="http://mock_url_2", type=ArtifactCredentialType.AWS_PRESIGNED_URL
            )
            for _ in range(_MAX_CREDENTIALS_REQUEST_SIZE)
        ]
        credential_infos_mock_3 = [
            ArtifactCredentialInfo(
                signed_uri="http://mock_url_3", type=ArtifactCredentialType.AWS_PRESIGNED_URL
            )
            for _ in range(5)
        ]

        with mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY_PACKAGE + ".message_to_json"
        ) as message_mock, mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + "._call_endpoint"
        ) as call_endpoint_mock:
            call_endpoint_mock.side_effect = [
                GetCredentialsForRead.Response(credential_infos=credential_infos_mock_1),
                GetCredentialsForRead.Response(credential_infos=credential_infos_mock_2),
                GetCredentialsForRead.Response(credential_infos=credential_infos_mock_3),
            ]

            databricks_artifact_repo._get_read_credential_infos(
                MOCK_RUN_ID,
                paths_chunk_1 + paths_chunk_2 + paths_chunk_3,
            )
            assert call_endpoint_mock.call_count == 3
            assert message_mock.call_count == 3
            message_mock.assert_has_calls(
                [
                    mock.call(GetCredentialsForRead(run_id=MOCK_RUN_ID, path=paths_chunk_1)),
                    mock.call(GetCredentialsForRead(run_id=MOCK_RUN_ID, path=paths_chunk_2)),
                    mock.call(GetCredentialsForRead(run_id=MOCK_RUN_ID, path=paths_chunk_3)),
                ]
            )

    def test_get_write_credential_infos_handles_pagination(self, databricks_artifact_repo):
        """
        Verifies that the `_get_write_credential_infos` method, which is used to resolve write
        access credentials for a collection of artifacts, handles paginated responses properly,
        issuing incremental requests until all pages have been consumed
        """
        credential_infos_mock_1 = [
            ArtifactCredentialInfo(
                signed_uri="http://mock_url_1", type=ArtifactCredentialType.AWS_PRESIGNED_URL
            ),
            ArtifactCredentialInfo(
                signed_uri="http://mock_url_2", type=ArtifactCredentialType.AWS_PRESIGNED_URL
            ),
        ]
        credential_infos_mock_2 = [
            ArtifactCredentialInfo(
                signed_uri="http://mock_url_3", type=ArtifactCredentialType.AWS_PRESIGNED_URL
            )
        ]
        credential_infos_mock_3 = []

        with mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY_PACKAGE + ".message_to_json"
        ) as message_mock, mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + "._call_endpoint"
        ) as call_endpoint_mock:
            get_credentials_for_write_responses = [
                GetCredentialsForWrite.Response(
                    credential_infos=credential_infos_mock_1, next_page_token="2"
                ),
                GetCredentialsForWrite.Response(
                    credential_infos=credential_infos_mock_2, next_page_token="3"
                ),
                GetCredentialsForWrite.Response(credential_infos=credential_infos_mock_3),
            ]
            call_endpoint_mock.side_effect = get_credentials_for_write_responses
            write_credential_infos = databricks_artifact_repo._get_write_credential_infos(
                MOCK_RUN_ID,
                ["testpath"],
            )
            assert write_credential_infos == credential_infos_mock_1 + credential_infos_mock_2
            message_mock.assert_has_calls(
                [
                    mock.call(GetCredentialsForWrite(run_id=MOCK_RUN_ID, path=["testpath"])),
                    mock.call(
                        GetCredentialsForWrite(
                            run_id=MOCK_RUN_ID, path=["testpath"], page_token="2"
                        )
                    ),
                    mock.call(
                        GetCredentialsForWrite(
                            run_id=MOCK_RUN_ID, path=["testpath"], page_token="3"
                        )
                    ),
                ]
            )
            assert call_endpoint_mock.call_count == 3

    def test_get_write_credential_infos_respects_max_request_size(self, databricks_artifact_repo):
        """
        Verifies that the `_get_write_credential_infos` method, which is used to resolve write
        access credentials for a collection of artifacts, batches requests according to a maximum
        request size configured by the backend
        """
        # Create 3 chunks of paths, two of which have the maximum request size and one of which
        # is smaller than the maximum chunk size. Aggregate and pass these to
        # `_get_write_credential_infos`, validating that this method decomposes the aggregate
        # list into these expected chunks and makes 3 separate requests
        paths_chunk_1 = ["path1"] * 2000
        paths_chunk_2 = ["path2"] * 2000
        paths_chunk_3 = ["path3"] * 5
        credential_infos_mock_1 = [
            ArtifactCredentialInfo(
                signed_uri="http://mock_url_1", type=ArtifactCredentialType.AWS_PRESIGNED_URL
            )
            for _ in range(2000)
        ]
        credential_infos_mock_2 = [
            ArtifactCredentialInfo(
                signed_uri="http://mock_url_2", type=ArtifactCredentialType.AWS_PRESIGNED_URL
            )
            for _ in range(2000)
        ]
        credential_infos_mock_3 = [
            ArtifactCredentialInfo(
                signed_uri="http://mock_url_3", type=ArtifactCredentialType.AWS_PRESIGNED_URL
            )
            for _ in range(5)
        ]

        with mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY_PACKAGE + ".message_to_json"
        ) as message_mock, mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + "._call_endpoint"
        ) as call_endpoint_mock:
            call_endpoint_mock.side_effect = [
                GetCredentialsForWrite.Response(credential_infos=credential_infos_mock_1),
                GetCredentialsForWrite.Response(credential_infos=credential_infos_mock_2),
                GetCredentialsForWrite.Response(credential_infos=credential_infos_mock_3),
            ]

            databricks_artifact_repo._get_write_credential_infos(
                MOCK_RUN_ID,
                paths_chunk_1 + paths_chunk_2 + paths_chunk_3,
            )
            assert call_endpoint_mock.call_count == message_mock.call_count == 3
            message_mock.assert_has_calls(
                [
                    mock.call(GetCredentialsForWrite(run_id=MOCK_RUN_ID, path=paths_chunk_1)),
                    mock.call(GetCredentialsForWrite(run_id=MOCK_RUN_ID, path=paths_chunk_2)),
                    mock.call(GetCredentialsForWrite(run_id=MOCK_RUN_ID, path=paths_chunk_3)),
                ]
            )

    @pytest.mark.parametrize(
        "remote_file_path, local_path, cloud_credential_type",
        [
            ("test_file.txt", "", ArtifactCredentialType.AZURE_SAS_URI),
            ("test_file.txt", None, ArtifactCredentialType.AZURE_SAS_URI),
            ("output/test_file", None, ArtifactCredentialType.AZURE_SAS_URI),
            ("test_file.txt", "", ArtifactCredentialType.AWS_PRESIGNED_URL),
            ("test_file.txt", "", ArtifactCredentialType.GCP_SIGNED_URL),
        ],
    )
    def test_databricks_download_file(
        self, databricks_artifact_repo, remote_file_path, local_path, cloud_credential_type
    ):
        with mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + "._get_read_credential_infos"
        ) as read_credential_infos_mock, mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + ".list_artifacts"
        ) as get_list_mock, mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + "._download_from_cloud"
        ) as download_mock:
            mock_credential_info = ArtifactCredentialInfo(
                signed_uri=MOCK_AZURE_SIGNED_URI, type=cloud_credential_type
            )
            read_credential_infos_mock.return_value = [mock_credential_info]
            download_mock.return_value = None
            get_list_mock.return_value = []
            databricks_artifact_repo.download_artifacts(remote_file_path, local_path)
            read_credential_infos_mock.assert_called_with(
                run_id=MOCK_RUN_ID, paths=[remote_file_path]
            )
            download_mock.assert_called_with(
                cloud_credential_info=mock_credential_info,
                dst_local_file_path=ANY,
            )

    @pytest.mark.parametrize(
        "remote_file_path, local_path", [("test_file.txt", ""), ("test_file.txt", None)]
    )
    def test_databricks_download_file_with_relative_path(self, remote_file_path, local_path):
        with mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + "._get_run_artifact_root"
        ) as get_run_artifact_root_mock, mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + "._get_read_credential_infos"
        ) as read_credential_infos_mock, mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + ".list_artifacts"
        ) as get_list_mock, mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + "._download_from_cloud"
        ) as download_mock:
            get_run_artifact_root_mock.return_value = MOCK_RUN_ROOT_URI
            mock_credential_info = ArtifactCredentialInfo(
                signed_uri=MOCK_AZURE_SIGNED_URI, type=ArtifactCredentialType.AZURE_SAS_URI
            )
            read_credential_infos_mock.return_value = [mock_credential_info]
            download_mock.return_value = None
            get_list_mock.return_value = []
            databricks_artifact_repo = get_artifact_repository(MOCK_SUBDIR_ROOT_URI)
            databricks_artifact_repo.download_artifacts(remote_file_path, local_path)
            read_credential_infos_mock.assert_called_with(
                run_id=MOCK_RUN_ID, paths=[posixpath.join(MOCK_SUBDIR, remote_file_path)]
            )
            download_mock.assert_called_with(
                cloud_credential_info=mock_credential_info,
                dst_local_file_path=ANY,
            )

    def test_databricks_download_file_get_request_fail(self, databricks_artifact_repo, test_file):
        with mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + "._get_read_credential_infos"
        ) as read_credential_infos_mock, mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + ".list_artifacts"
        ) as get_list_mock, mock.patch(
            "requests.get"
        ) as request_mock:
            mock_credential_info = ArtifactCredentialInfo(
                signed_uri=MOCK_AZURE_SIGNED_URI, type=ArtifactCredentialType.AZURE_SAS_URI
            )
            read_credential_infos_mock.return_value = [mock_credential_info]
            get_list_mock.return_value = []
            request_mock.return_value = MlflowException("MOCK ERROR")
            with pytest.raises(MlflowException, match=r".+"):
                databricks_artifact_repo.download_artifacts(test_file.strpath)
            read_credential_infos_mock.assert_called_with(
                run_id=MOCK_RUN_ID, paths=[test_file.strpath]
            )

    def test_download_artifacts_awaits_download_completion(self, databricks_artifact_repo, tmpdir):
        """
        Verifies that all asynchronous artifact downloads are joined before `download_artifacts()`
        returns a result to the caller
        """
        with mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + "._get_read_credential_infos"
        ) as read_credential_infos_mock, mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + ".list_artifacts"
        ) as get_list_mock, mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + "._download_from_cloud"
        ) as download_mock:
            ArtifactCredentialInfo(
                signed_uri=MOCK_AZURE_SIGNED_URI, type=ArtifactCredentialType.AZURE_SAS_URI
            )
            read_credential_infos_mock.return_value = [
                ArtifactCredentialInfo(
                    signed_uri=MOCK_AZURE_SIGNED_URI, type=ArtifactCredentialType.AZURE_SAS_URI
                ),
                ArtifactCredentialInfo(
                    signed_uri=MOCK_AZURE_SIGNED_URI, type=ArtifactCredentialType.AZURE_SAS_URI
                ),
            ]
            get_list_mock.return_value = [
                FileInfo(path="file_1.txt", is_dir=False, file_size=100),
                FileInfo(path="file_2.txt", is_dir=False, file_size=0),
            ]

            def mock_download_from_cloud(
                cloud_credential_info, dst_local_file_path
            ):  # pylint: disable=unused-argument
                # Sleep in order to simulate a longer-running asynchronous download
                time.sleep(2)
                with open(dst_local_file_path, "w") as f:
                    f.write("content")

            download_mock.side_effect = mock_download_from_cloud

            databricks_artifact_repo.download_artifacts("test_path", str(tmpdir))

            expected_file1_path = os.path.join(str(tmpdir), "file_1.txt")
            expected_file2_path = os.path.join(str(tmpdir), "file_2.txt")
            for path in [expected_file1_path, expected_file2_path]:
                assert os.path.exists(path)
                with open(path, "r") as f:
                    assert f.read() == "content"

    def test_artifact_logging(self, databricks_artifact_repo, tmpdir):
        """
        Verifies that `log_artifact()` and `log_artifacts()` initiate all expected asynchronous
        artifact uploads and await their completion before returning results to the caller
        """
        src_dir = os.path.join(str(tmpdir), "src")
        os.makedirs(src_dir)
        src_file1_path = os.path.join(src_dir, "file_1.txt")
        with open(src_file1_path, "w") as f:
            f.write("file1")
        src_file2_path = os.path.join(src_dir, "file_2.txt")
        with open(src_file2_path, "w") as f:
            f.write("file2")

        dst_dir = os.path.join(str(tmpdir), "dst")
        os.makedirs(dst_dir)

        with mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + "._get_write_credential_infos"
        ) as write_credential_infos_mock, mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + "._upload_to_cloud"
        ) as upload_mock:
            write_credential_infos_mock.return_value = [
                ArtifactCredentialInfo(
                    signed_uri=MOCK_AZURE_SIGNED_URI, type=ArtifactCredentialType.AZURE_SAS_URI
                ),
                ArtifactCredentialInfo(
                    signed_uri=MOCK_AZURE_SIGNED_URI, type=ArtifactCredentialType.AZURE_SAS_URI
                ),
            ]

            def mock_upload_to_cloud(
                cloud_credential_info, src_file_path, dst_run_relative_artifact_path
            ):  # pylint: disable=unused-argument
                # Sleep in order to simulate a longer-running asynchronous upload
                time.sleep(2)
                dst_run_relative_artifact_path = os.path.join(
                    dst_dir, dst_run_relative_artifact_path
                )
                os.makedirs(os.path.dirname(dst_run_relative_artifact_path), exist_ok=True)
                shutil.copyfile(src=src_file_path, dst=dst_run_relative_artifact_path)

            upload_mock.side_effect = mock_upload_to_cloud

            databricks_artifact_repo.log_artifacts(src_dir, "dir_artifact")

            expected_dst_dir_file1_path = os.path.join(dst_dir, "dir_artifact", "file_1.txt")
            expected_dst_dir_file2_path = os.path.join(dst_dir, "dir_artifact", "file_2.txt")
            assert os.path.exists(expected_dst_dir_file1_path)
            assert os.path.exists(expected_dst_dir_file2_path)
            with open(expected_dst_dir_file1_path, "r") as f:
                assert f.read() == "file1"
            with open(expected_dst_dir_file2_path, "r") as f:
                assert f.read() == "file2"

            databricks_artifact_repo.log_artifact(src_file1_path)

            expected_dst_file_path = os.path.join(dst_dir, "file_1.txt")
            assert os.path.exists(expected_dst_file_path)
            with open(expected_dst_file_path, "r") as f:
                assert f.read() == "file1"

    def test_download_artifacts_provides_failure_info(self, databricks_artifact_repo):
        with mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + "._get_read_credential_infos"
        ) as read_credentials_mock, mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + ".list_artifacts"
        ) as get_list_mock, mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + "._download_from_cloud"
        ) as download_mock:
            get_list_mock.return_value = [
                FileInfo(path="file_1.txt", is_dir=False, file_size=100),
                FileInfo(path="file_2.txt", is_dir=False, file_size=0),
            ]
            read_credentials_mock.return_value = [
                ArtifactCredentialInfo(
                    signed_uri=MOCK_AZURE_SIGNED_URI, type=ArtifactCredentialType.AZURE_SAS_URI
                ),
                ArtifactCredentialInfo(
                    signed_uri=MOCK_AZURE_SIGNED_URI, type=ArtifactCredentialType.AZURE_SAS_URI
                ),
            ]
            download_mock.side_effect = [
                MlflowException("MOCK ERROR 1"),
                MlflowException("MOCK ERROR 2"),
            ]

            match = (
                r"The following failures occurred while downloading one or more artifacts.+"
                r"MOCK ERROR 1.+"
                r"MOCK ERROR 2"
            )
            with pytest.raises(MlflowException, match=match) as exc:
                databricks_artifact_repo.download_artifacts("test_path")

            err_msg = str(exc.value)
            assert MOCK_RUN_ROOT_URI in err_msg
            assert "file_1.txt" in err_msg
            assert "MOCK ERROR 1" in err_msg
            assert "file_2.txt" in err_msg
            assert "MOCK ERROR 2" in err_msg

    def test_log_artifacts_provides_failure_info(self, databricks_artifact_repo, tmpdir):
        src_file1_path = os.path.join(str(tmpdir), "file_1.txt")
        with open(src_file1_path, "w") as f:
            f.write("file1")
        src_file2_path = os.path.join(str(tmpdir), "file_2.txt")
        with open(src_file2_path, "w") as f:
            f.write("file2")

        with mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + "._get_write_credential_infos"
        ) as write_credentials_mock, mock.patch(
            DATABRICKS_ARTIFACT_REPOSITORY + "._upload_to_cloud"
        ) as upload_mock:
            write_credentials_mock.return_value = [
                ArtifactCredentialInfo(
                    signed_uri=MOCK_AZURE_SIGNED_URI, type=ArtifactCredentialType.AZURE_SAS_URI
                ),
                ArtifactCredentialInfo(
                    signed_uri=MOCK_AZURE_SIGNED_URI, type=ArtifactCredentialType.AZURE_SAS_URI
                ),
            ]
            upload_mock.side_effect = [
                MlflowException("MOCK ERROR 1"),
                MlflowException("MOCK ERROR 2"),
            ]
            match = (
                r"The following failures occurred while uploading one or more artifacts.+"
                r"MOCK ERROR 1.+"
                r"MOCK ERROR 2"
            )
            with pytest.raises(MlflowException, match=match) as exc:
                databricks_artifact_repo.log_artifacts(str(tmpdir), "test_artifacts")

            err_msg = str(exc.value)
            assert MOCK_RUN_ROOT_URI in err_msg
            assert "file_1.txt" in err_msg
            assert "MOCK ERROR 1" in err_msg
            assert "file_2.txt" in err_msg
            assert "MOCK ERROR 2" in err_msg
