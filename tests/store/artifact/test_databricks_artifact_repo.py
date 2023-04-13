import os
import time
import shutil
import json
import re

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
    CreateMultipartUpload,
    CompleteMultipartUpload,
    GetPresignedUploadPartUrl,
)
from mlflow.protos.service_pb2 import ListArtifacts, FileInfo
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.store.artifact.databricks_artifact_repo import (
    DatabricksArtifactRepository,
    _MAX_CREDENTIALS_REQUEST_SIZE,
)

DATABRICKS_ARTIFACT_REPOSITORY_PACKAGE = "mlflow.store.artifact.databricks_artifact_repo"
DATABRICKS_ARTIFACT_REPOSITORY = (
    f"{DATABRICKS_ARTIFACT_REPOSITORY_PACKAGE}.DatabricksArtifactRepository"
)

MOCK_AZURE_SIGNED_URI = "http://this_is_a_mock_sas_for_azure"
MOCK_ADLS_GEN2_SIGNED_URI = "http://this_is_a_mock_sas_for_adls_gen2"
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
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._get_run_artifact_root", return_value=MOCK_RUN_ROOT_URI
    ):
        return get_artifact_repository(
            "dbfs:/databricks/mlflow-tracking/MOCK-EXP/MOCK-RUN-ID/artifacts"
        )


@pytest.fixture()
def test_file(tmpdir):
    test_file_content = "Hello 🍆🍔".encode()
    p = tmpdir.join("test.txt")
    with open(p.strpath, "wb") as f:
        f.write(test_file_content)
    return p


@pytest.fixture()
def test_dir(tmpdir):
    test_file_content = "World 🍆🍔🍆".encode()
    with open(tmpdir.mkdir("subdir").join("test.txt").strpath, "wb") as f:
        f.write(test_file_content)
    with open(tmpdir.join("test.txt").strpath, "wb") as f:
        f.write(bytes(test_file_content))
    with open(tmpdir.join("empty-file.txt").strpath, "w"):
        pass
    return tmpdir


def test_init_validation_and_cleaning():
    with mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._get_run_artifact_root",
        return_value=MOCK_RUN_ROOT_URI,
    ):
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
    ("artifact_uri", "expected_uri", "expected_db_uri"),
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
def test_init_artifact_uri(artifact_uri, expected_uri, expected_db_uri):
    with mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY_PACKAGE}.get_databricks_host_creds", return_value=None
    ), mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._get_run_artifact_root", return_value="whatever"
    ), mock.patch(
        "mlflow.tracking.get_tracking_uri", return_value="databricks://getTrackingUriDefault"
    ):
        repo = DatabricksArtifactRepository(artifact_uri)
        assert repo.artifact_uri == expected_uri
        assert repo.databricks_profile_uri == expected_db_uri


@pytest.mark.parametrize(
    ("artifact_uri", "expected_relative_path"),
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
def test_run_relative_artifact_repo_root_path(artifact_uri, expected_relative_path):
    with mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._get_run_artifact_root",
        return_value=MOCK_RUN_ROOT_URI,
    ):
        # Basic artifact uri
        repo = get_artifact_repository(artifact_uri)
        assert repo.run_id == MOCK_RUN_ID
        assert repo.run_relative_artifact_repo_root_path == expected_relative_path


def test_extract_run_id():
    with mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._get_run_artifact_root",
        return_value=MOCK_RUN_ROOT_URI,
    ):
        expected_run_id = "RUN_ID"
        repo = get_artifact_repository("dbfs:/databricks/mlflow-tracking/EXP/RUN_ID/artifact")
        assert repo.run_id == expected_run_id
        repo = get_artifact_repository("dbfs:/databricks/mlflow-tracking/EXP_ID/RUN_ID/artifacts")
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
    ("artifact_path", "expected_location"),
    [(None, "test.txt"), ("output", "output/test.txt"), ("", "test.txt")],
)
def test_log_artifact_azure(databricks_artifact_repo, test_file, artifact_path, expected_location):
    mock_credential_info = ArtifactCredentialInfo(
        signed_uri=MOCK_AZURE_SIGNED_URI, type=ArtifactCredentialType.AZURE_SAS_URI
    )
    with mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._get_write_credential_infos",
        return_value=[mock_credential_info],
    ) as write_credential_infos_mock, mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._azure_upload_file", return_value=None
    ) as azure_upload_mock:
        databricks_artifact_repo.log_artifact(test_file.strpath, artifact_path)
        write_credential_infos_mock.assert_called_with(
            run_id=MOCK_RUN_ID, paths=[expected_location]
        )
        azure_upload_mock.assert_called_with(
            mock_credential_info, test_file.strpath, expected_location
        )


@pytest.mark.parametrize(("artifact_path", "expected_location"), [(None, "test.txt")])
def test_log_artifact_azure_with_headers(
    databricks_artifact_repo, test_file, artifact_path, expected_location
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
    mock_credential_info = ArtifactCredentialInfo(
        signed_uri=MOCK_AZURE_SIGNED_URI,
        type=ArtifactCredentialType.AZURE_SAS_URI,
        headers=[
            ArtifactCredentialInfo.HttpHeader(name=header_name, value=header_value)
            for header_name, header_value in mock_azure_headers.items()
        ],
    )
    with mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._get_write_credential_infos",
        return_value=[mock_credential_info],
    ) as write_credential_infos_mock, mock.patch(
        "requests.Session.request", return_value=mock_response
    ) as request_mock:
        databricks_artifact_repo.log_artifact(test_file.strpath, artifact_path)
        write_credential_infos_mock.assert_called_with(
            run_id=MOCK_RUN_ID, paths=[expected_location]
        )
        request_mock.assert_called_with(
            "put",
            f"{MOCK_AZURE_SIGNED_URI}?comp=blocklist",
            data=ANY,
            headers=filtered_azure_headers,
            timeout=None,
        )


def test_log_artifact_azure_blob_client_sas_error(databricks_artifact_repo, test_file):
    mock_credential_info = ArtifactCredentialInfo(
        signed_uri=MOCK_AZURE_SIGNED_URI, type=ArtifactCredentialType.AZURE_SAS_URI
    )
    with mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._get_write_credential_infos",
        return_value=[mock_credential_info],
    ) as write_credential_infos_mock, mock.patch(
        "requests.Session.request", side_effect=MlflowException("MOCK ERROR")
    ):
        with pytest.raises(MlflowException, match=r"MOCK ERROR"):
            databricks_artifact_repo.log_artifact(test_file.strpath)
        write_credential_infos_mock.assert_called_with(run_id=MOCK_RUN_ID, paths=ANY)


@pytest.mark.parametrize(
    ("artifact_path", "expected_location"),
    [(None, "test.txt"), ("output", "output/test.txt"), ("", "test.txt")],
)
def test_log_artifact_adls_gen2(
    databricks_artifact_repo, test_file, artifact_path, expected_location
):
    mock_credential_info = ArtifactCredentialInfo(
        signed_uri=MOCK_ADLS_GEN2_SIGNED_URI,
        type=ArtifactCredentialType.AZURE_ADLS_GEN2_SAS_URI,
    )
    with mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._get_write_credential_infos",
        return_value=[mock_credential_info],
    ) as write_credential_infos_mock, mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._azure_adls_gen2_upload_file", return_value=None
    ) as azure_adls_gen2_upload_mock:
        databricks_artifact_repo.log_artifact(test_file.strpath, artifact_path)
        write_credential_infos_mock.assert_called_with(
            run_id=MOCK_RUN_ID, paths=[expected_location]
        )
        azure_adls_gen2_upload_mock.assert_called_with(
            mock_credential_info, test_file.strpath, expected_location
        )


@pytest.mark.parametrize(("artifact_path", "expected_location"), [(None, "test.txt")])
def test_log_artifact_adls_gen2_with_headers(
    databricks_artifact_repo, test_file, artifact_path, expected_location
):
    mock_azure_headers = {
        "x-ms-content-type": "test-type",
        "x-ms-owner": "some-owner",
        "x-ms-something_not_supported": "some-value",
    }
    filtered_azure_headers = {
        "x-ms-content-type": "test-type",
        "x-ms-owner": "some-owner",
    }
    mock_response = Response()
    mock_response.status_code = 200
    mock_response.close = lambda: None
    mock_credential_info = ArtifactCredentialInfo(
        signed_uri=MOCK_ADLS_GEN2_SIGNED_URI,
        type=ArtifactCredentialType.AZURE_ADLS_GEN2_SAS_URI,
        headers=[
            ArtifactCredentialInfo.HttpHeader(name=header_name, value=header_value)
            for header_name, header_value in mock_azure_headers.items()
        ],
    )
    with mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._get_write_credential_infos",
        return_value=[mock_credential_info],
    ) as write_credential_infos_mock, mock.patch(
        "requests.Session.request", return_value=mock_response
    ) as request_mock, mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY_PACKAGE}._MULTIPART_UPLOAD_CHUNK_SIZE",
        5,
    ):
        databricks_artifact_repo.log_artifact(test_file.strpath, artifact_path)
        write_credential_infos_mock.assert_called_with(
            run_id=MOCK_RUN_ID, paths=[expected_location]
        )
        # test with block size 5
        request_mock.assert_any_call(
            "put",
            f"{MOCK_ADLS_GEN2_SIGNED_URI}?resource=file",
            headers=filtered_azure_headers,
            timeout=None,
        )
        request_mock.assert_any_call(
            "patch",
            f"{MOCK_ADLS_GEN2_SIGNED_URI}?action=append&position=0",
            data=ANY,
            headers=filtered_azure_headers,
            timeout=None,
        )
        request_mock.assert_any_call(
            "patch",
            f"{MOCK_ADLS_GEN2_SIGNED_URI}?action=append&position=5",
            data=ANY,
            headers=filtered_azure_headers,
            timeout=None,
        )
        request_mock.assert_any_call(
            "patch",
            f"{MOCK_ADLS_GEN2_SIGNED_URI}?action=append&position=10",
            data=ANY,
            headers=filtered_azure_headers,
            timeout=None,
        )
        request_mock.assert_called_with(
            "patch",
            f"{MOCK_ADLS_GEN2_SIGNED_URI}?action=flush&position=14",
            headers=filtered_azure_headers,
            timeout=None,
        )


def test_log_artifact_adls_gen2_flush_error(databricks_artifact_repo, test_file):
    mock_successful_response = Response()
    mock_successful_response.status_code = 200
    mock_successful_response.close = lambda: None
    mock_error_response = MlflowException("MOCK ERROR")
    mock_credential_info = ArtifactCredentialInfo(
        signed_uri=MOCK_ADLS_GEN2_SIGNED_URI,
        type=ArtifactCredentialType.AZURE_ADLS_GEN2_SAS_URI,
    )
    with mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._get_write_credential_infos",
        return_value=[mock_credential_info],
    ) as write_credential_infos_mock, mock.patch(
        "requests.Session.request", side_effect=[mock_successful_response, mock_error_response]
    ) as request_mock:
        mock_credential_info = ArtifactCredentialInfo(
            signed_uri=MOCK_ADLS_GEN2_SIGNED_URI,
            type=ArtifactCredentialType.AZURE_ADLS_GEN2_SAS_URI,
        )
        with pytest.raises(MlflowException, match=r"MOCK ERROR"):
            databricks_artifact_repo.log_artifact(test_file.strpath)
        write_credential_infos_mock.assert_called_with(run_id=MOCK_RUN_ID, paths=ANY)
        assert request_mock.mock_calls == [
            mock.call(
                "put",
                f"{MOCK_ADLS_GEN2_SIGNED_URI}?resource=file",
                headers={},
                timeout=None,
            ),
            mock.call(
                "patch",
                f"{MOCK_ADLS_GEN2_SIGNED_URI}?action=append&position=0&flush=true",
                data=ANY,
                headers={},
                timeout=None,
            ),
        ]


@pytest.mark.parametrize(("artifact_path", "expected_location"), [(None, "test.txt")])
def test_log_artifact_aws(databricks_artifact_repo, test_file, artifact_path, expected_location):
    mock_response = Response()
    mock_response.status_code = 200
    mock_response.close = lambda: None
    mock_credential_info = ArtifactCredentialInfo(
        signed_uri=MOCK_AWS_SIGNED_URI, type=ArtifactCredentialType.AWS_PRESIGNED_URL
    )
    with mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._get_write_credential_infos",
        return_value=[mock_credential_info],
    ) as write_credential_infos_mock, mock.patch(
        "requests.Session.request", return_value=mock_response
    ) as request_mock:
        databricks_artifact_repo.log_artifact(test_file.strpath, artifact_path)
        write_credential_infos_mock.assert_called_with(
            run_id=MOCK_RUN_ID, paths=[expected_location]
        )
        request_mock.assert_called_with(
            "put", MOCK_AWS_SIGNED_URI, data=ANY, headers={}, timeout=None
        )


@pytest.mark.parametrize(("artifact_path", "expected_location"), [(None, "test.txt")])
def test_log_artifact_aws_with_headers(
    databricks_artifact_repo, test_file, artifact_path, expected_location
):
    expected_headers = {header.name: header.value for header in MOCK_HEADERS}
    mock_response = Response()
    mock_response.status_code = 200
    mock_response.close = lambda: None
    mock_credential_info = ArtifactCredentialInfo(
        signed_uri=MOCK_AWS_SIGNED_URI,
        type=ArtifactCredentialType.AWS_PRESIGNED_URL,
        headers=MOCK_HEADERS,
    )
    with mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._get_write_credential_infos",
        return_value=[mock_credential_info],
    ) as write_credential_infos_mock, mock.patch(
        "requests.Session.request", return_value=mock_response
    ) as request_mock:
        databricks_artifact_repo.log_artifact(test_file.strpath, artifact_path)
        write_credential_infos_mock.assert_called_with(
            run_id=MOCK_RUN_ID, paths=[expected_location]
        )
        request_mock.assert_called_with(
            "put", MOCK_AWS_SIGNED_URI, data=ANY, headers=expected_headers, timeout=None
        )


def test_log_artifact_aws_presigned_url_error(databricks_artifact_repo, test_file):
    mock_credential_info = ArtifactCredentialInfo(
        signed_uri=MOCK_AWS_SIGNED_URI, type=ArtifactCredentialType.AWS_PRESIGNED_URL
    )
    with mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._get_write_credential_infos",
        return_value=[mock_credential_info],
    ) as write_credential_infos_mock, mock.patch(
        "requests.Session.request", side_effect=MlflowException("MOCK ERROR")
    ):
        with pytest.raises(MlflowException, match="MOCK ERROR"):
            databricks_artifact_repo.log_artifact(test_file.strpath)
        write_credential_infos_mock.assert_called_with(run_id=MOCK_RUN_ID, paths=ANY)


@pytest.mark.parametrize(("artifact_path", "expected_location"), [(None, "test.txt")])
def test_log_artifact_gcp(databricks_artifact_repo, test_file, artifact_path, expected_location):
    mock_response = Response()
    mock_response.status_code = 200
    mock_response.close = lambda: None
    mock_credential_info = ArtifactCredentialInfo(
        signed_uri=MOCK_GCP_SIGNED_URL, type=ArtifactCredentialType.GCP_SIGNED_URL
    )
    with mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._get_write_credential_infos",
        return_value=[mock_credential_info],
    ) as write_credential_infos_mock, mock.patch(
        "requests.Session.request", return_value=mock_response
    ) as request_mock:
        databricks_artifact_repo.log_artifact(test_file.strpath, artifact_path)
        write_credential_infos_mock.assert_called_with(
            run_id=MOCK_RUN_ID, paths=[expected_location]
        )
        request_mock.assert_called_with(
            "put", MOCK_GCP_SIGNED_URL, data=ANY, headers={}, timeout=None
        )


@pytest.mark.parametrize(("artifact_path", "expected_location"), [(None, "test.txt")])
def test_log_artifact_gcp_with_headers(
    databricks_artifact_repo, test_file, artifact_path, expected_location
):
    expected_headers = {header.name: header.value for header in MOCK_HEADERS}
    mock_response = Response()
    mock_response.status_code = 200
    mock_response.close = lambda: None
    mock_credential_info = ArtifactCredentialInfo(
        signed_uri=MOCK_GCP_SIGNED_URL,
        type=ArtifactCredentialType.GCP_SIGNED_URL,
        headers=MOCK_HEADERS,
    )
    with mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._get_write_credential_infos",
        return_value=[mock_credential_info],
    ) as write_credential_infos_mock, mock.patch(
        "requests.Session.request", return_value=mock_response
    ) as request_mock:
        databricks_artifact_repo.log_artifact(test_file.strpath, artifact_path)
        write_credential_infos_mock.assert_called_with(
            run_id=MOCK_RUN_ID, paths=[expected_location]
        )
        request_mock.assert_called_with(
            "put", MOCK_GCP_SIGNED_URL, data=ANY, headers=expected_headers, timeout=None
        )


def test_log_artifact_gcp_presigned_url_error(databricks_artifact_repo, test_file):
    mock_credential_info = ArtifactCredentialInfo(
        signed_uri=MOCK_GCP_SIGNED_URL, type=ArtifactCredentialType.GCP_SIGNED_URL
    )
    with mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._get_write_credential_infos",
        return_value=[mock_credential_info],
    ) as write_credential_infos_mock, mock.patch(
        "requests.Session.request", side_effect=MlflowException("MOCK ERROR")
    ):
        with pytest.raises(MlflowException, match="MOCK ERROR"):
            databricks_artifact_repo.log_artifact(test_file.strpath)
        write_credential_infos_mock.assert_called_with(run_id=MOCK_RUN_ID, paths=ANY)


@pytest.mark.parametrize(
    ("artifact_path", "expected_location"),
    [
        (None, posixpath.join(MOCK_SUBDIR, "test.txt")),
        ("test_path", posixpath.join(MOCK_SUBDIR, "test_path/test.txt")),
    ],
)
def test_log_artifact_with_relative_path(test_file, artifact_path, expected_location):
    mock_credential_info = ArtifactCredentialInfo(
        signed_uri=MOCK_AZURE_SIGNED_URI, type=ArtifactCredentialType.AZURE_SAS_URI
    )
    with mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._get_run_artifact_root",
        return_value=MOCK_RUN_ROOT_URI,
    ), mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._get_write_credential_infos",
        return_value=[mock_credential_info],
    ) as write_credential_infos_mock, mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._upload_to_cloud", return_value=None
    ) as upload_mock:
        databricks_artifact_repo = get_artifact_repository(MOCK_SUBDIR_ROOT_URI)
        databricks_artifact_repo.log_artifact(test_file.strpath, artifact_path)
        write_credential_infos_mock.assert_called_with(
            run_id=MOCK_RUN_ID, paths=[expected_location]
        )
        upload_mock.assert_called_with(
            cloud_credential_info=mock_credential_info,
            src_file_path=test_file.strpath,
            dst_run_relative_artifact_path=expected_location,
        )


def test_list_artifacts(databricks_artifact_repo):
    list_artifact_file_proto_mock = [FileInfo(path="a.txt", is_dir=False, file_size=0)]
    list_artifacts_dir_proto_mock = [
        FileInfo(path="test/a.txt", is_dir=False, file_size=100),
        FileInfo(path="test/dir", is_dir=True, file_size=0),
    ]
    list_artifact_response_proto = ListArtifacts.Response(
        root_uri="", files=list_artifacts_dir_proto_mock, next_page_token=None
    )
    with mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._call_endpoint",
        return_value=list_artifact_response_proto,
    ):
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
    with mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._call_endpoint",
        return_value=list_artifact_response_proto,
    ):
        artifacts = databricks_artifact_repo.list_artifacts("a.txt")
        assert len(artifacts) == 0


def test_list_artifacts_with_relative_path():
    list_artifact_file_proto_mock = [
        FileInfo(path=posixpath.join(MOCK_SUBDIR, "a.txt"), is_dir=False, file_size=0)
    ]
    list_artifacts_dir_proto_mock = [
        FileInfo(path=posixpath.join(MOCK_SUBDIR, "test/a.txt"), is_dir=False, file_size=100),
        FileInfo(path=posixpath.join(MOCK_SUBDIR, "test/dir"), is_dir=True, file_size=0),
    ]
    with mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._get_run_artifact_root",
        return_value=MOCK_RUN_ROOT_URI,
    ), mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY_PACKAGE}.message_to_json", return_value=None
    ) as message_mock:
        list_artifact_response_proto = ListArtifacts.Response(
            root_uri="", files=list_artifacts_dir_proto_mock, next_page_token=None
        )
        with mock.patch(
            f"{DATABRICKS_ARTIFACT_REPOSITORY}._call_endpoint",
            return_value=list_artifact_response_proto,
        ):
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
        with mock.patch(
            f"{DATABRICKS_ARTIFACT_REPOSITORY}._call_endpoint",
            return_value=ListArtifacts.Response(
                root_uri="", files=list_artifact_file_proto_mock, next_page_token=None
            ),
        ):
            artifacts = databricks_artifact_repo.list_artifacts("a.txt")
            assert len(artifacts) == 0


def test_list_artifacts_handles_pagination(databricks_artifact_repo):
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
    list_artifact_paginated_response_protos = [
        ListArtifacts.Response(root_uri="", files=list_artifacts_proto_mock_1, next_page_token="2"),
        ListArtifacts.Response(root_uri="", files=list_artifacts_proto_mock_2, next_page_token="4"),
        ListArtifacts.Response(root_uri="", files=list_artifacts_proto_mock_3, next_page_token="6"),
        ListArtifacts.Response(root_uri="", files=list_artifacts_proto_mock_4, next_page_token="8"),
    ]
    with mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY_PACKAGE}.message_to_json", return_value=None
    ) as message_mock, mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._call_endpoint",
        side_effect=list_artifact_paginated_response_protos,
    ):
        artifacts = databricks_artifact_repo.list_artifacts()
        assert {file.path for file in artifacts} == {"a.txt", "b", "c.txt", "d", "e.txt", "f"}
        calls = [
            mock.call(ListArtifacts(run_id=MOCK_RUN_ID, path="")),
            mock.call(ListArtifacts(run_id=MOCK_RUN_ID, path="", page_token="2")),
            mock.call(ListArtifacts(run_id=MOCK_RUN_ID, path="", page_token="4")),
            mock.call(ListArtifacts(run_id=MOCK_RUN_ID, path="", page_token="6")),
        ]
        message_mock.assert_has_calls(calls)


def test_get_read_credential_infos_handles_pagination(databricks_artifact_repo):
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
    get_credentials_for_read_responses = [
        GetCredentialsForRead.Response(
            credential_infos=credential_infos_mock_1, next_page_token="2"
        ),
        GetCredentialsForRead.Response(
            credential_infos=credential_infos_mock_2, next_page_token="3"
        ),
        GetCredentialsForRead.Response(credential_infos=credential_infos_mock_3),
    ]
    with mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY_PACKAGE}.message_to_json"
    ) as message_mock, mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._call_endpoint",
        side_effect=get_credentials_for_read_responses,
    ) as call_endpoint_mock:
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


def test_get_read_credential_infos_respects_max_request_size(databricks_artifact_repo):
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
        f"{DATABRICKS_ARTIFACT_REPOSITORY_PACKAGE}.message_to_json"
    ) as message_mock, mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._call_endpoint",
        side_effect=[
            GetCredentialsForRead.Response(credential_infos=credential_infos_mock_1),
            GetCredentialsForRead.Response(credential_infos=credential_infos_mock_2),
            GetCredentialsForRead.Response(credential_infos=credential_infos_mock_3),
        ],
    ) as call_endpoint_mock:
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


def test_get_write_credential_infos_handles_pagination(databricks_artifact_repo):
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
    get_credentials_for_write_responses = [
        GetCredentialsForWrite.Response(
            credential_infos=credential_infos_mock_1, next_page_token="2"
        ),
        GetCredentialsForWrite.Response(
            credential_infos=credential_infos_mock_2, next_page_token="3"
        ),
        GetCredentialsForWrite.Response(credential_infos=credential_infos_mock_3),
    ]
    with mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY_PACKAGE}.message_to_json"
    ) as message_mock, mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._call_endpoint",
        side_effect=get_credentials_for_write_responses,
    ) as call_endpoint_mock:
        write_credential_infos = databricks_artifact_repo._get_write_credential_infos(
            MOCK_RUN_ID,
            ["testpath"],
        )
        assert write_credential_infos == credential_infos_mock_1 + credential_infos_mock_2
        message_mock.assert_has_calls(
            [
                mock.call(GetCredentialsForWrite(run_id=MOCK_RUN_ID, path=["testpath"])),
                mock.call(
                    GetCredentialsForWrite(run_id=MOCK_RUN_ID, path=["testpath"], page_token="2")
                ),
                mock.call(
                    GetCredentialsForWrite(run_id=MOCK_RUN_ID, path=["testpath"], page_token="3")
                ),
            ]
        )
        assert call_endpoint_mock.call_count == 3


def test_get_write_credential_infos_respects_max_request_size(databricks_artifact_repo):
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
        f"{DATABRICKS_ARTIFACT_REPOSITORY_PACKAGE}.message_to_json"
    ) as message_mock, mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._call_endpoint",
        side_effect=[
            GetCredentialsForWrite.Response(credential_infos=credential_infos_mock_1),
            GetCredentialsForWrite.Response(credential_infos=credential_infos_mock_2),
            GetCredentialsForWrite.Response(credential_infos=credential_infos_mock_3),
        ],
    ) as call_endpoint_mock:
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
    ("remote_file_path", "local_path", "cloud_credential_type"),
    [
        ("test_file.txt", "", ArtifactCredentialType.AZURE_SAS_URI),
        ("test_file.txt", None, ArtifactCredentialType.AZURE_SAS_URI),
        ("output/test_file", None, ArtifactCredentialType.AZURE_SAS_URI),
        ("test_file.txt", "", ArtifactCredentialType.AWS_PRESIGNED_URL),
        ("test_file.txt", "", ArtifactCredentialType.GCP_SIGNED_URL),
    ],
)
def test_databricks_download_file(
    databricks_artifact_repo, remote_file_path, local_path, cloud_credential_type
):
    mock_credential_info = ArtifactCredentialInfo(
        signed_uri=MOCK_AZURE_SIGNED_URI, type=cloud_credential_type
    )
    with mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._get_read_credential_infos",
        return_value=[mock_credential_info],
    ) as read_credential_infos_mock, mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}.list_artifacts", return_value=[]
    ), mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._download_from_cloud", return_value=None
    ) as download_mock:
        databricks_artifact_repo.download_artifacts(remote_file_path, local_path)
        read_credential_infos_mock.assert_called_with(run_id=MOCK_RUN_ID, paths=[remote_file_path])
        download_mock.assert_called_with(
            cloud_credential_info=mock_credential_info,
            dst_local_file_path=ANY,
        )


@pytest.mark.parametrize(
    ("remote_file_path", "local_path"), [("test_file.txt", ""), ("test_file.txt", None)]
)
def test_databricks_download_file_with_relative_path(remote_file_path, local_path):
    mock_credential_info = ArtifactCredentialInfo(
        signed_uri=MOCK_AZURE_SIGNED_URI, type=ArtifactCredentialType.AZURE_SAS_URI
    )
    with mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._get_run_artifact_root",
        return_value=MOCK_RUN_ROOT_URI,
    ), mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._get_read_credential_infos",
        return_value=[mock_credential_info],
    ) as read_credential_infos_mock, mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}.list_artifacts", return_value=[]
    ), mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._download_from_cloud", return_value=None
    ) as download_mock:
        databricks_artifact_repo = get_artifact_repository(MOCK_SUBDIR_ROOT_URI)
        databricks_artifact_repo.download_artifacts(remote_file_path, local_path)
        read_credential_infos_mock.assert_called_with(
            run_id=MOCK_RUN_ID, paths=[posixpath.join(MOCK_SUBDIR, remote_file_path)]
        )
        download_mock.assert_called_with(
            cloud_credential_info=mock_credential_info,
            dst_local_file_path=ANY,
        )


def test_databricks_download_file_get_request_fail(databricks_artifact_repo, test_file):
    mock_credential_info = ArtifactCredentialInfo(
        signed_uri=MOCK_AZURE_SIGNED_URI, type=ArtifactCredentialType.AZURE_SAS_URI
    )
    with mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._get_read_credential_infos",
        return_value=[mock_credential_info],
    ) as read_credential_infos_mock, mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}.list_artifacts", return_value=[]
    ), mock.patch(
        "requests.Session.request", side_effect=MlflowException("MOCK ERROR")
    ):
        with pytest.raises(MlflowException, match=r"MOCK ERROR"):
            databricks_artifact_repo.download_artifacts(test_file.strpath)
        read_credential_infos_mock.assert_called_with(run_id=MOCK_RUN_ID, paths=[test_file.strpath])


def test_download_artifacts_awaits_download_completion(databricks_artifact_repo, tmpdir):
    """
    Verifies that all asynchronous artifact downloads are joined before `download_artifacts()`
    returns a result to the caller
    """

    def mock_download_from_cloud(
        cloud_credential_info,  # pylint: disable=unused-argument
        dst_local_file_path,
    ):
        # Sleep in order to simulate a longer-running asynchronous download
        time.sleep(2)
        with open(dst_local_file_path, "w") as f:
            f.write("content")

    with mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._get_read_credential_infos",
        return_value=[
            ArtifactCredentialInfo(
                signed_uri=MOCK_AZURE_SIGNED_URI, type=ArtifactCredentialType.AZURE_SAS_URI
            ),
        ],
    ), mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}.list_artifacts",
        return_value=[
            FileInfo(path="file_1.txt", is_dir=False, file_size=100),
            FileInfo(path="file_2.txt", is_dir=False, file_size=0),
        ],
    ), mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._download_from_cloud",
        side_effect=mock_download_from_cloud,
    ):
        databricks_artifact_repo.download_artifacts("test_path", str(tmpdir))
        expected_file1_path = os.path.join(str(tmpdir), "file_1.txt")
        expected_file2_path = os.path.join(str(tmpdir), "file_2.txt")
        for path in [expected_file1_path, expected_file2_path]:
            assert os.path.exists(path)
            with open(path) as f:
                assert f.read() == "content"


def test_artifact_logging(databricks_artifact_repo, tmpdir):
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

    def mock_upload_to_cloud(
        cloud_credential_info,  # pylint: disable=unused-argument
        src_file_path,
        dst_run_relative_artifact_path,
    ):
        # Sleep in order to simulate a longer-running asynchronous upload
        time.sleep(2)
        dst_run_relative_artifact_path = os.path.join(dst_dir, dst_run_relative_artifact_path)
        os.makedirs(os.path.dirname(dst_run_relative_artifact_path), exist_ok=True)
        shutil.copyfile(src=src_file_path, dst=dst_run_relative_artifact_path)

    with mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._get_write_credential_infos",
        return_value=[
            ArtifactCredentialInfo(
                signed_uri=MOCK_AZURE_SIGNED_URI, type=ArtifactCredentialType.AZURE_SAS_URI
            ),
            ArtifactCredentialInfo(
                signed_uri=MOCK_AZURE_SIGNED_URI, type=ArtifactCredentialType.AZURE_SAS_URI
            ),
        ],
    ), mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._upload_to_cloud", side_effect=mock_upload_to_cloud
    ):
        databricks_artifact_repo.log_artifacts(src_dir, "dir_artifact")

        expected_dst_dir_file1_path = os.path.join(dst_dir, "dir_artifact", "file_1.txt")
        expected_dst_dir_file2_path = os.path.join(dst_dir, "dir_artifact", "file_2.txt")
        assert os.path.exists(expected_dst_dir_file1_path)
        assert os.path.exists(expected_dst_dir_file2_path)
        with open(expected_dst_dir_file1_path) as f:
            assert f.read() == "file1"
        with open(expected_dst_dir_file2_path) as f:
            assert f.read() == "file2"

        databricks_artifact_repo.log_artifact(src_file1_path)

        expected_dst_file_path = os.path.join(dst_dir, "file_1.txt")
        assert os.path.exists(expected_dst_file_path)
        with open(expected_dst_file_path) as f:
            assert f.read() == "file1"


def test_download_artifacts_provides_failure_info(databricks_artifact_repo):
    with mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._get_read_credential_infos",
        return_value=[
            ArtifactCredentialInfo(
                signed_uri=MOCK_AZURE_SIGNED_URI, type=ArtifactCredentialType.AZURE_SAS_URI
            ),
        ],
    ), mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}.list_artifacts",
        return_value=[
            FileInfo(path="file_1.txt", is_dir=False, file_size=100),
            FileInfo(path="file_2.txt", is_dir=False, file_size=0),
        ],
    ), mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._download_from_cloud",
        side_effect=[
            MlflowException("MOCK ERROR 1"),
            MlflowException("MOCK ERROR 2"),
        ],
    ):
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


def test_log_artifacts_provides_failure_info(databricks_artifact_repo, tmpdir):
    src_file1_path = os.path.join(str(tmpdir), "file_1.txt")
    with open(src_file1_path, "w") as f:
        f.write("file1")
    src_file2_path = os.path.join(str(tmpdir), "file_2.txt")
    with open(src_file2_path, "w") as f:
        f.write("file2")

    with mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._get_write_credential_infos",
        return_value=[
            ArtifactCredentialInfo(
                signed_uri=MOCK_AZURE_SIGNED_URI, type=ArtifactCredentialType.AZURE_SAS_URI
            ),
            ArtifactCredentialInfo(
                signed_uri=MOCK_AZURE_SIGNED_URI, type=ArtifactCredentialType.AZURE_SAS_URI
            ),
        ],
    ), mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._upload_to_cloud",
        side_effect=[
            MlflowException("MOCK ERROR 1"),
            MlflowException("MOCK ERROR 2"),
        ],
    ):
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


@pytest.fixture
def mock_chunk_size():
    # Use a smaller chunk size for faster comparison
    chunk_size = 10
    with mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY_PACKAGE}._MULTIPART_UPLOAD_CHUNK_SIZE", chunk_size
    ):
        yield chunk_size


@pytest.fixture
def large_file(tmp_path, mock_chunk_size):
    path = tmp_path.joinpath("large_file")
    with path.open("a") as f:
        f.write("a" * mock_chunk_size)
        f.write("b" * mock_chunk_size)
    yield path


def extract_part_number(url):
    return int(re.search(r"partNumber=(\d+)", url).group(1))


def mock_request(method, url, *args, **kwargs):
    resp = Response()
    resp.status_code = 200
    resp.close = lambda: None
    if method.lower() == "delete":
        # Abort-multipart-upload request
        return resp
    elif method.lower() == "put":
        # Upload-part request
        part_number = extract_part_number(url)
        resp.headers = {"ETag": f"etag-{part_number}"}
        return resp
    else:
        raise Exception("Unreachable")


def test_multipart_upload(databricks_artifact_repo, large_file, mock_chunk_size):
    mock_credential_info = ArtifactCredentialInfo(
        signed_uri=MOCK_AWS_SIGNED_URI, type=ArtifactCredentialType.AWS_PRESIGNED_URL
    )
    mock_upload_id = "upload_id"
    create_mpu_response = CreateMultipartUpload.Response(
        upload_id=mock_upload_id,
        upload_credential_infos=[
            ArtifactCredentialInfo(
                signed_uri=f"{MOCK_AWS_SIGNED_URI}partNumber={i + 1}",
                type=ArtifactCredentialType.AWS_PRESIGNED_URL,
                headers=[ArtifactCredentialInfo.HttpHeader(name="header", value=f"part-{i + 1}")],
            )
            for i in range(2)
        ],
        abort_credential_info=ArtifactCredentialInfo(
            signed_uri=f"{MOCK_AWS_SIGNED_URI}uploadId=abort",
            type=ArtifactCredentialType.AWS_PRESIGNED_URL,
            headers=[ArtifactCredentialInfo.HttpHeader(name="header", value="abort")],
        ),
    )
    complete_mpu_response = CompleteMultipartUpload.Response()
    with mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._get_write_credential_infos",
        return_value=[mock_credential_info],
    ), mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._call_endpoint",
        side_effect=[create_mpu_response, complete_mpu_response],
    ) as call_endpoint_mock, mock.patch(
        "requests.Session.request", side_effect=mock_request
    ) as http_request_mock:
        databricks_artifact_repo.log_artifact(large_file)
        with large_file.open("rb") as f:
            expected_calls = [
                mock.call(
                    "put",
                    f"{MOCK_AWS_SIGNED_URI}partNumber={i + 1}",
                    data=f.read(mock_chunk_size),
                    headers={"header": f"part-{i + 1}"},
                    timeout=None,
                )
                for i in range(2)
            ]
        # The upload-part requests are sent in parallel, so the order of the calls is not
        # deterministic
        assert sorted(http_request_mock.call_args_list, key=lambda c: c.args[1]) == expected_calls
        complete_request_body = json.loads(call_endpoint_mock.call_args_list[-1].args[-1])
        assert complete_request_body["upload_id"] == mock_upload_id
        assert complete_request_body["part_etags"] == [
            {"part_number": 1, "etag": "etag-1"},
            {"part_number": 2, "etag": "etag-2"},
        ]


# The first request will fail with a 403, and the second will succeed
STATUS_CODE_GENERATOR = (s for s in (403, 200))


def mock_request_retry(method, url, *args, **kwargs):
    resp = Response()
    resp.status_code = 200
    resp.close = lambda: None
    if method.lower() == "delete":
        # Abort-multipart-upload request
        return resp
    elif method.lower() == "put":
        # Upload-part request
        part_number = extract_part_number(url)
        resp.headers = {"ETag": f"etag-{part_number}"}
        # To ensure the upload-part retry logic works correctly,
        # make the first attempt of the second part upload fail by responding with a 403,
        # then make the second attempt succeed by responding with a 200
        if part_number == 2:
            status_code = next(STATUS_CODE_GENERATOR)
            resp.status_code = status_code
        return resp
    else:
        raise Exception("Unreachable")


def test_multipart_upload_retry_part_upload(databricks_artifact_repo, large_file, mock_chunk_size):
    mock_credential_info = ArtifactCredentialInfo(
        signed_uri=MOCK_AWS_SIGNED_URI, type=ArtifactCredentialType.AWS_PRESIGNED_URL
    )
    mock_upload_id = "upload_id"
    create_mpu_response = CreateMultipartUpload.Response(
        upload_id=mock_upload_id,
        upload_credential_infos=[
            ArtifactCredentialInfo(
                signed_uri=f"{MOCK_AWS_SIGNED_URI}partNumber={i + 1}",
                type=ArtifactCredentialType.AWS_PRESIGNED_URL,
                headers=[ArtifactCredentialInfo.HttpHeader(name="header", value=f"part-{i + 1}")],
            )
            for i in range(2)
        ],
        abort_credential_info=ArtifactCredentialInfo(
            signed_uri=f"{MOCK_AWS_SIGNED_URI}uploadId=abort",
            type=ArtifactCredentialType.AWS_PRESIGNED_URL,
            headers=[ArtifactCredentialInfo.HttpHeader(name="header", value="abort")],
        ),
    )
    part_upload_url_response = GetPresignedUploadPartUrl.Response(
        upload_credential_info=ArtifactCredentialInfo(
            signed_uri=f"{MOCK_AWS_SIGNED_URI}partNumber=2",
            type=ArtifactCredentialType.AWS_PRESIGNED_URL,
            headers=[ArtifactCredentialInfo.HttpHeader(name="header", value="part-2")],
        ),
    )
    complete_mpu_response = CompleteMultipartUpload.Response()

    with mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._get_write_credential_infos",
        return_value=[mock_credential_info],
    ), mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._call_endpoint",
        side_effect=[create_mpu_response, part_upload_url_response, complete_mpu_response],
    ) as call_endpoint_mock, mock.patch(
        "requests.Session.request", side_effect=mock_request_retry
    ) as http_request_mock:
        databricks_artifact_repo.log_artifact(large_file)

        with large_file.open("rb") as f:
            expected_calls = [
                mock.call(
                    "put",
                    f"{MOCK_AWS_SIGNED_URI}partNumber={i + 1}",
                    data=f.read(mock_chunk_size),
                    headers={"header": f"part-{i + 1}"},
                    timeout=None,
                )
                for i in range(2)
            ]
        expected_calls += expected_calls[-1:]  # Append the second part upload call
        # The upload-part requests are sent in parallel, so the order of the calls is not
        # deterministic
        assert sorted(http_request_mock.call_args_list, key=lambda c: c.args[1]) == expected_calls
        complete_request_body = json.loads(call_endpoint_mock.call_args_list[-1].args[-1])
        assert complete_request_body["upload_id"] == mock_upload_id
        assert complete_request_body["part_etags"] == [
            {"part_number": 1, "etag": "etag-1"},
            {"part_number": 2, "etag": "etag-2"},
        ]


def test_multipart_upload_abort(databricks_artifact_repo, large_file, mock_chunk_size):
    mock_credential_info = ArtifactCredentialInfo(
        signed_uri=MOCK_AWS_SIGNED_URI,
        type=ArtifactCredentialType.AWS_PRESIGNED_URL,
    )
    mock_upload_id = "upload_id"
    create_mpu_response = CreateMultipartUpload.Response(
        upload_id=mock_upload_id,
        upload_credential_infos=[
            ArtifactCredentialInfo(
                signed_uri=f"{MOCK_AWS_SIGNED_URI}partNumber={i + 1}",
                type=ArtifactCredentialType.AWS_PRESIGNED_URL,
                headers=[ArtifactCredentialInfo.HttpHeader(name="header", value=f"part-{i + 1}")],
            )
            for i in range(2)
        ],
        abort_credential_info=ArtifactCredentialInfo(
            signed_uri=f"{MOCK_AWS_SIGNED_URI}uploadId=abort",
            type=ArtifactCredentialType.AWS_PRESIGNED_URL,
            headers=[ArtifactCredentialInfo.HttpHeader(name="header", value="abort")],
        ),
    )
    with mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._get_write_credential_infos",
        return_value=[mock_credential_info],
    ), mock.patch(
        f"{DATABRICKS_ARTIFACT_REPOSITORY}._call_endpoint",
        side_effect=[create_mpu_response, Exception("Failed to complete multipart upload")],
    ) as call_endpoint_mock, mock.patch(
        "requests.Session.request", side_effect=mock_request
    ) as http_request_mock:
        with pytest.raises(Exception, match="Failed to complete multipart upload"):
            databricks_artifact_repo.log_artifact(large_file)

        (*part_upload_calls, abort_call) = http_request_mock.call_args_list
        with large_file.open("rb") as f:
            expected_calls = [
                mock.call(
                    "put",
                    f"{MOCK_AWS_SIGNED_URI}partNumber={i + 1}",
                    data=f.read(mock_chunk_size),
                    headers={"header": f"part-{i + 1}"},
                    timeout=None,
                )
                for i in range(2)
            ]
        assert part_upload_calls == expected_calls
        # The upload-part requests are sent in parallel, so the order of the calls is not
        # deterministic
        assert sorted(part_upload_calls, key=lambda c: c.args[1]) == expected_calls
        complete_request_body = json.loads(call_endpoint_mock.call_args_list[-1].args[-1])
        assert complete_request_body["upload_id"] == mock_upload_id
        assert complete_request_body["part_etags"] == [
            {"part_number": 1, "etag": "etag-1"},
            {"part_number": 2, "etag": "etag-2"},
        ]
        assert abort_call == mock.call(
            "delete",
            f"{MOCK_AWS_SIGNED_URI}uploadId=abort",
            headers={"header": "abort"},
            timeout=None,
        )
