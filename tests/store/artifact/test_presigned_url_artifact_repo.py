import json
from unittest import mock
from unittest.mock import ANY

from mlflow.protos.databricks_filesystem_service_pb2 import ListDirectoryResponse, DirectoryEntry, HttpHeader, \
    CreateDownloadUrlResponse, CreateDownloadUrlRequest, CreateUploadUrlResponse, CreateUploadUrlRequest
from mlflow.store.artifact.presigned_url_artifact_repo import PresignedUrlArtifactRepository, DIRECTORIES_ENDPOINT
from mlflow.utils.proto_json_utils import message_to_json

MODEL_NAME = "catalog.schema.model"
MODEL_VERSION = 1
MODEL_URI = "/Models/catalog/schema/model/1"
PRESIGNED_URL_ARTIFACT_REPOSITORY = "mlflow.store.artifact.presigned_url_artifact_repo"


def test_artifact_uri():
    artifact_repo = PresignedUrlArtifactRepository(MODEL_NAME, MODEL_VERSION)
    assert MODEL_URI == artifact_repo.artifact_uri


def mock_list_directory(*args, **kwargs):
    endpoint = kwargs["endpoint"]
    json_body = kwargs["json_body"]

    if (endpoint == f'{DIRECTORIES_ENDPOINT}{MODEL_URI}/dir'
            and json_body == json.dumps({"page_token": "some_token"})):
        return ListDirectoryResponse(
            contents=[
                DirectoryEntry(is_directory=False, path=f"{MODEL_URI}/dir/file2", file_size=2)
            ]
        )
    elif endpoint == f'{DIRECTORIES_ENDPOINT}{MODEL_URI}/dir':
        return ListDirectoryResponse(
            contents=[DirectoryEntry(is_directory=False, path=f"{MODEL_URI}/dir/file1", file_size=1)],
            next_page_token="some_token"
        )
    elif endpoint == f'{DIRECTORIES_ENDPOINT}{MODEL_URI}/':
        return ListDirectoryResponse(
            contents=[DirectoryEntry(is_directory=True, path=f"{MODEL_URI}/dir")],
        )
    else:
        raise ValueError(f"Unexpected endpoint: {endpoint}")


def test_list_artifact_pagination():
    artifact_repo = PresignedUrlArtifactRepository(MODEL_NAME, MODEL_VERSION)

    with mock.patch(f'{PRESIGNED_URL_ARTIFACT_REPOSITORY}.call_endpoint', side_effect=mock_list_directory) as mock_list:
        resp = artifact_repo.list_artifacts()
        assert len(resp) == 1
        assert resp[0].path == "dir"
        assert resp[0].is_dir is True
        assert resp[0].file_size is None

        resp = artifact_repo.list_artifacts("dir")
        assert len(resp) == 2
        assert {r.path for r in resp} == {"dir/file1", "dir/file2"}
        assert {r.is_dir for r in resp} == {False}
        assert {r.file_size for r in resp} == {1, 2}

        assert mock_list.call_count == 3


def _make_pesigned_url(remote_path):
    return f"presigned_url/{remote_path}"


def _make_headers(remote_path):
    return {f"path_header-{remote_path}": f"remote-path={remote_path}"}


def mock_create_download_url(*args, **kwargs):
    remote_path = json.loads(kwargs["json_body"])["path"]
    return CreateDownloadUrlResponse(url=_make_pesigned_url(remote_path),
                                     headers=[HttpHeader(name=header, value=val) for header, val in
                                              _make_headers(remote_path).items()])


def test_get_read_credentials():
    remote_file_paths = ["file", "dir/file1", "dir/file2"]
    artifact_repo = PresignedUrlArtifactRepository(MODEL_NAME, MODEL_VERSION)

    download_url_endpoint = "/api/2.0/fs/create-download-url"

    with mock.patch(f'{PRESIGNED_URL_ARTIFACT_REPOSITORY}.call_endpoint',
                    side_effect=mock_create_download_url) as mock_call_endpoint:
        creds = artifact_repo._get_read_credential_infos(remote_file_paths)
        assert mock_call_endpoint.call_count == 3

        for remote_file_path in remote_file_paths:
            mock_call_endpoint.assert_any_call(host_creds=ANY,
                                               endpoint=f"{download_url_endpoint}",
                                               method="POST",
                                               json_body=message_to_json(
                                                   CreateDownloadUrlRequest(path=f"{MODEL_URI}/{remote_file_path}")),
                                               response_proto=ANY)

        assert {_make_pesigned_url(f"{MODEL_URI}/{path}") for path in remote_file_paths} == {cred.signed_uri for cred in
                                                                                             creds}
        expected_headers = {}
        for path in remote_file_paths:
            expected_headers.update(_make_headers(f"{MODEL_URI}/{path}"))
        actual_headers = {}
        for cred in creds:
            actual_headers.update({header.name: header.value for header in cred.headers})
        assert expected_headers == actual_headers


def mock_create_upload_url(*args, **kwargs):
    remote_path = json.loads(kwargs["json_body"])["path"]
    return CreateUploadUrlResponse(url=_make_pesigned_url(remote_path),
                                   headers=[HttpHeader(name=header, value=val) for header, val in
                                            _make_headers(remote_path).items()])


def test_get_write_credentials():
    remote_file_paths = ["file", "dir/file1", "dir/file2"]
    artifact_repo = PresignedUrlArtifactRepository(MODEL_NAME, MODEL_VERSION)

    upload_url_endpoint = "/api/2.0/fs/create-upload-url"

    with mock.patch(f'{PRESIGNED_URL_ARTIFACT_REPOSITORY}.call_endpoint',
                    side_effect=mock_create_upload_url) as mock_call_endpoint:
        creds = artifact_repo._get_write_credential_infos(remote_file_paths)
        assert mock_call_endpoint.call_count == 3
        for remote_file_path in remote_file_paths:
            mock_call_endpoint.assert_any_call(host_creds=ANY,
                                               endpoint=f"{upload_url_endpoint}",
                                               method="POST",
                                               json_body=message_to_json(
                                                   CreateUploadUrlRequest(
                                                       path=f"{MODEL_URI}/{remote_file_path}")),
                                               response_proto=ANY)

        expected_headers = {}
        for path in remote_file_paths:
            expected_headers.update(_make_headers(f"{MODEL_URI}/{path}"))
        actual_headers = {}
        for cred in creds:
            actual_headers.update({header.name: header.value for header in cred.headers})
        assert expected_headers == actual_headers
