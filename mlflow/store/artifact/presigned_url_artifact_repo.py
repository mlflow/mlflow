import json
import os
import posixpath

from mlflow.entities import FileInfo
from mlflow.protos.databricks_artifacts_pb2 import ArtifactCredentialInfo
from mlflow.protos.databricks_filesystem_service_pb2 import (
    CreateDownloadUrlRequest,
    CreateDownloadUrlResponse,
    CreateUploadUrlRequest,
    CreateUploadUrlResponse,
    FilesystemService,
    ListDirectoryResponse,
)
from mlflow.store.artifact.artifact_repo import _retry_with_new_creds
from mlflow.store.artifact.cloud_artifact_repo import CloudArtifactRepository
from mlflow.utils.file_utils import download_file_using_http_uri
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.request_utils import augmented_raise_for_status, cloud_storage_http_request
from mlflow.utils.rest_utils import (
    _REST_API_PATH_PREFIX,
    call_endpoint,
    extract_api_info_for_service,
)

FILESYSTEM_METHOD_TO_INFO = extract_api_info_for_service(FilesystemService, _REST_API_PATH_PREFIX)
DIRECTORIES_ENDPOINT = "/api/2.0/fs/directories"


class PresignedUrlArtifactRepository(CloudArtifactRepository):
    """
    Stores and retrieves model artifacts using presigned URLs.
    """

    def __init__(self, db_creds, model_full_name, model_version):
        artifact_uri = posixpath.join(
            "/Models", model_full_name.replace(".", "/"), str(model_version)
        )
        super().__init__(artifact_uri)
        self.db_creds = db_creds

    def log_artifact(self, local_file, artifact_path=None):
        artifact_file_path = os.path.basename(local_file)
        if artifact_path:
            artifact_file_path = posixpath.join(artifact_path, artifact_file_path)
        cloud_credentials = self._get_write_credential_infos(
            remote_file_paths=[artifact_file_path]
        )[0]
        self._upload_to_cloud(
            cloud_credential_info=cloud_credentials,
            src_file_path=local_file,
            artifact_file_path=artifact_file_path,
        )

    def _get_write_credential_infos(self, remote_file_paths):
        endpoint, method = FILESYSTEM_METHOD_TO_INFO[CreateUploadUrlRequest]
        credential_infos = []
        for relative_path in remote_file_paths:
            fs_full_path = posixpath.join(self.artifact_uri, relative_path)
            req_body = message_to_json(CreateUploadUrlRequest(path=fs_full_path))
            response_proto = CreateUploadUrlResponse()
            resp = call_endpoint(
                host_creds=self.db_creds,
                endpoint=endpoint,
                method=method,
                json_body=req_body,
                response_proto=response_proto,
            )
            headers = [
                ArtifactCredentialInfo.HttpHeader(name=header.name, value=header.value)
                for header in resp.headers
            ]
            credential_infos.append(ArtifactCredentialInfo(signed_uri=resp.url, headers=headers))
        return credential_infos

    def _upload_to_cloud(self, cloud_credential_info, src_file_path, artifact_file_path=None):
        # artifact_file_path is unused in this implementation because the presigned URL
        # and local file path are sufficient for upload to cloud storage
        def try_func(creds):
            presigned_url = creds.signed_uri
            headers = {header.name: header.value for header in creds.headers}
            with open(src_file_path, "rb") as source_file:
                data = source_file.read()
                with cloud_storage_http_request(
                    "put", presigned_url, data=data, headers=headers
                ) as response:
                    augmented_raise_for_status(response)

        def creds_func():
            return self._get_write_credential_infos(remote_file_paths=[artifact_file_path])[0]

        _retry_with_new_creds(
            try_func=try_func, creds_func=creds_func, orig_creds=cloud_credential_info
        )

    def list_artifacts(self, path=""):
        infos = []
        page_token = ""
        while True:
            endpoint = posixpath.join(DIRECTORIES_ENDPOINT, self.artifact_uri.lstrip("/"), path)
            req_body = json.dumps({"page_token": page_token}) if page_token else None

            response_proto = ListDirectoryResponse()
            resp = call_endpoint(
                host_creds=self.db_creds,
                endpoint=endpoint,
                method="GET",
                json_body=req_body,
                response_proto=response_proto,
            )
            for dir_entry in resp.contents:
                rel_path = posixpath.relpath(dir_entry.path, self.artifact_uri)
                if dir_entry.is_directory:
                    infos.append(FileInfo(rel_path, True, None))
                else:
                    infos.append(FileInfo(rel_path, False, dir_entry.file_size))
            page_token = resp.next_page_token
            if not page_token:
                break
        return sorted(infos, key=lambda f: f.path)

    def _get_read_credential_infos(self, remote_file_paths):
        credential_infos = []
        for remote_file_path in remote_file_paths:
            resp = self._get_download_presigned_url_and_headers(remote_file_path)
            headers = [
                ArtifactCredentialInfo.HttpHeader(name=header.name, value=header.value)
                for header in resp.headers
            ]
            credential_infos.append(ArtifactCredentialInfo(signed_uri=resp.url, headers=headers))
        return credential_infos

    def _download_from_cloud(self, remote_file_path, local_path):
        def creds_func():
            return self._get_download_presigned_url_and_headers(remote_file_path)

        def try_func(creds):
            presigned_url = creds.url
            headers = {header.name: header.value for header in creds.headers}
            download_file_using_http_uri(
                http_uri=presigned_url, download_path=local_path, headers=headers
            )

        _retry_with_new_creds(try_func=try_func, creds_func=creds_func)

    def _get_download_presigned_url_and_headers(self, remote_file_path):
        remote_file_full_path = posixpath.join(self.artifact_uri, remote_file_path)
        endpoint, method = FILESYSTEM_METHOD_TO_INFO[CreateDownloadUrlRequest]
        req_body = message_to_json(CreateDownloadUrlRequest(path=remote_file_full_path))
        response_proto = CreateDownloadUrlResponse()
        return call_endpoint(
            host_creds=self.db_creds,
            endpoint=endpoint,
            method=method,
            json_body=req_body,
            response_proto=response_proto,
        )
