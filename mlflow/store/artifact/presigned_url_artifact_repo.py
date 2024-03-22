import os
import posixpath
from collections import namedtuple

import mlflow
from mlflow.protos.databricks_artifacts_pb2 import ArtifactCredentialInfo
from mlflow.protos.databricks_fs_service_pb2 import FilesystemService, CreateDownloadUrlRequest, \
    CreateDownloadUrlResponse, ListRequest, ListResponse, CreateUploadUrlRequest, CreateUploadUrlResponse, \
    ListDirectoryResponse
from mlflow.store.artifact.cloud_artifact_repo import CloudArtifactRepository
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.file_utils import download_file_using_http_uri
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.request_utils import cloud_storage_http_request, augmented_raise_for_status
from mlflow.utils.rest_utils import call_endpoint, extract_api_info_for_service, _REST_API_PATH_PREFIX

_METHOD_TO_INFO = extract_api_info_for_service(FilesystemService, _REST_API_PATH_PREFIX)

PresignedUrlAndHeaders = namedtuple(
    "PresignedUrlAndHeaders",
    [
        # presigned URL for the artifact
        "url",
        # headers to include in http request when using the presigned URL
        "headers",
    ],
)


class PresignedUrlArtifactRepository(CloudArtifactRepository):
    """
    Stores artifacts on S3 using presigned URLs.
    """

    def __init__(self, artifact_uri):
        super().__init__(artifact_uri)

    def log_artifact(self, local_file, artifact_path=None):
        raise NotImplementedError("this is for testing purposes only, not for production use.")
        pass

    def _get_write_credential_infos(self, remote_file_paths):
        db_creds = get_databricks_host_creds(mlflow.get_registry_uri())
        endpoint, method = _METHOD_TO_INFO[CreateUploadUrlRequest]
        credential_infos = []
        for relative_path in remote_file_paths:
            fs_full_path = posixpath.join(self.artifact_uri, relative_path)
            req_body = message_to_json(
                CreateUploadUrlRequest(
                    path=fs_full_path
                )
            )
            response_proto = CreateUploadUrlResponse()
            resp = call_endpoint(
                host_creds=db_creds,
                endpoint=endpoint,
                method=method,
                json_body=req_body,
                response_proto=response_proto,
            )
            elem = PresignedUrlAndHeaders(
                url=resp.url,
                headers={header.name: header.value for header in resp.headers},
            )
            credential_infos.append(elem)
        return credential_infos

    def _upload_to_cloud(self, cloud_credential_info, src_file_path, artifact_file_path):
        presigned_url = cloud_credential_info.url
        headers = cloud_credential_info.headers
        with open(src_file_path, "rb") as f:
            data = f.read()
            with cloud_storage_http_request("put", presigned_url, data=data, headers=headers) as response:
                augmented_raise_for_status(response)
                return response.headers["ETag"]

    def list_artifacts(self, path):
        fs_full_path = posixpath.join(self.artifact_uri, path)
        db_creds = get_databricks_host_creds(mlflow.get_registry_uri())
        method = "GET"

        paths = []
        directories = [fs_full_path]
        while directories:
            directory = directories.pop()
            page_token = ''
            while True:
                req_body = "" if not page_token else message_to_json(
                    {"page_token": page_token}
                )
                endpoint = posixpath.join("/api/2.0/fs/directories", directory.lstrip('/'))

                response_proto = ListDirectoryResponse()
                resp = call_endpoint(
                    host_creds=db_creds,
                    endpoint=endpoint,
                    method=method,
                    json_body=req_body,
                    response_proto=response_proto,
                )
                for dir_entry in resp.contents:
                    if dir_entry.is_directory:
                        directories.append(dir_entry.path)
                    else:
                        paths.append(dir_entry.path)
                page_token = resp.next_page_token
                if not page_token:
                    break
        return paths

    def _create_download_destination(self, src_artifact_path, dst_local_dir_path=None):
        src_artifact_path = src_artifact_path.lstrip("/")  # Ensure relative path for posixpath.join
        return super()._create_download_destination(src_artifact_path, dst_local_dir_path)

    def _get_read_credential_infos(self, remote_file_paths):
        return [
            ArtifactCredentialInfo(signed_url=self._get_presigned_url_and_headers(remote_file_path).url)
            for remote_file_path in remote_file_paths
        ]

    def _download_from_cloud(self, remote_file_path, local_path):
        # raise NotImplementedError("this is for testing purposes only, not for production use.")
        print(f"getting presigned url and headers for {remote_file_path}")
        presigned_url, headers = self._get_presigned_url_and_headers(remote_file_path)
        print(f"got presigned url and headers: {presigned_url}, {headers}")
        print(f"downloading file using http uri: {presigned_url} to {local_path}")
        download_file_using_http_uri(http_uri=presigned_url, download_path=local_path, headers=headers)
        print(f"downloaded file using http uri: {presigned_url} to {local_path}")

    def _get_presigned_url_and_headers(self, remote_file_path) -> PresignedUrlAndHeaders:
        # raise NotImplementedError("this is for testing purposes only, not for production use.")
        db_creds = get_databricks_host_creds(mlflow.get_registry_uri())
        endpoint, method = _METHOD_TO_INFO[CreateDownloadUrlRequest]
        req_body = message_to_json(
            CreateDownloadUrlRequest(path=remote_file_path)
        )
        response_proto = CreateDownloadUrlResponse()
        resp = call_endpoint(
            host_creds=db_creds,
            endpoint=endpoint,
            method=method,
            json_body=req_body,
            response_proto=response_proto,
        )
        return PresignedUrlAndHeaders(
            url=resp.url,
            headers={header.name: header.value for header in resp.headers},
        )
