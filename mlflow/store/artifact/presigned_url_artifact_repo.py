import os
import posixpath
from collections import namedtuple
from typing import List

import mlflow
from mlflow.protos.databricks_artifacts_pb2 import ArtifactCredentialInfo
from mlflow.protos.databricks_fs_service_pb2 import FilesystemService, CreateDownloadUrlRequest, \
    CreateDownloadUrlResponse, ListRequest, ListResponse
from mlflow.protos.databricks_uc_registry_service_pb2 import UcModelRegistryService
from mlflow.store.artifact.cloud_artifact_repo import CloudArtifactRepository, StagedArtifactUpload
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.file_utils import download_file_using_http_uri
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import call_endpoint, extract_api_info_for_service, _REST_API_PATH_PREFIX

_METHOD_TO_INFO_UCMR = extract_api_info_for_service(UcModelRegistryService, _REST_API_PATH_PREFIX)
_METHOD_TO_INFO_FS = extract_api_info_for_service(FilesystemService, _REST_API_PATH_PREFIX)

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

    def upload_artifacts_iter(self, staged_uploads: List[StagedArtifactUpload]):
        raise NotImplementedError("this is for testing purposes only, not for production use.")
        # for staged_upload_chunk in chunk_list(staged_uploads, _ARTIFACT_UPLOAD_BATCH_SIZE):
        #     write_credential_infos = self._get_write_credential_infos(
        #         local_file_paths=[
        #             staged_upload.src_file_path for staged_upload in staged_upload_chunk
        #         ],
        #     )
        #
        #     inflight_uploads = {}
        #     for staged_upload, write_credential_info in zip(
        #             staged_upload_chunk, write_credential_infos
        #     ):
        #         upload_future = self.thread_pool.submit(
        #             self._upload_to_cloud,
        #             cloud_credential_info=write_credential_info,
        #             src_file_path=staged_upload.src_file_path,
        #             artifact_file_path=staged_upload.artifact_file_path,
        #         )
        #         inflight_uploads[staged_upload.src_file_path] = upload_future
        #
        #     yield from inflight_uploads.items()

    def _get_write_credential_infos(self, local_file_paths):
        raise NotImplementedError("this is for testing purposes only, not for production use.")
        # db_creds = get_databricks_host_creds()
        # endpoint, method = _METHOD_TO_INFO_FS[CreateUploadUrlRequest]
        # credential_infos = []
        # for local_file_path in local_file_paths:
        #     req_body = message_to_json(
        #         CreateUploadUrlRequest(
        #             path=local_file_path
        #         )
        #     )
        #     response_proto = CreateUploadUrlResponse()
        #     resp = call_endpoint(
        #         host_creds=db_creds,
        #         endpoint=endpoint,
        #         method=method,
        #         json_body=req_body,
        #         response_proto=response_proto,
        #     )
        #     elem = PresignedUrlAndHeaders(
        #         url=resp.url,
        #         headers={header.name: header.value for header in resp.headers},
        #     )
        #     credential_infos.append(elem)
        # return credential_infos

    def _upload_to_cloud(self, cloud_credential_info, src_file_path, artifact_file_path):
        raise NotImplementedError("this is for testing purposes only, not for production use.")
        # presigned_url = cloud_credential_info.url
        # headers = cloud_credential_info.headers
        # with open(src_file_path, "rb") as f:
        #     data = f.read()
        #     with cloud_storage_http_request("put", presigned_url, data=data, headers=headers) as response:
        #         augmented_raise_for_status(response)
        #         return response.headers["ETag"]

    def list_artifacts(self, path):
        fs_full_path = posixpath.join(self.artifact_uri, path)
        db_creds = get_databricks_host_creds(mlflow.get_registry_uri())
        endpoint, method = _METHOD_TO_INFO_FS[ListRequest]
        req_body = message_to_json(
            ListRequest(
                path=fs_full_path,
                recursive=True
            )
        )
        response_proto = ListResponse()
        resp = call_endpoint(
            host_creds=db_creds,
            endpoint=endpoint,
            method=method,
            json_body=req_body,
            response_proto=response_proto,
        ).files
        return resp

    def _create_download_destination(self, src_artifact_path, dst_local_dir_path=None):
        src_artifact_path = src_artifact_path.rstrip("/")  # Ensure correct dirname for trailing '/'
        src_artifact_path = src_artifact_path.lstrip("/")  # Ensure relative path for posixpath.join
        dirpath = posixpath.dirname(src_artifact_path)
        local_dir_path = os.path.join(dst_local_dir_path, dirpath)
        local_file_path = os.path.join(dst_local_dir_path, src_artifact_path)
        if not os.path.exists(local_dir_path):
            os.makedirs(local_dir_path, exist_ok=True)
        return local_file_path

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
        endpoint, method = _METHOD_TO_INFO_FS[CreateDownloadUrlRequest]
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
