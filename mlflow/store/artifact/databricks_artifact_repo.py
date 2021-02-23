import base64
import logging
import os
import posixpath
import requests
import uuid

from mlflow.azure.client import put_block, put_block_list
import mlflow.tracking
from mlflow.entities import FileInfo
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, INTERNAL_ERROR
from mlflow.protos.databricks_artifacts_pb2 import (
    DatabricksMlflowArtifactsService,
    GetCredentialsForWrite,
    GetCredentialsForRead,
    ArtifactCredentialType,
)
from mlflow.protos.service_pb2 import MlflowService, GetRun, ListArtifacts
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.file_utils import (
    download_file_using_http_uri,
    relative_path_to_artifact_path,
    yield_file_in_chunks,
)
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils import rest_utils
from mlflow.utils.rest_utils import (
    call_endpoint,
    extract_api_info_for_service,
    _REST_API_PATH_PREFIX,
)
from mlflow.utils.uri import (
    extract_and_normalize_path,
    get_databricks_profile_uri_from_artifact_uri,
    is_databricks_acled_artifacts_uri,
    is_valid_dbfs_uri,
    remove_databricks_profile_info_from_artifact_uri,
)

_logger = logging.getLogger(__name__)
_AZURE_MAX_BLOCK_CHUNK_SIZE = 100000000  # Max. size of each block allowed is 100 MB in stage_block
_DOWNLOAD_CHUNK_SIZE = 100000000
_SERVICE_AND_METHOD_TO_INFO = {
    service: extract_api_info_for_service(service, _REST_API_PATH_PREFIX)
    for service in [MlflowService, DatabricksMlflowArtifactsService]
}


class DatabricksArtifactRepository(ArtifactRepository):
    """
    Performs storage operations on artifacts in the access-controlled
    `dbfs:/databricks/mlflow-tracking` location.

    Signed access URIs for S3 / Azure Blob Storage are fetched from the MLflow service and used to
    read and write files from/to this location.

    The artifact_uri is expected to be of the form
    dbfs:/databricks/mlflow-tracking/<EXP_ID>/<RUN_ID>/
    """

    def __init__(self, artifact_uri):
        if not is_valid_dbfs_uri(artifact_uri):
            raise MlflowException(
                message="DBFS URI must be of the form dbfs:/<path> or "
                + "dbfs://profile@databricks/<path>",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if not is_databricks_acled_artifacts_uri(artifact_uri):
            raise MlflowException(
                message=(
                    "Artifact URI incorrect. Expected path prefix to be"
                    " databricks/mlflow-tracking/path/to/artifact/.."
                ),
                error_code=INVALID_PARAMETER_VALUE,
            )
        # The dbfs:/ path ultimately used for artifact operations should not contain the
        # Databricks profile info, so strip it before setting ``artifact_uri``.
        super().__init__(remove_databricks_profile_info_from_artifact_uri(artifact_uri))

        self.databricks_profile_uri = (
            get_databricks_profile_uri_from_artifact_uri(artifact_uri)
            or mlflow.tracking.get_tracking_uri()
        )
        self.run_id = self._extract_run_id(self.artifact_uri)
        # Fetch the artifact root for the MLflow Run associated with `artifact_uri` and compute
        # the path of `artifact_uri` relative to the MLflow Run's artifact root
        # (the `run_relative_artifact_repo_root_path`). All operations performed on this artifact
        # repository will be performed relative to this computed location
        artifact_repo_root_path = extract_and_normalize_path(artifact_uri)
        run_artifact_root_uri = self._get_run_artifact_root(self.run_id)
        run_artifact_root_path = extract_and_normalize_path(run_artifact_root_uri)
        run_relative_root_path = posixpath.relpath(
            path=artifact_repo_root_path, start=run_artifact_root_path
        )
        # If the paths are equal, then use empty string over "./" for ListArtifact compatibility.
        self.run_relative_artifact_repo_root_path = (
            "" if run_artifact_root_path == artifact_repo_root_path else run_relative_root_path
        )

    @staticmethod
    def _extract_run_id(artifact_uri):
        """
        The artifact_uri is expected to be
        dbfs:/databricks/mlflow-tracking/<EXP_ID>/<RUN_ID>/artifacts/<path>
        Once the path from the input uri is extracted and normalized, it is
        expected to be of the form
        databricks/mlflow-tracking/<EXP_ID>/<RUN_ID>/artifacts/<path>

        Hence the run_id is the 4th element of the normalized path.

        :return: run_id extracted from the artifact_uri
        """
        artifact_path = extract_and_normalize_path(artifact_uri)
        return artifact_path.split("/")[3]

    def _call_endpoint(self, service, api, json_body):
        db_creds = get_databricks_host_creds(self.databricks_profile_uri)
        endpoint, method = _SERVICE_AND_METHOD_TO_INFO[service][api]
        response_proto = api.Response()
        return call_endpoint(db_creds, endpoint, method, json_body, response_proto)

    def _get_run_artifact_root(self, run_id):
        json_body = message_to_json(GetRun(run_id=run_id))
        run_response = self._call_endpoint(MlflowService, GetRun, json_body)
        return run_response.run.info.artifact_uri

    def _get_write_credentials(self, run_id, path=None):
        json_body = message_to_json(GetCredentialsForWrite(run_id=run_id, path=path))
        return self._call_endpoint(
            DatabricksMlflowArtifactsService, GetCredentialsForWrite, json_body
        )

    def _get_read_credentials(self, run_id, path=None):
        json_body = message_to_json(GetCredentialsForRead(run_id=run_id, path=path))
        return self._call_endpoint(
            DatabricksMlflowArtifactsService, GetCredentialsForRead, json_body
        )

    def _extract_headers_from_credentials(self, headers):
        return {header.name: header.value for header in headers}

    def _azure_upload_file(self, credentials, local_file, artifact_path):
        """
        Uploads a file to a given Azure storage location.
        The function uses a file chunking generator with 100 MB being the size limit for each chunk.
        This limit is imposed by the stage_block API in azure-storage-blob.
        In the case the file size is large and the upload takes longer than the validity of the
        given credentials, a new set of credentials are generated and the operation continues. This
        is the reason for the first nested try-except block
        Finally, since the prevailing credentials could expire in the time between the last
        stage_block and the commit, a second try-except block refreshes credentials if needed.
        """
        try:
            headers = self._extract_headers_from_credentials(credentials.headers)
            uploading_block_list = list()
            for chunk in yield_file_in_chunks(local_file, _AZURE_MAX_BLOCK_CHUNK_SIZE):
                # Base64-encode a UUID, producing a UTF8-encoded bytestring. Then, decode
                # the bytestring for compliance with Azure Blob Storage API requests
                block_id = base64.b64encode(uuid.uuid4().hex.encode()).decode("utf-8")
                try:
                    put_block(credentials.signed_uri, block_id, chunk, headers=headers)
                except requests.HTTPError as e:
                    if e.response.status_code in [401, 403]:
                        _logger.info(
                            "Failed to authorize request, possibly due to credential expiration."
                            " Refreshing credentials and trying again..."
                        )
                        credentials = self._get_write_credentials(
                            self.run_id, artifact_path
                        ).credentials
                        put_block(credentials.signed_uri, block_id, chunk, headers=headers)
                    else:
                        raise e
                uploading_block_list.append(block_id)
            try:
                put_block_list(credentials.signed_uri, uploading_block_list, headers=headers)
            except requests.HTTPError as e:
                if e.response.status_code in [401, 403]:
                    _logger.info(
                        "Failed to authorize request, possibly due to credential expiration."
                        " Refreshing credentials and trying again..."
                    )
                    credentials = self._get_write_credentials(
                        self.run_id, artifact_path
                    ).credentials
                    put_block_list(credentials.signed_uri, uploading_block_list, headers=headers)
                else:
                    raise e
        except Exception as err:
            raise MlflowException(err)

    def _signed_url_upload_file(self, credentials, local_file):
        try:
            headers = self._extract_headers_from_credentials(credentials.headers)
            signed_write_uri = credentials.signed_uri
            # Putting an empty file in a request by reading file bytes gives 501 error.
            if os.stat(local_file).st_size == 0:
                with rest_utils.cloud_storage_http_request(
                    "put", signed_write_uri, "", headers=headers
                ) as response:
                    response.raise_for_status()
            else:
                with open(local_file, "rb") as file:
                    with rest_utils.cloud_storage_http_request(
                        "put", signed_write_uri, file, headers=headers
                    ) as response:
                        response.raise_for_status()
        except Exception as err:
            raise MlflowException(err)

    def _upload_to_cloud(self, cloud_credentials, local_file, artifact_path):
        if cloud_credentials.credentials.type == ArtifactCredentialType.AZURE_SAS_URI:
            self._azure_upload_file(cloud_credentials.credentials, local_file, artifact_path)
        elif cloud_credentials.credentials.type in [
            ArtifactCredentialType.AWS_PRESIGNED_URL,
            ArtifactCredentialType.GCP_SIGNED_URL,
        ]:
            self._signed_url_upload_file(cloud_credentials.credentials, local_file)
        else:
            raise MlflowException(
                message="Cloud provider not supported.", error_code=INTERNAL_ERROR
            )

    def _download_from_cloud(self, cloud_credential, local_file_path):
        """
        Downloads a file from the input `cloud_credential` and save it to `local_path`.

        Since the download mechanism for both cloud services, i.e., Azure and AWS is the same,
        a single download method is sufficient.

        The default working of `requests.get` is to download the entire response body immediately.
        However, this could be inefficient for large files. Hence the parameter `stream` is set to
        true. This only downloads the response headers at first and keeps the connection open,
        allowing content retrieval to be made via `iter_content`.
        In addition, since the connection is kept open, refreshing credentials is not required.
        """
        if cloud_credential.type not in [
            ArtifactCredentialType.AZURE_SAS_URI,
            ArtifactCredentialType.AWS_PRESIGNED_URL,
            ArtifactCredentialType.GCP_SIGNED_URL,
        ]:
            raise MlflowException(
                message="Cloud provider not supported.", error_code=INTERNAL_ERROR
            )
        try:
            signed_read_uri = cloud_credential.signed_uri
            download_file_using_http_uri(signed_read_uri, local_file_path, _DOWNLOAD_CHUNK_SIZE)
        except Exception as err:
            raise MlflowException(err)

    def log_artifact(self, local_file, artifact_path=None):
        basename = os.path.basename(local_file)
        artifact_path = artifact_path or ""
        artifact_path = posixpath.join(artifact_path, basename)
        if len(artifact_path) > 0:
            run_relative_artifact_path = posixpath.join(
                self.run_relative_artifact_repo_root_path, artifact_path
            )
        else:
            run_relative_artifact_path = self.run_relative_artifact_repo_root_path
        write_credentials = self._get_write_credentials(self.run_id, run_relative_artifact_path)
        self._upload_to_cloud(write_credentials, local_file, run_relative_artifact_path)

    def log_artifacts(self, local_dir, artifact_path=None):
        artifact_path = artifact_path or ""
        for (dirpath, _, filenames) in os.walk(local_dir):
            artifact_subdir = artifact_path
            if dirpath != local_dir:
                rel_path = os.path.relpath(dirpath, local_dir)
                rel_path = relative_path_to_artifact_path(rel_path)
                artifact_subdir = posixpath.join(artifact_path, rel_path)
            for name in filenames:
                file_path = os.path.join(dirpath, name)
                self.log_artifact(file_path, artifact_subdir)

    def list_artifacts(self, path=None):
        if path:
            run_relative_path = posixpath.join(self.run_relative_artifact_repo_root_path, path)
        else:
            run_relative_path = self.run_relative_artifact_repo_root_path
        infos = []
        page_token = None
        while True:
            if page_token:
                json_body = message_to_json(
                    ListArtifacts(run_id=self.run_id, path=run_relative_path, page_token=page_token)
                )
            else:
                json_body = message_to_json(
                    ListArtifacts(run_id=self.run_id, path=run_relative_path)
                )
            response = self._call_endpoint(MlflowService, ListArtifacts, json_body)
            artifact_list = response.files
            # If `path` is a file, ListArtifacts returns a single list element with the
            # same name as `path`. The list_artifacts API expects us to return an empty list in this
            # case, so we do so here.
            if (
                len(artifact_list) == 1
                and artifact_list[0].path == run_relative_path
                and not artifact_list[0].is_dir
            ):
                return []
            for output_file in artifact_list:
                file_rel_path = posixpath.relpath(
                    path=output_file.path, start=self.run_relative_artifact_repo_root_path
                )
                artifact_size = None if output_file.is_dir else output_file.file_size
                infos.append(FileInfo(file_rel_path, output_file.is_dir, artifact_size))
            if len(artifact_list) == 0 or not response.next_page_token:
                break
            page_token = response.next_page_token
        return infos

    def _download_file(self, remote_file_path, local_path):
        run_relative_remote_file_path = posixpath.join(
            self.run_relative_artifact_repo_root_path, remote_file_path
        )
        read_credentials = self._get_read_credentials(self.run_id, run_relative_remote_file_path)
        self._download_from_cloud(read_credentials.credentials, local_path)

    def delete_artifacts(self, artifact_path=None):
        raise MlflowException("Not implemented yet")
