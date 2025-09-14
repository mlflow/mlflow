import base64
import json
import logging
import os
import posixpath
import uuid
from typing import Any

import requests

import mlflow.tracking
from mlflow.azure.client import (
    patch_adls_file_upload,
    patch_adls_flush,
    put_adls_file_creation,
    put_block,
    put_block_list,
)
from mlflow.entities import FileInfo
from mlflow.environment_variables import (
    MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT,
    MLFLOW_MULTIPART_DOWNLOAD_CHUNK_SIZE,
    MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE,
    MLFLOW_MULTIPART_UPLOAD_MINIMUM_FILE_SIZE,
)
from mlflow.exceptions import (
    MlflowException,
    MlflowTraceDataCorrupted,
    MlflowTraceDataNotFound,
)
from mlflow.protos.databricks_artifacts_pb2 import (
    ArtifactCredentialType,
    CompleteMultipartUpload,
    CreateMultipartUpload,
    DatabricksMlflowArtifactsService,
    GetPresignedUploadPartUrl,
    PartEtag,
)
from mlflow.protos.databricks_pb2 import (
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
)
from mlflow.protos.service_pb2 import MlflowService
from mlflow.store.artifact.artifact_repo import write_local_temp_trace_data_file
from mlflow.store.artifact.cloud_artifact_repo import (
    CloudArtifactRepository,
    _complete_futures,
    _compute_num_chunks,
    _validate_chunk_size_aws,
)
from mlflow.store.artifact.databricks_artifact_repo_resources import (
    _CredentialType,
    _LoggedModel,
    _Resource,
    _Run,
    _Trace,
)
from mlflow.tracing.constant import TRACE_REQUEST_ID_PREFIX
from mlflow.utils import chunk_list
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.file_utils import (
    download_file_using_http_uri,
    read_chunk,
)
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.request_utils import cloud_storage_http_request
from mlflow.utils.rest_utils import (
    _REST_API_PATH_PREFIX,
    augmented_raise_for_status,
    call_endpoint,
    extract_api_info_for_service,
)
from mlflow.utils.uri import (
    extract_and_normalize_path,
    get_databricks_profile_uri_from_artifact_uri,
    is_databricks_acled_artifacts_uri,
    is_valid_dbfs_uri,
    remove_databricks_profile_info_from_artifact_uri,
)

_logger = logging.getLogger(__name__)
_MAX_CREDENTIALS_REQUEST_SIZE = 2000  # Max number of artifact paths in a single credentials request
_SERVICE_AND_METHOD_TO_INFO = {
    service: extract_api_info_for_service(service, _REST_API_PATH_PREFIX)
    for service in [MlflowService, DatabricksMlflowArtifactsService]
}


class DatabricksArtifactRepository(CloudArtifactRepository):
    """
    Performs storage operations on artifacts in the access-controlled
    `dbfs:/databricks/mlflow-tracking` location.

    Signed access URIs for S3 / Azure Blob Storage are fetched from the MLflow service and used to
    read and write files from/to this location.

    The artifact_uri is expected to be in one of the following forms:
    - dbfs:/databricks/mlflow-tracking/<EXP_ID>/<RUN_ID>/
    - databricks/mlflow-tracking/<EXP_ID>/logged_models/<MODEL_ID>/artifacts/<path>
    """

    def __init__(self, artifact_uri: str, tracking_uri: str | None = None) -> None:
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
        super().__init__(
            remove_databricks_profile_info_from_artifact_uri(artifact_uri), tracking_uri
        )

        self.databricks_profile_uri = (
            get_databricks_profile_uri_from_artifact_uri(artifact_uri)
            or mlflow.tracking.get_tracking_uri()
        )
        self.resource = self._extract_resource(self.artifact_uri)

    def _extract_resource(self, artifact_uri) -> _Resource:
        """
        The artifact_uri is expected to be in one of the following formats:
        - dbfs:/databricks/mlflow-tracking/<EXP_ID>/<RUN_ID>/artifacts/<path>
        - dbfs:/databricks/mlflow-tracking/<EXP_ID>/logged_models/<MODEL_ID>/artifacts/<path>
        - databricks/mlflow-tracking/<EXP_ID>/<RUN_ID>/artifacts/<path>
        - databricks/mlflow-tracking/<EXP_ID>/logged_models/<MODEL_ID>/artifacts/<path>

        Returns:
            A `_Resource` object representing the MLflow resource associated with the specified
            artifact URI.
        """
        artifact_path = extract_and_normalize_path(artifact_uri)
        parts = artifact_path.split("/")

        if parts[3] == "logged_models":
            return _LoggedModel(
                id_=parts[4], artifact_uri=artifact_uri, call_endpoint=self._call_endpoint
            )

        if parts[3].startswith(TRACE_REQUEST_ID_PREFIX):
            return _Trace(
                id_=parts[3], artifact_uri=artifact_uri, call_endpoint=self._call_endpoint
            )

        return _Run(id_=parts[3], artifact_uri=artifact_uri, call_endpoint=self._call_endpoint)

    @staticmethod
    def _extract_run_id(artifact_uri: str) -> str | None:
        """
        Extracts the run ID from the run artifact URI.
        """
        artifact_path = extract_and_normalize_path(artifact_uri)
        parts = artifact_path.split("/")
        if len(parts) < 4:
            return None

        if parts[3] == "logged_models" or parts[3].startswith(TRACE_REQUEST_ID_PREFIX):
            return None

        return parts[3]

    def _call_endpoint(
        self, service, api, json_body=None, path_params=None, retry_timeout_seconds=None
    ):
        """
        Calls the specified REST endpoint with the specified JSON body and path parameters.

        Args:
            service: The service to call.
            api: The API to call.
            json_body: The JSON body of the request.
            path_params: The path parameters to substitute into the endpoint URI.
            retry_timeout_seconds: The timeout in seconds for retrying failed requests.

        Returns:
            The response from the REST endpoint.
        """
        db_creds = get_databricks_host_creds(self.databricks_profile_uri)
        endpoint, method = _SERVICE_AND_METHOD_TO_INFO[service][api]
        if path_params:
            endpoint = endpoint.format(**path_params)
        response_proto = api.Response()

        return call_endpoint(
            host_creds=db_creds,
            endpoint=endpoint,
            method=method,
            json_body=json_body,
            response_proto=response_proto,
            retry_timeout_seconds=retry_timeout_seconds,
        )

    def _get_credential_infos(self, cred_type: _CredentialType, paths: list[str]):
        """
        Issue one or more requests for artifact credentials, providing read or write
        access to the specified resource relative artifact `paths` within the MLflow
        resource specified by `self.resource.id`. The type of access credentials, read or write,
        is specified by `request_message_class`.

        Args:
            cred_type: Specifies the type of access credentials, read or write.
            paths: The specified relative artifact paths within the MLflow resource.

        Returns:
            A list of `ArtifactCredentialInfo` objects providing read access to the specified
            relative artifact `paths` within the MLflow resource specified by `resource`.
        """
        credential_infos = []

        for paths_chunk in chunk_list(paths, _MAX_CREDENTIALS_REQUEST_SIZE):
            page_token = None
            while True:
                cred_infos, next_page_token = self.resource.get_credentials(
                    cred_type=cred_type,
                    paths=paths_chunk,
                    page_token=page_token,
                )
                credential_infos += cred_infos
                page_token = next_page_token
                if not page_token or len(cred_infos) == 0:
                    break

        return credential_infos

    def _get_write_credential_infos(self, remote_file_paths):
        """
        A list of `ArtifactCredentialInfo` objects providing write access to the specified
        relative artifact `paths` within the MLflow resource specified by `self.resource.id`.
        """
        relative_remote_paths = [
            posixpath.join(self.resource.relative_path, p or "") for p in remote_file_paths
        ]
        return self._get_credential_infos(_CredentialType.WRITE, relative_remote_paths)

    def download_trace_data(self) -> dict[str, Any]:
        [cred], _ = self.resource.get_credentials(cred_type=_CredentialType.READ)
        signed_uri = cred.signed_uri
        headers = self._extract_headers_from_credentials(cred.headers)
        with cloud_storage_http_request("get", signed_uri, headers=headers) as resp:
            try:
                augmented_raise_for_status(resp)
            except requests.HTTPError as e:
                if e.response.status_code == 404:
                    raise MlflowTraceDataNotFound(request_id=self.resource.id) from e
                raise

            try:
                return json.loads(resp.content)
            except json.JSONDecodeError as e:
                raise MlflowTraceDataCorrupted(request_id=self.resource.id) from e

    def upload_trace_data(self, trace_data: str) -> None:
        cred = self._get_upload_trace_data_cred_info()
        with write_local_temp_trace_data_file(trace_data) as temp_file:
            # Upload trace data synchronously to avoid ThreadPoolExecutor deadlock during Python
            # interpreter shutdown, which causes "cannot schedule new futures after shutdown" error.
            if cred.type == ArtifactCredentialType.AZURE_ADLS_GEN2_SAS_URI:
                self._azure_adls_gen2_upload_file(
                    credentials=cred,
                    local_file=temp_file,
                    artifact_file_path=None,
                    get_credentials=lambda artifact_paths: [
                        self._get_upload_trace_data_cred_info()
                    ],
                    is_sync=True,
                )
            elif cred.type == ArtifactCredentialType.AZURE_SAS_URI:
                self._azure_upload_file(
                    credentials=cred,
                    local_file=temp_file,
                    artifact_file_path=None,
                    get_credentials=lambda artifact_paths: [
                        self._get_upload_trace_data_cred_info()
                    ],
                    is_sync=True,
                )
            elif (
                cred.type == ArtifactCredentialType.AWS_PRESIGNED_URL
                or cred.type == ArtifactCredentialType.GCP_SIGNED_URL
            ):
                self._signed_url_upload_file(cred, temp_file)

    def _get_upload_trace_data_cred_info(self):
        """Returns the credential info for trace data upload."""
        [cred], _ = self.resource.get_credentials(
            cred_type=_CredentialType.WRITE,
            timeout=MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT.get(),
        )
        return cred

    def _get_read_credential_infos(self, remote_file_paths):
        """
        Returns:
            A list of `ArtifactCredentialInfo` objects providing read access to the specified
            relative artifact `paths` within the MLflow resource specified.
        """
        if type(remote_file_paths) == str:
            remote_file_paths = [remote_file_paths]
        if type(remote_file_paths) != list:
            raise MlflowException(
                f"Expected `paths` to be a list of strings. Got {type(remote_file_paths)}"
            )
        relative_remote_paths = [
            posixpath.join(self.resource.relative_path, p) for p in remote_file_paths
        ]
        return self._get_credential_infos(_CredentialType.READ, relative_remote_paths)

    def _extract_headers_from_credentials(self, headers):
        """
        Returns:
            A python dictionary of http headers converted from the protobuf credentials.
        """
        return {header.name: header.value for header in headers}

    def _azure_upload_chunk(
        self,
        credentials,
        headers,
        local_file,
        artifact_file_path,
        start_byte,
        size,
        get_credentials,
    ):
        """
        Uploads a chunk of a file to a given Azure storage location.

        Args:
            credentials: The credentials for the upload.
            headers: The headers for the upload.
            local_file: The local file to upload.
            artifact_file_path: The path to the artifact file.
            start_byte: The starting byte of the chunk.
            size: The size of the chunk.
            get_credentials: The function to call to get new credentials.
        """
        # Base64-encode a UUID, producing a UTF8-encoded bytestring. Then, decode
        # the bytestring for compliance with Azure Blob Storage API requests
        block_id = base64.b64encode(uuid.uuid4().hex.encode()).decode("utf-8")
        chunk = read_chunk(local_file, size, start_byte)
        try:
            put_block(credentials.signed_uri, block_id, chunk, headers=headers)
        except requests.HTTPError as e:
            if e.response.status_code in [401, 403]:
                _logger.info(
                    "Failed to authorize request, possibly due to credential expiration."
                    " Refreshing credentials and trying again..."
                )
                credential_info = get_credentials([artifact_file_path])[0]
                put_block(credential_info.signed_uri, block_id, chunk, headers=headers)
            else:
                raise e
        return block_id

    def _azure_upload_file(
        self, credentials, local_file, artifact_file_path, get_credentials, is_sync=False
    ):
        """
        Uploads a file to a given Azure storage location.
        The function uses a file chunking generator with 100 MB being the size limit for each chunk.
        This limit is imposed by the stage_block API in azure-storage-blob.
        In the case the file size is large and the upload takes longer than the validity of the
        given credentials, a new set of credentials are generated and the operation continues. This
        is the reason for the first nested try-except block
        Finally, since the prevailing credentials could expire in the time between the last
        stage_block and the commit, a second try-except block refreshes credentials if needed.

        Args:
            credentials: The credentials for the upload.
            local_file: The local file to upload.
            artifact_file_path: The path to the artifact file.
            get_credentials: The function to call to get new credentials.
            is_sync: If True, upload synchronously without threading.
        """
        try:
            headers = self._extract_headers_from_credentials(credentials.headers)
            num_chunks = _compute_num_chunks(local_file, MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE.get())

            def upload_chunks_func(start_byte):
                return self._azure_upload_chunk(
                    credentials=credentials,
                    headers=headers,
                    local_file=local_file,
                    artifact_file_path=artifact_file_path,
                    start_byte=start_byte,
                    size=MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE.get(),
                    get_credentials=get_credentials,
                )

            if is_sync:
                # Upload chunks synchronously without threading
                results = {}
                for index in range(num_chunks):
                    start_byte = index * MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE.get()
                    block_id = upload_chunks_func(start_byte)
                    results[index] = block_id
                uploading_block_list = [results[index] for index in sorted(results)]
            else:
                # Upload chunks asynchronously with threading
                futures = {}
                for index in range(num_chunks):
                    start_byte = index * MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE.get()
                    future = self.chunk_thread_pool.submit(
                        upload_chunks_func,
                        start_byte,
                    )
                    futures[future] = index

                results, errors = _complete_futures(futures, local_file)
                if errors:
                    raise MlflowException(
                        f"Failed to upload at least one part of {local_file}. Errors: {errors}"
                    )
                # Sort results by the chunk index
                uploading_block_list = [results[index] for index in sorted(results)]

            try:
                put_block_list(credentials.signed_uri, uploading_block_list, headers=headers)
            except requests.HTTPError as e:
                if e.response.status_code in [401, 403]:
                    _logger.info(
                        "Failed to authorize request, possibly due to credential expiration."
                        " Refreshing credentials and trying again..."
                    )
                    credential_info = get_credentials([artifact_file_path])[0]
                    put_block_list(
                        credential_info.signed_uri, uploading_block_list, headers=headers
                    )
                else:
                    raise e
        except Exception as err:
            raise MlflowException(err)

    def _retryable_adls_function(self, func, artifact_file_path, get_credentials, **kwargs):
        """
        Calls the passed function, retrying if the credentials have expired.

        Args:
            func: The function to call.
            artifact_file_path: The artifact file path.
            get_credentials: The function to call to get new credentials.
            **kwargs: The keyword arguments to pass to the function.
        """
        # Attempt to call the passed function.  Retry if the credentials have expired
        try:
            func(**kwargs)
        except requests.HTTPError as e:
            if e.response.status_code in [403]:
                _logger.info(
                    "Failed to authorize ADLS operation, possibly due "
                    "to credential expiration. Refreshing credentials and trying again..."
                )
                new_credentials = get_credentials([artifact_file_path])[0]
                kwargs["sas_url"] = new_credentials.signed_uri
                func(**kwargs)
            else:
                raise e

    def _azure_adls_gen2_upload_file(
        self, credentials, local_file, artifact_file_path, get_credentials, is_sync=False
    ):
        """
        Uploads a file to a given Azure storage location using the ADLS gen2 API.

        Args:
            credentials: The credentials for the upload.
            local_file: The local file to upload.
            artifact_file_path: The path to the artifact file.
            get_credentials: The function to call to get new credentials.
            is_sync: If True, upload synchronously without threading.
        """
        try:
            headers = self._extract_headers_from_credentials(credentials.headers)

            # try to create the file
            self._retryable_adls_function(
                func=put_adls_file_creation,
                artifact_file_path=artifact_file_path,
                get_credentials=get_credentials,
                sas_url=credentials.signed_uri,
                headers=headers,
            )

            # next try to append the file
            file_size = os.path.getsize(local_file)
            num_chunks = _compute_num_chunks(local_file, MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE.get())
            use_single_part_upload = num_chunks == 1

            def upload_chunks_func(start_byte):
                return self._retryable_adls_function(
                    func=patch_adls_file_upload,
                    artifact_file_path=artifact_file_path,
                    get_credentials=get_credentials,
                    sas_url=credentials.signed_uri,
                    local_file=local_file,
                    start_byte=start_byte,
                    size=MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE.get(),
                    position=start_byte,
                    headers=headers,
                    is_single=use_single_part_upload,
                )

            if is_sync:
                # Upload chunks synchronously without threading
                for index in range(num_chunks):
                    start_byte = index * MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE.get()
                    upload_chunks_func(start_byte)
            else:
                # Upload chunks asynchronously with threading
                futures = {}
                for index in range(num_chunks):
                    start_byte = index * MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE.get()
                    future = self.chunk_thread_pool.submit(
                        upload_chunks_func,
                        start_byte,
                    )
                    futures[future] = index

                _, errors = _complete_futures(futures, local_file)
                if errors:
                    raise MlflowException(
                        f"Failed to upload at least one part of {artifact_file_path}. "
                        f"Errors: {errors}"
                    )

            # finally try to flush the file
            if not use_single_part_upload:
                self._retryable_adls_function(
                    func=patch_adls_flush,
                    artifact_file_path=artifact_file_path,
                    get_credentials=get_credentials,
                    sas_url=credentials.signed_uri,
                    position=file_size,
                    headers=headers,
                )
        except Exception as err:
            raise MlflowException(err)

    def _signed_url_upload_file(self, credentials, local_file):
        try:
            headers = self._extract_headers_from_credentials(credentials.headers)
            signed_write_uri = credentials.signed_uri
            # Putting an empty file in a request by reading file bytes gives 501 error.
            if os.stat(local_file).st_size == 0:
                with cloud_storage_http_request(
                    "put", signed_write_uri, data="", headers=headers
                ) as response:
                    augmented_raise_for_status(response)
            else:
                with open(local_file, "rb") as file:
                    with cloud_storage_http_request(
                        "put", signed_write_uri, data=file, headers=headers
                    ) as response:
                        augmented_raise_for_status(response)
        except Exception as err:
            raise MlflowException(err)

    def _upload_to_cloud(self, cloud_credential_info, src_file_path, artifact_file_path):
        """
        Upload a local file to the cloud. Note that in this artifact repository, files are uploaded
        to resource relative artifact file paths in the artifact repository.

        Args:
            cloud_credential_info: ArtifactCredentialInfo object with presigned URL for the file.
            src_file_path: Local source file path for the upload.
            artifact_file_path: Path in the artifact repository, relative to the resource root path,
                where the artifact will be logged.

        """
        if cloud_credential_info.type == ArtifactCredentialType.AZURE_SAS_URI:
            self._azure_upload_file(
                cloud_credential_info,
                src_file_path,
                artifact_file_path,
                get_credentials=self._get_write_credential_infos,
            )
        elif cloud_credential_info.type == ArtifactCredentialType.AZURE_ADLS_GEN2_SAS_URI:
            self._azure_adls_gen2_upload_file(
                cloud_credential_info,
                src_file_path,
                artifact_file_path,
                self._get_write_credential_infos,
            )
        elif cloud_credential_info.type == ArtifactCredentialType.AWS_PRESIGNED_URL:
            if os.path.getsize(src_file_path) > MLFLOW_MULTIPART_UPLOAD_MINIMUM_FILE_SIZE.get():
                _validate_chunk_size_aws(MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE.get())
                self._multipart_upload(src_file_path, artifact_file_path)
            else:
                self._signed_url_upload_file(cloud_credential_info, src_file_path)
        elif cloud_credential_info.type == ArtifactCredentialType.GCP_SIGNED_URL:
            self._signed_url_upload_file(cloud_credential_info, src_file_path)
        else:
            raise MlflowException(
                message="Cloud provider not supported.", error_code=INTERNAL_ERROR
            )

    def _download_from_cloud(self, remote_file_path, local_path):
        """
        Download a file from the input `remote_file_path` and save it to `local_path`.

        Args:
            remote_file_path: Path relative to the resource root path to file in remote artifact
                repository.
            local_path: Local path to download file to.

        """
        read_credentials = self._get_read_credential_infos(remote_file_path)
        # Read credentials for only one file were requested. So we expected only one value in
        # the response.
        assert len(read_credentials) == 1
        cloud_credential_info = read_credentials[0]

        if cloud_credential_info.type not in [
            ArtifactCredentialType.AZURE_SAS_URI,
            ArtifactCredentialType.AZURE_ADLS_GEN2_SAS_URI,
            ArtifactCredentialType.AWS_PRESIGNED_URL,
            ArtifactCredentialType.GCP_SIGNED_URL,
        ]:
            raise MlflowException(
                message="Cloud provider not supported.", error_code=INTERNAL_ERROR
            )
        try:
            download_file_using_http_uri(
                cloud_credential_info.signed_uri,
                local_path,
                MLFLOW_MULTIPART_DOWNLOAD_CHUNK_SIZE.get(),
                self._extract_headers_from_credentials(cloud_credential_info.headers),
            )
        except Exception as err:
            raise MlflowException(err)

    def _create_multipart_upload(self, run_id, path, num_parts):
        return self._call_endpoint(
            DatabricksMlflowArtifactsService,
            CreateMultipartUpload,
            message_to_json(CreateMultipartUpload(run_id=run_id, path=path, num_parts=num_parts)),
        )

    def _get_presigned_upload_part_url(self, run_id, path, upload_id, part_number):
        return self._call_endpoint(
            DatabricksMlflowArtifactsService,
            GetPresignedUploadPartUrl,
            message_to_json(
                GetPresignedUploadPartUrl(
                    run_id=run_id, path=path, upload_id=upload_id, part_number=part_number
                )
            ),
        )

    def _upload_part(self, cred_info, data):
        headers = self._extract_headers_from_credentials(cred_info.headers)
        with cloud_storage_http_request(
            "put",
            cred_info.signed_uri,
            data=data,
            headers=headers,
        ) as response:
            augmented_raise_for_status(response)
            return response.headers["ETag"]

    def _upload_part_retry(self, cred_info, upload_id, part_number, local_file, start_byte, size):
        data = read_chunk(local_file, size, start_byte)
        try:
            return self._upload_part(cred_info, data)
        except requests.HTTPError as e:
            if e.response.status_code not in (401, 403):
                raise e
            _logger.info(
                "Failed to authorize request, possibly due to credential expiration."
                " Refreshing credentials and trying again..."
            )
            resp = self._get_presigned_upload_part_url(
                cred_info.run_id, cred_info.path, upload_id, part_number
            )
            return self._upload_part(resp.upload_credential_info, data)

    def _upload_parts(self, local_file, create_mpu_resp):
        futures = {}
        for index, cred_info in enumerate(create_mpu_resp.upload_credential_infos):
            part_number = index + 1
            start_byte = index * MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE.get()
            future = self.chunk_thread_pool.submit(
                self._upload_part_retry,
                cred_info=cred_info,
                upload_id=create_mpu_resp.upload_id,
                part_number=part_number,
                local_file=local_file,
                start_byte=start_byte,
                size=MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE.get(),
            )
            futures[future] = part_number

        results, errors = _complete_futures(futures, local_file)
        if errors:
            raise MlflowException(
                f"Failed to upload at least one part of {local_file}. Errors: {errors}"
            )

        return [
            PartEtag(part_number=part_number, etag=results[part_number])
            for part_number in sorted(results)
        ]

    def _complete_multipart_upload(self, run_id, path, upload_id, part_etags):
        return self._call_endpoint(
            DatabricksMlflowArtifactsService,
            CompleteMultipartUpload,
            message_to_json(
                CompleteMultipartUpload(
                    run_id=run_id,
                    path=path,
                    upload_id=upload_id,
                    part_etags=part_etags,
                )
            ),
        )

    def _abort_multipart_upload(self, cred_info):
        headers = self._extract_headers_from_credentials(cred_info.headers)
        with cloud_storage_http_request(
            "delete", cred_info.signed_uri, headers=headers
        ) as response:
            augmented_raise_for_status(response)
            return response

    def _multipart_upload(self, local_file, artifact_file_path):
        run_relative_artifact_path = posixpath.join(
            self.resource.relative_path, artifact_file_path or ""
        )
        num_parts = _compute_num_chunks(local_file, MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE.get())
        create_mpu_resp = self._create_multipart_upload(
            self.resource.id, run_relative_artifact_path, num_parts
        )
        try:
            part_etags = self._upload_parts(local_file, create_mpu_resp)
            self._complete_multipart_upload(
                self.resource.id,
                run_relative_artifact_path,
                create_mpu_resp.upload_id,
                part_etags,
            )
        except Exception as e:
            _logger.warning(
                "Encountered an unexpected error during multipart upload: %s, aborting", e
            )
            self._abort_multipart_upload(create_mpu_resp.abort_credential_info)
            raise e

    def log_artifact(self, local_file, artifact_path=None):
        src_file_name = os.path.basename(local_file)
        artifact_file_path = posixpath.join(artifact_path or "", src_file_name)
        write_credential_info = self._get_write_credential_infos([artifact_file_path])[0]
        self._upload_to_cloud(
            cloud_credential_info=write_credential_info,
            src_file_path=local_file,
            artifact_file_path=artifact_file_path,
        )

    def list_artifacts(self, path: str | None = None) -> list[FileInfo]:
        return self.resource.list_artifacts(path)

    def delete_artifacts(self, artifact_path=None):
        raise MlflowException("Not implemented yet")
