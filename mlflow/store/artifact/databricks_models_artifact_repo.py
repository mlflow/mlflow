import json
import logging
import os
import posixpath
from typing import Optional

import mlflow.tracking
from mlflow.entities import FileInfo
from mlflow.environment_variables import (
    MLFLOW_ENABLE_MULTIPART_DOWNLOAD,
    MLFLOW_MULTIPART_DOWNLOAD_CHUNK_SIZE,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.store.artifact.utils.models import (
    get_model_name_and_version,
    is_using_databricks_registry,
)
from mlflow.utils.databricks_utils import (
    get_databricks_host_creds,
    warn_on_deprecated_cross_workspace_registry_uri,
)
from mlflow.utils.file_utils import (
    download_chunk_retries,
    download_file_using_http_uri,
    parallelized_download_file_using_http_uri,
    remove_on_error,
)
from mlflow.utils.rest_utils import http_request
from mlflow.utils.uri import get_databricks_profile_uri_from_artifact_uri

_logger = logging.getLogger(__name__)
# The constant REGISTRY_LIST_ARTIFACT_ENDPOINT is defined as @developer_stable
REGISTRY_LIST_ARTIFACTS_ENDPOINT = "/api/2.0/mlflow/model-versions/list-artifacts"
# The constant REGISTRY_ARTIFACT_PRESIGNED_URI_ENDPOINT is defined as @developer_stable
REGISTRY_ARTIFACT_PRESIGNED_URI_ENDPOINT = "/api/2.0/mlflow/model-versions/get-signed-download-uri"


class DatabricksModelsArtifactRepository(ArtifactRepository):
    """
    Performs storage operations on artifacts controlled by a Databricks-hosted model registry.

    Signed access URIs for the appropriate cloud storage locations are fetched from the
    MLflow service and used to download model artifacts.

    The artifact_uri is expected to be of the form
    - `models:/<model_name>/<model_version>`
    - `models:/<model_name>/<stage>`  (refers to the latest model version in the given stage)
    - `models:/<model_name>/latest`  (refers to the latest of all model versions)
    - `models://<profile>/<model_name>/<model_version or stage or 'latest'>`

    Note : This artifact repository is meant is to be instantiated by the ModelsArtifactRepository
    when the client is pointing to a Databricks-hosted model registry.
    """

    def __init__(self, artifact_uri: str, tracking_uri: Optional[str] = None) -> None:
        if not is_using_databricks_registry(artifact_uri):
            raise MlflowException(
                message="A valid databricks profile is required to instantiate this repository",
                error_code=INVALID_PARAMETER_VALUE,
            )
        super().__init__(artifact_uri, tracking_uri)
        from mlflow.tracking.client import MlflowClient

        self.databricks_profile_uri = (
            get_databricks_profile_uri_from_artifact_uri(artifact_uri) or mlflow.get_registry_uri()
        )
        warn_on_deprecated_cross_workspace_registry_uri(self.databricks_profile_uri)
        client = MlflowClient(registry_uri=self.databricks_profile_uri)
        self.model_name, self.model_version = get_model_name_and_version(client, artifact_uri)
        # Use an isolated thread pool executor for chunk uploads/downloads to avoid a deadlock
        # caused by waiting for a chunk-upload/download task within a file-upload/download task.
        # See https://superfastpython.com/threadpoolexecutor-deadlock/#Deadlock_1_Submit_and_Wait_for_a_Task_Within_a_Task
        # for more details
        self.chunk_thread_pool = self._create_thread_pool()

    def _call_endpoint(self, json, endpoint):
        db_creds = get_databricks_host_creds(self.databricks_profile_uri)
        return http_request(host_creds=db_creds, endpoint=endpoint, method="GET", params=json)

    def _make_json_body(self, path, page_token=None):
        body = {"name": self.model_name, "version": self.model_version, "path": path}
        if page_token:
            body["page_token"] = page_token
        return body

    def list_artifacts(self, path: Optional[str] = None) -> list[FileInfo]:
        infos = []
        page_token = None
        if not path:
            path = ""
        while True:
            json_body = self._make_json_body(path, page_token)
            response = self._call_endpoint(json_body, REGISTRY_LIST_ARTIFACTS_ENDPOINT)
            try:
                response.raise_for_status()
                json_response = json.loads(response.text)
            except Exception:
                raise MlflowException(
                    f"API request to list files under path `{path}` failed with status code "
                    f"{response.status_code}. Response body: {response.text}"
                )
            artifact_list = json_response.get("files", [])
            next_page_token = json_response.get("next_page_token", None)
            # If `path` is a file, ListArtifacts returns a single list element with the
            # same name as `path`. The list_artifacts API expects us to return an empty list in this
            # case, so we do so here.
            if (
                len(artifact_list) == 1
                and artifact_list[0]["path"] == path
                and not artifact_list[0]["is_dir"]
            ):
                return []
            for output_file in artifact_list:
                artifact_size = None if output_file["is_dir"] else output_file["file_size"]
                infos.append(FileInfo(output_file["path"], output_file["is_dir"], artifact_size))
            if len(artifact_list) == 0 or not next_page_token:
                break
            page_token = next_page_token
        return infos

    # TODO: Change the implementation of this to match how databricks_artifact_repo.py handles this
    def _get_signed_download_uri(self, path=None):
        if not path:
            path = ""
        json_body = self._make_json_body(path)
        response = self._call_endpoint(json_body, REGISTRY_ARTIFACT_PRESIGNED_URI_ENDPOINT)
        try:
            json_response = json.loads(response.text)
        except ValueError:
            raise MlflowException(
                f"API request to get presigned uri to for file under path `{path}` failed with"
                f" status code {response.status_code}. Response body: {response.text}"
            )
        return json_response.get("signed_uri", None), json_response.get("headers", None)

    def _extract_headers_from_signed_url(self, headers):
        if headers is None:
            return {}
        filtered_headers = filter(lambda h: "name" in h and "value" in h, headers)
        return {header.get("name"): header.get("value") for header in filtered_headers}

    def _parallelized_download_from_cloud(
        self, signed_uri, headers, file_size, dst_local_file_path, dst_run_relative_artifact_path
    ):
        from mlflow.utils.databricks_utils import get_databricks_env_vars

        with remove_on_error(dst_local_file_path):
            parallel_download_subproc_env = os.environ.copy()
            parallel_download_subproc_env.update(
                get_databricks_env_vars(self.databricks_profile_uri)
            )
            failed_downloads = parallelized_download_file_using_http_uri(
                thread_pool_executor=self.chunk_thread_pool,
                http_uri=signed_uri,
                download_path=dst_local_file_path,
                remote_file_path=dst_run_relative_artifact_path,
                file_size=file_size,
                # URI type is not known in this context
                uri_type=None,
                chunk_size=MLFLOW_MULTIPART_DOWNLOAD_CHUNK_SIZE.get(),
                env=parallel_download_subproc_env,
                headers=headers,
            )
            if failed_downloads:
                new_signed_uri, new_headers = self._get_signed_download_uri(
                    dst_run_relative_artifact_path
                )
                new_headers = self._extract_headers_from_signed_url(new_headers)
                download_chunk_retries(
                    chunks=list(failed_downloads),
                    http_uri=new_signed_uri,
                    headers=new_headers,
                    download_path=dst_local_file_path,
                )

    def _download_file(self, remote_file_path, local_path):
        try:
            parent_dir, _ = posixpath.split(remote_file_path)
            file_infos = self.list_artifacts(parent_dir)
            file_info = [info for info in file_infos if info.path == remote_file_path]
            file_size = file_info[0].file_size if len(file_info) == 1 else None
            signed_uri, raw_headers = self._get_signed_download_uri(remote_file_path)
            headers = {}
            if raw_headers is not None:
                # Don't send None to _extract_headers_from_signed_url
                headers = self._extract_headers_from_signed_url(raw_headers)
            if (
                not file_size
                or file_size <= MLFLOW_MULTIPART_DOWNLOAD_CHUNK_SIZE.get()
                or not MLFLOW_ENABLE_MULTIPART_DOWNLOAD.get()
            ):
                download_file_using_http_uri(
                    signed_uri, local_path, MLFLOW_MULTIPART_DOWNLOAD_CHUNK_SIZE.get(), headers
                )
            else:
                self._parallelized_download_from_cloud(
                    signed_uri,
                    headers,
                    file_size,
                    local_path,
                    remote_file_path,
                )

        except Exception as err:
            raise MlflowException(err)

    def log_artifact(self, local_file, artifact_path=None):
        raise MlflowException("This repository does not support logging artifacts.")

    def log_artifacts(self, local_dir, artifact_path=None):
        raise MlflowException("This repository does not support logging artifacts.")

    def delete_artifacts(self, artifact_path=None):
        raise NotImplementedError("This artifact repository does not support deleting artifacts")
