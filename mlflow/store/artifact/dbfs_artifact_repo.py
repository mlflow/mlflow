import json
import os
import posixpath
from typing import Optional

import mlflow.utils.databricks_utils
from mlflow.entities import FileInfo
from mlflow.environment_variables import MLFLOW_ENABLE_DBFS_FUSE_ARTIFACT_REPO
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.store.artifact.databricks_artifact_repo import DatabricksArtifactRepository
from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository
from mlflow.store.tracking.rest_store import RestStore
from mlflow.tracking._tracking_service import utils
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.file_utils import relative_path_to_artifact_path
from mlflow.utils.rest_utils import (
    RESOURCE_NON_EXISTENT,
    http_request,
    http_request_safe,
)
from mlflow.utils.string_utils import strip_prefix
from mlflow.utils.uri import (
    get_databricks_profile_uri_from_artifact_uri,
    is_databricks_acled_artifacts_uri,
    is_databricks_model_registry_artifacts_uri,
    is_valid_dbfs_uri,
    remove_databricks_profile_info_from_artifact_uri,
    strip_scheme,
)

# The following constants are defined as @developer_stable
LIST_API_ENDPOINT = "/api/2.0/dbfs/list"
GET_STATUS_ENDPOINT = "/api/2.0/dbfs/get-status"
DOWNLOAD_CHUNK_SIZE = 1024


class DbfsRestArtifactRepository(ArtifactRepository):
    """
    Stores artifacts on DBFS using the DBFS REST API.

    This repository is used with URIs of the form ``dbfs:/<path>``. The repository can only be used
    together with the RestStore.
    """

    def __init__(self, artifact_uri):
        if not is_valid_dbfs_uri(artifact_uri):
            raise MlflowException(
                message="DBFS URI must be of the form dbfs:/<path> or "
                + "dbfs://profile@databricks/<path>",
                error_code=INVALID_PARAMETER_VALUE,
            )

        # The dbfs:/ path ultimately used for artifact operations should not contain the
        # Databricks profile info, so strip it before setting ``artifact_uri``.
        super().__init__(remove_databricks_profile_info_from_artifact_uri(artifact_uri))

        databricks_profile_uri = get_databricks_profile_uri_from_artifact_uri(artifact_uri)
        if databricks_profile_uri:
            hostcreds_from_uri = get_databricks_host_creds(databricks_profile_uri)
            self.get_host_creds = lambda: hostcreds_from_uri
        else:
            self.get_host_creds = _get_host_creds_from_default_store()

    def _databricks_api_request(self, endpoint, method, **kwargs):
        host_creds = self.get_host_creds()
        return http_request_safe(host_creds=host_creds, endpoint=endpoint, method=method, **kwargs)

    def _dbfs_list_api(self, json):
        host_creds = self.get_host_creds()
        return http_request(
            host_creds=host_creds, endpoint=LIST_API_ENDPOINT, method="GET", params=json
        )

    def _dbfs_download(self, output_path, endpoint):
        with open(output_path, "wb") as f:
            response = self._databricks_api_request(endpoint=endpoint, method="GET", stream=True)
            try:
                for content in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    f.write(content)
            finally:
                response.close()

    def _is_directory(self, artifact_path):
        dbfs_path = self._get_dbfs_path(artifact_path) if artifact_path else self._get_dbfs_path("")
        return self._dbfs_is_dir(dbfs_path)

    def _dbfs_is_dir(self, dbfs_path):
        response = self._databricks_api_request(
            endpoint=GET_STATUS_ENDPOINT, method="GET", params={"path": dbfs_path}
        )
        json_response = json.loads(response.text)
        try:
            return json_response["is_dir"]
        except KeyError:
            raise MlflowException(f"DBFS path {dbfs_path} does not exist")

    def _get_dbfs_path(self, artifact_path):
        return "/{}/{}".format(
            strip_scheme(self.artifact_uri).lstrip("/"),
            artifact_path.lstrip("/"),
        )

    def _get_dbfs_endpoint(self, artifact_path):
        return f"/dbfs{self._get_dbfs_path(artifact_path)}"

    def log_artifact(self, local_file, artifact_path=None):
        basename = os.path.basename(local_file)
        if artifact_path:
            http_endpoint = self._get_dbfs_endpoint(posixpath.join(artifact_path, basename))
        else:
            http_endpoint = self._get_dbfs_endpoint(basename)
        if os.stat(local_file).st_size == 0:
            # The API frontend doesn't like it when we post empty files to it using
            # `requests.request`, potentially due to the bug described in
            # https://github.com/requests/requests/issues/4215
            self._databricks_api_request(
                endpoint=http_endpoint, method="POST", data="", allow_redirects=False
            )
        else:
            with open(local_file, "rb") as f:
                self._databricks_api_request(
                    endpoint=http_endpoint, method="POST", data=f, allow_redirects=False
                )

    def log_artifacts(self, local_dir, artifact_path=None):
        artifact_path = artifact_path or ""
        for dirpath, _, filenames in os.walk(local_dir):
            artifact_subdir = artifact_path
            if dirpath != local_dir:
                rel_path = os.path.relpath(dirpath, local_dir)
                rel_path = relative_path_to_artifact_path(rel_path)
                artifact_subdir = posixpath.join(artifact_path, rel_path)
            for name in filenames:
                file_path = os.path.join(dirpath, name)
                self.log_artifact(file_path, artifact_subdir)

    def list_artifacts(self, path: Optional[str] = None) -> list:
        dbfs_path = self._get_dbfs_path(path) if path else self._get_dbfs_path("")
        dbfs_list_json = {"path": dbfs_path}
        response = self._dbfs_list_api(dbfs_list_json)
        try:
            json_response = json.loads(response.text)
        except ValueError:
            raise MlflowException(
                f"API request to list files under DBFS path {dbfs_path} failed with "
                f"status code {response.status_code}. Response body: {response.text}"
            )
        # /api/2.0/dbfs/list will not have the 'files' key in the response for empty directories.
        infos = []
        artifact_prefix = strip_prefix(self.artifact_uri, "dbfs:")
        if json_response.get("error_code", None) == RESOURCE_NON_EXISTENT:
            return []
        dbfs_files = json_response.get("files", [])
        for dbfs_file in dbfs_files:
            stripped_path = strip_prefix(dbfs_file["path"], artifact_prefix + "/")
            # If `path` is a file, the DBFS list API returns a single list element with the
            # same name as `path`. The list_artifacts API expects us to return an empty list in this
            # case, so we do so here.
            if stripped_path == path:
                return []
            is_dir = dbfs_file["is_dir"]
            artifact_size = None if is_dir else dbfs_file["file_size"]
            infos.append(FileInfo(stripped_path, is_dir, artifact_size))
        return sorted(infos, key=lambda f: f.path)

    def _download_file(self, remote_file_path, local_path):
        self._dbfs_download(
            output_path=local_path, endpoint=self._get_dbfs_endpoint(remote_file_path)
        )

    def delete_artifacts(self, artifact_path=None):
        raise MlflowException("Not implemented yet")


def _get_host_creds_from_default_store():
    store = utils._get_store()
    if not isinstance(store, RestStore):
        raise MlflowException(
            "Failed to get credentials for DBFS; they are read from the "
            + "Databricks CLI credentials or MLFLOW_TRACKING* environment "
            + "variables."
        )
    return store.get_host_creds


def dbfs_artifact_repo_factory(artifact_uri):
    """
    Returns an ArtifactRepository subclass for storing artifacts on DBFS.

    This factory method is used with URIs of the form ``dbfs:/<path>``. DBFS-backed artifact
    storage can only be used together with the RestStore.

    In the special case where the URI is of the form
    `dbfs:/databricks/mlflow-tracking/<Exp-ID>/<Run-ID>/<path>',
    a DatabricksArtifactRepository is returned. This is capable of storing access controlled
    artifacts.

    Args:
        artifact_uri: DBFS root artifact URI.

    Returns:
        Subclass of ArtifactRepository capable of storing artifacts on DBFS.
    """
    if not is_valid_dbfs_uri(artifact_uri):
        raise MlflowException(
            "DBFS URI must be of the form dbfs:/<path> or "
            + "dbfs://profile@databricks/<path>, but received "
            + artifact_uri
        )

    cleaned_artifact_uri = artifact_uri.rstrip("/")
    db_profile_uri = get_databricks_profile_uri_from_artifact_uri(cleaned_artifact_uri)
    if is_databricks_acled_artifacts_uri(artifact_uri):
        return DatabricksArtifactRepository(cleaned_artifact_uri)
    elif (
        mlflow.utils.databricks_utils.is_dbfs_fuse_available()
        and MLFLOW_ENABLE_DBFS_FUSE_ARTIFACT_REPO.get()
        and not is_databricks_model_registry_artifacts_uri(artifact_uri)
        and (db_profile_uri is None or db_profile_uri == "databricks")
    ):
        # If the DBFS FUSE mount is available, write artifacts directly to
        # /dbfs/... using local filesystem APIs.
        # Note: it is possible for a named Databricks profile to point to the current workspace,
        # but we're going to avoid doing a complex check and assume users will use `databricks`
        # to mean the current workspace. Using `DbfsRestArtifactRepository` to access the current
        # workspace's DBFS should still work; it just may be slower.
        final_artifact_uri = remove_databricks_profile_info_from_artifact_uri(cleaned_artifact_uri)
        file_uri = "file:///dbfs/{}".format(strip_prefix(final_artifact_uri, "dbfs:/"))
        return LocalArtifactRepository(file_uri)
    return DbfsRestArtifactRepository(cleaned_artifact_uri)
