import json
import os
import posixpath

import mlflow.utils.databricks_utils
from mlflow.entities import FileInfo
from mlflow.environment_variables import MLFLOW_ENABLE_DBFS_FUSE_ARTIFACT_REPO
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository
from mlflow.store.tracking.rest_store import RestStore
from mlflow.tracking._tracking_service import utils
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.file_utils import relative_path_to_artifact_path
from mlflow.utils.rest_utils import RESOURCE_NON_EXISTENT, http_request, http_request_safe
from mlflow.utils.string_utils import strip_prefix
from mlflow.utils.uri import (
    get_databricks_profile_uri_from_artifact_uri,
    is_databricks_model_registry_artifacts_uri,
    is_valid_dbfs_uri,
    remove_databricks_profile_info_from_artifact_uri,
)

# The following constants are defined as @developer_stable
DIR_API_ENDPOINT = "/api/2.0/fs/directories"
FILE_API_ENDPOINT = "/api/2.0/fs/files"
DOWNLOAD_CHUNK_SIZE = 1024


class UCVolumesRestArtifactRepository(ArtifactRepository):
    """
    Stores artifacts on Volumes using the Files REST API.

    This repository is used with URIs of the form `dbfs:/Volumes/<path>`. The repository can only be
    used together with the RestStore.
    """

    def __init__(self, artifact_uri):
        # TODO: add is_valid_volumes_uri
        if not is_valid_dbfs_uri(artifact_uri):
            raise MlflowException(
                message="DBFS URI must be of the form dbfs:/Volumes/<path>",
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

    def _volumes_list_api(self, path):
        host_creds = self.get_host_creds()
        return http_request(
            host_creds=host_creds, endpoint=f"{DIR_API_ENDPOINT}{path}", method="GET"
        )

    def _volumes_download(self, output_path, endpoint):
        with open(output_path, "wb") as f:
            response = self._databricks_api_request(endpoint=endpoint, method="GET", stream=True)
            try:
                for content in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    f.write(content)
            finally:
                response.close()

    def _is_directory(self, artifact_path):
        volume_path = (
            self._get_volumes_path(artifact_path) if artifact_path else self._get_volumes_path("")
        )
        return self._volumes_is_dir(volume_path)

    def _volumes_is_dir(self, volume_path):
        response = self._databricks_api_request(
            endpoint=f"{DIR_API_ENDPOINT}{volume_path}", method="HEAD"
        )
        try:
            return response.status_code == 200
        except KeyError:
            raise MlflowException(f"Volume path {volume_path} does not exist")

    def _get_volumes_path(self, artifact_path):
        return "/{}/{}".format(
            strip_prefix(self.artifact_uri, "dbfs:/"),
            strip_prefix(artifact_path, "/"),
        )

    def _get_volumes_endpoint(self, artifact_path):
        return f"{FILE_API_ENDPOINT}{self._get_volumes_path(artifact_path)}"

    def log_artifact(self, local_file, artifact_path=None):
        basename = os.path.basename(local_file)
        if artifact_path:
            http_endpoint = self._get_volumes_endpoint(posixpath.join(artifact_path, basename))
        else:
            http_endpoint = self._get_volumes_endpoint(basename)
        if os.stat(local_file).st_size == 0:
            # The API frontend doesn't like it when we post empty files to it using
            # `requests.request`, potentially due to the bug described in
            # https://github.com/requests/requests/issues/4215
            self._databricks_api_request(
                endpoint=http_endpoint, method="PUT", data="", allow_redirects=False
            )
        else:
            with open(local_file, "rb") as f:
                self._databricks_api_request(
                    endpoint=http_endpoint, method="PUT", data=f, allow_redirects=False
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

    def list_artifacts(self, path=None):
        volumes_path = self._get_volumes_path(path) if path else self._get_volumes_path("")
        response = self._volumes_list_api(volumes_path)
        try:
            json_response = json.loads(response.text)
        except ValueError:
            raise MlflowException(
                f"API request to list files under Volumes path {volumes_path} failed with "
                f"status code {response.status_code}. Response body: {response.text}"
            )
        # /api/2.0/fs will not have the 'files' key in the response for empty directories.
        infos = []
        artifact_prefix = strip_prefix(self.artifact_uri, "dbfs:")
        if json_response.get("error_code", None) == RESOURCE_NON_EXISTENT:
            return []
        volume_files = json_response.get("contents", [])
        for volume_file in volume_files:
            stripped_path = strip_prefix(volume_file["path"], artifact_prefix + "/")
            # If `path` is a file, the DBFS list API returns a single list element with the
            # same name as `path`. The list_artifacts API expects us to return an empty list in this
            # case, so we do so here.
            if stripped_path == path:
                return []
            is_dir = volume_file["is_directory"]
            artifact_size = None if is_dir else volume_file["file_size"]
            infos.append(FileInfo(stripped_path, is_dir, artifact_size))
        return sorted(infos, key=lambda f: f.path)

    def _download_file(self, remote_file_path, local_path):
        self._volumes_download(
            output_path=local_path, endpoint=self._get_volumes_endpoint(remote_file_path)
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


def uc_volumes_artifact_repo_factory(artifact_uri):
    """
    Returns an ArtifactRepository subclass for storing artifacts on Volumes.

    This factory method is used with URIs of the form ``dbfs:/Volumes/<path>``. Volume-backed
    artifact storage can only be used together with the RestStore.

    Args:
        artifact_uri: Volume root artifact URI.

    Returns:
        Subclass of ArtifactRepository capable of storing artifacts on DBFS.
    """
    if not is_valid_dbfs_uri(artifact_uri):
        raise MlflowException(
            "DBFS URI must be of the form dbfs:/Volumes/<path> or "
            + "dbfs://profile@databricks/Volumes/<path>, but received "
            + artifact_uri
        )

    cleaned_artifact_uri = artifact_uri.rstrip("dbfs:")
    db_profile_uri = get_databricks_profile_uri_from_artifact_uri(cleaned_artifact_uri)
    if (
        mlflow.utils.databricks_utils.is_dbfs_fuse_available()
        and MLFLOW_ENABLE_DBFS_FUSE_ARTIFACT_REPO.get()
        and not is_databricks_model_registry_artifacts_uri(artifact_uri)
        and (db_profile_uri is None or db_profile_uri == "databricks")
    ):
        # If the DBFS FUSE mount is available, write artifacts directly to
        # /Volumes/... using local filesystem APIs.
        # Note: it is possible for a named Databricks profile to point to the current workspace,
        # but we're going to avoid doing a complex check and assume users will use `databricks`
        # to mean the current workspace. Using `VolumeRestArtifactRepository` to access the current
        # workspace's Volumes should still work; it just may be slower.
        final_artifact_uri = remove_databricks_profile_info_from_artifact_uri(cleaned_artifact_uri)
        file_uri = "file:///{}".format(strip_prefix(final_artifact_uri, "dbfs:/"))
        return LocalArtifactRepository(file_uri)
    return UCVolumesRestArtifactRepository(cleaned_artifact_uri)
