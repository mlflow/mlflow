import os
import posixpath
from pathlib import Path

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
from mlflow.utils.request_utils import augmented_raise_for_status
from mlflow.utils.rest_utils import http_request
from mlflow.utils.string_utils import strip_prefix
from mlflow.utils.uri import (
    get_databricks_profile_uri_from_artifact_uri,
    is_databricks_model_registry_artifacts_uri,
    is_valid_dbfs_uri,
    remove_databricks_profile_info_from_artifact_uri,
    strip_scheme,
)

# https://docs.databricks.com/api/workspace/files
DIRECTORIES_API_ENDPOINT = "/api/2.0/fs/directories"
FILES_API_ENDPOINT = "/api/2.0/fs/files"
DOWNLOAD_CHUNK_SIZE = 1024


class UCVolumesRestArtifactRepository(ArtifactRepository):
    """
    Stores artifacts on UC Volumes using the Files REST API.
    """

    def __init__(self, artifact_uri):
        if not is_valid_dbfs_uri(artifact_uri):
            raise MlflowException(
                message="Artifact URI must be of the form dbfs:/Volumes/<path>",
                error_code=INVALID_PARAMETER_VALUE,
            )

        # The dbfs:/ path ultimately used for artifact operations should not contain the
        # Databricks profile info, so strip it before setting ``artifact_uri``.
        super().__init__(remove_databricks_profile_info_from_artifact_uri(artifact_uri))

        self.root = "/" + strip_scheme(self.artifact_uri).strip("/")

        if databricks_profile_uri := get_databricks_profile_uri_from_artifact_uri(artifact_uri):
            hostcreds_from_uri = get_databricks_host_creds(databricks_profile_uri)
            self.get_host_creds = lambda: hostcreds_from_uri
        else:
            self.get_host_creds = _get_host_creds_from_default_store()

    def _api_request(self, endpoint, method, **kwargs):
        return http_request(
            host_creds=self.get_host_creds(), endpoint=endpoint, method=method, **kwargs
        )

    def _list_directory_contents(self, directory_path: str):
        # https://docs.databricks.com/api/workspace/files/listdirectorycontents
        endpoint = f"{DIRECTORIES_API_ENDPOINT}{directory_path}"
        return self._api_request(endpoint=endpoint, method="GET")

    def _download(self, output_path: str, file_path: str):
        # https://docs.databricks.com/api/workspace/files/download
        endpoint = f"{FILES_API_ENDPOINT}{file_path}"
        with open(output_path, "wb") as f:
            with self._api_request(endpoint=endpoint, method="GET", stream=True) as resp:
                for content in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    f.write(content)
                return resp

    def _upload(self, local_file, file_path):
        # https://docs.databricks.com/api/workspace/files/upload
        endpoint = f"{FILES_API_ENDPOINT}{file_path}"
        with open(local_file, "rb") as f:
            return self._api_request(endpoint=endpoint, method="PUT", data=f, allow_redirects=False)

    def _get_path(self, artifact_path=None):
        return self.root if artifact_path else posixpath.join(self.root, artifact_path.strip("/"))

    def log_artifact(self, local_file, artifact_path=None):
        basename = os.path.basename(local_file)
        artifact_path = posixpath.join(artifact_path, basename) if artifact_path else basename
        resp = self._upload(local_file, self._get_path(artifact_path))
        augmented_raise_for_status(resp)

    def log_artifacts(self, local_dir, artifact_path=None):
        local_dir = Path(local_dir).resolve()
        for local_path in local_dir.rglob("*"):
            if local_path.is_file():
                if local_path.parent == local_dir:
                    artifact_subdir = artifact_path
                else:
                    rel_path = local_path.relative_to(local_dir)
                    rel_path = relative_path_to_artifact_path(rel_path)
                    artifact_subdir = (
                        posixpath.join(artifact_path, rel_path) if artifact_path else rel_path
                    )
                self.log_artifact(local_path, artifact_subdir)

    def list_artifacts(self, path=None):
        response = self._list_directory_contents(self._get_path(path))
        if response.status_code == 404:
            return []
        json_response = response.json()

        artifact_prefix = strip_scheme(self.artifact_uri).rstrip("/")
        # Response sample (https://docs.databricks.com/api/workspace/files/listdirectorycontents):
        # {
        #    "contents": [
        #        {
        #            "path": "string",
        #            "is_directory": True,
        #            "file_size": 0,
        #            "last_modified": 0,
        #            "name": "string",
        #        }
        #    ],
        #    "next_page_token": "string",
        # }
        infos = []
        for file in json_response.get("contents", []):
            stripped_path = strip_prefix(file["path"], artifact_prefix + "/")
            # If `path` is a file, the Files list API returns a single list element with the
            # same name as `path`. The list_artifacts API expects us to return an empty list in this
            # case, so we do so here.
            if stripped_path == path:
                return []
            infos.append(FileInfo(stripped_path, file["is_directory"], file.get("file_size")))
        return sorted(infos, key=lambda f: f.path)

    def _download_file(self, remote_file_path, local_path):
        resp = self._download(output_path=local_path, endpoint=self._get_path(remote_file_path))
        augmented_raise_for_status(resp)

    def delete_artifacts(self, artifact_path=None):
        raise NotImplementedError()


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

    cleaned_artifact_uri = strip_scheme(artifact_uri).rstrip("/")
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
        path = strip_scheme(final_artifact_uri).lstrip("/")
        return LocalArtifactRepository(f"file:///{path}")
    return UCVolumesRestArtifactRepository(cleaned_artifact_uri)
