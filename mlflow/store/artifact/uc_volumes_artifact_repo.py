import os
import posixpath
from pathlib import Path
from typing import Optional

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
from mlflow.utils.uri import (
    get_databricks_profile_uri_from_artifact_uri,
    is_databricks_model_registry_artifacts_uri,
    is_valid_uc_volume_uri,
    remove_databricks_profile_info_from_artifact_uri,
    strip_scheme,
)

# https://docs.databricks.com/api/workspace/files
DIRECTORIES_API_ENDPOINT = "/api/2.0/fs/directories"
FILES_API_ENDPOINT = "/api/2.0/fs/files"
DOWNLOAD_CHUNK_SIZE = 1024


def _get_host_creds_factory(artist_uri: str):
    if databricks_profile_uri := get_databricks_profile_uri_from_artifact_uri(artist_uri):
        hostcreds_from_uri = get_databricks_host_creds(databricks_profile_uri)
        return lambda: hostcreds_from_uri
    return _get_host_creds_from_default_store()


class UCVolumesRestArtifactRepository(ArtifactRepository):
    """
    Stores artifacts on UC Volumes using the Files REST API.
    """

    def __init__(self, artifact_uri):
        if not is_valid_uc_volume_uri(artifact_uri):
            raise MlflowException(
                message=(
                    f"UC volume URI must be of the form "
                    f"dbfs:/Volumes/<catalog>/<schema>/<volume>/<path>: {artifact_uri}"
                ),
                error_code=INVALID_PARAMETER_VALUE,
            )

        # The dbfs:/ path ultimately used for artifact operations should not contain the
        # Databricks profile info, so strip it before setting `artifact_uri`.
        super().__init__(remove_databricks_profile_info_from_artifact_uri(artifact_uri))

        # Absolute path to the root of the volume. For example, '/Volumes/my-volume' for
        # 'dbfs:/Volumes/my-volume'.
        self.root_path = "/" + strip_scheme(self.artifact_uri).strip("/")
        self.get_host_creds = _get_host_creds_factory(artifact_uri)

    def _relative_to_root(self, path):
        return posixpath.relpath(path, self.root_path)

    def _api_request(self, endpoint, method, **kwargs):
        return http_request(
            host_creds=self.get_host_creds(), endpoint=endpoint, method=method, **kwargs
        )

    def _list_directory_contents(self, directory_path: str, next_page_token: Optional[str] = None):
        """
        Lists the contents of a directory.

        Args:
            directory_path: The absolute path of a directory.

        Returns:
            The response from the API.

        See also:
            https://docs.databricks.com/api/workspace/files/listdirectorycontents
        """
        endpoint = f"{DIRECTORIES_API_ENDPOINT}{directory_path}"
        return self._api_request(endpoint=endpoint, method="GET")

    def _paginated_list_directory_contents(self, path, next_page_token=None):
        response = self._list_directory_contents(
            self._get_path(path), next_page_token=next_page_token
        )
        if response.status_code == 404:
            return []

        augmented_raise_for_status(response)

        response_json = response.json()
        contents = response_json.get("contents", [])
        if next_page_token := response_json.get("next_page_token"):
            return contents + self._paginated_list_directory_contents(path, next_page_token)
        return contents

    def _download(self, output_path: str, file_path: str):
        """
        Downloads a file.

        Args:
            output_path: The local path to save the downloaded file.
            file_path: The absolute path of the file to download.

        Returns:
            The response from the API.

        See also:
            https://docs.databricks.com/api/workspace/files/download
        """
        endpoint = f"{FILES_API_ENDPOINT}{file_path}"
        with open(output_path, "wb") as f:
            with self._api_request(endpoint=endpoint, method="GET", stream=True) as resp:
                for content in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    f.write(content)
                return resp

    def _upload(self, local_file, file_path):
        """
        Uploads a file.

        Args:
            local_file: The local path of the file to upload.
            file_path: The absolute path of the file to upload.

        Returns:
            The response from the API.

        See also:
            https://docs.databricks.com/api/workspace/files/upload
        """
        endpoint = f"{FILES_API_ENDPOINT}{file_path}"
        with open(local_file, "rb") as f:
            return self._api_request(endpoint=endpoint, method="PUT", data=f, allow_redirects=False)

    def _get_path(self, artifact_path=None):
        return (
            posixpath.join(self.root_path, artifact_path.strip("/"))
            if artifact_path
            else self.root_path
        )

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
                    rel_path = local_path.parent.relative_to(local_dir)
                    posix_rel_path = relative_path_to_artifact_path(rel_path)
                    artifact_subdir = (
                        posixpath.join(artifact_path, posix_rel_path)
                        if artifact_path
                        else posix_rel_path
                    )
                self.log_artifact(local_path, artifact_subdir)

    def list_artifacts(self, path=None):
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
        for content in self._paginated_list_directory_contents(path):
            rel_path = self._relative_to_root(content["path"])
            infos.append(FileInfo(rel_path, content["is_directory"], content.get("file_size")))
        return sorted(infos, key=lambda f: f.path)

    def _download_file(self, remote_file_path, local_path):
        resp = self._download(output_path=local_path, file_path=self._get_path(remote_file_path))
        augmented_raise_for_status(resp)

    def delete_artifacts(self, artifact_path=None):
        raise NotImplementedError("Not implemented yet")


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
    if not is_valid_uc_volume_uri(artifact_uri):
        raise MlflowException(
            message=(
                f"UC volume URI must be of the form "
                f"dbfs:/Volumes/<catalog>/<schema>/<volume>/<path>: {artifact_uri}"
            ),
            error_code=INVALID_PARAMETER_VALUE,
        )

    cleaned_artifact_uri = artifact_uri.rstrip("/")
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
        path = strip_scheme(final_artifact_uri).strip("/")
        return LocalArtifactRepository(f"file:///{path}")
    return UCVolumesRestArtifactRepository(cleaned_artifact_uri)
