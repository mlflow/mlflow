import os

import posixpath
import urllib.parse

from mlflow.entities import FileInfo
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.utils.file_utils import relative_path_to_artifact_path
from mlflow.environment_variables import (
    MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT,
    MLFLOW_GCS_DEFAULT_TIMEOUT,
    MLFLOW_GCS_DOWNLOAD_CHUNK_SIZE,
    MLFLOW_GCS_UPLOAD_CHUNK_SIZE,
)


class GCSArtifactRepository(ArtifactRepository):
    """
    Stores artifacts on Google Cloud Storage.

    :param artifact_uri: URI of GCS bucket
    :param client: Optional. The client to use for GCS operations; a default
                       client object will be created if unspecified, using default
                       credentials as described in https://google-cloud.readthedocs.io/en/latest/core/auth.html
    """

    def __init__(self, artifact_uri, client=None):
        super().__init__(artifact_uri)
        from google.cloud.storage.constants import _DEFAULT_TIMEOUT
        from google.auth.exceptions import DefaultCredentialsError
        from google.cloud import storage as gcs_storage

        self._GCS_DOWNLOAD_CHUNK_SIZE = MLFLOW_GCS_DOWNLOAD_CHUNK_SIZE.get()
        self._GCS_UPLOAD_CHUNK_SIZE = MLFLOW_GCS_UPLOAD_CHUNK_SIZE.get()
        self._GCS_DEFAULT_TIMEOUT = (
            MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT.get()
            or MLFLOW_GCS_DEFAULT_TIMEOUT.get()
            or _DEFAULT_TIMEOUT
        )

        # If the user-supplied timeout environment variable value is -1,
        # use `None` for `self._GCS_DEFAULT_TIMEOUT`
        # to use indefinite timeout
        self._GCS_DEFAULT_TIMEOUT = (
            None if self._GCS_DEFAULT_TIMEOUT == -1 else self._GCS_DEFAULT_TIMEOUT
        )
        if client is not None:
            self.client = client
        else:
            try:
                self.client = gcs_storage.Client()
            except DefaultCredentialsError:
                self.client = gcs_storage.Client.create_anonymous_client()

    @staticmethod
    def parse_gcs_uri(uri):
        """Parse an GCS URI, returning (bucket, path)"""
        parsed = urllib.parse.urlparse(uri)
        if parsed.scheme != "gs":
            raise Exception("Not a GCS URI: %s" % uri)
        path = parsed.path
        if path.startswith("/"):
            path = path[1:]
        return parsed.netloc, path

    def _get_bucket(self, bucket):
        return self.client.bucket(bucket)

    def log_artifact(self, local_file, artifact_path=None):
        (bucket, dest_path) = self.parse_gcs_uri(self.artifact_uri)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        dest_path = posixpath.join(dest_path, os.path.basename(local_file))

        gcs_bucket = self._get_bucket(bucket)
        blob = gcs_bucket.blob(dest_path, chunk_size=self._GCS_UPLOAD_CHUNK_SIZE)
        blob.upload_from_filename(local_file, timeout=self._GCS_DEFAULT_TIMEOUT)

    def log_artifacts(self, local_dir, artifact_path=None):
        (bucket, dest_path) = self.parse_gcs_uri(self.artifact_uri)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        gcs_bucket = self._get_bucket(bucket)

        local_dir = os.path.abspath(local_dir)
        for root, _, filenames in os.walk(local_dir):
            upload_path = dest_path
            if root != local_dir:
                rel_path = os.path.relpath(root, local_dir)
                rel_path = relative_path_to_artifact_path(rel_path)
                upload_path = posixpath.join(dest_path, rel_path)
            for f in filenames:
                path = posixpath.join(upload_path, f)
                gcs_bucket.blob(path, chunk_size=self._GCS_UPLOAD_CHUNK_SIZE).upload_from_filename(
                    os.path.join(root, f), timeout=self._GCS_DEFAULT_TIMEOUT
                )

    def list_artifacts(self, path=None):
        (bucket, artifact_path) = self.parse_gcs_uri(self.artifact_uri)
        dest_path = artifact_path
        if path:
            dest_path = posixpath.join(dest_path, path)
        prefix = dest_path if dest_path.endswith("/") else dest_path + "/"

        bkt = self._get_bucket(bucket)

        infos = self._list_folders(bkt, prefix, artifact_path)

        results = bkt.list_blobs(prefix=prefix, delimiter="/")
        for result in results:
            # skip blobs matching current directory path as list_blobs api
            # returns subdirectories as well
            if result.name == prefix:
                continue
            blob_path = result.name[len(artifact_path) + 1 :]
            infos.append(FileInfo(blob_path, False, result.size))

        return sorted(infos, key=lambda f: f.path)

    def _list_folders(self, bkt, prefix, artifact_path):
        results = bkt.list_blobs(prefix=prefix, delimiter="/")
        dir_paths = set()
        for page in results.pages:
            dir_paths.update(page.prefixes)

        return [FileInfo(path[len(artifact_path) + 1 : -1], True, None) for path in dir_paths]

    def _download_file(self, remote_file_path, local_path):
        (bucket, remote_root_path) = self.parse_gcs_uri(self.artifact_uri)
        remote_full_path = posixpath.join(remote_root_path, remote_file_path)
        gcs_bucket = self._get_bucket(bucket)
        gcs_bucket.blob(
            remote_full_path, chunk_size=self._GCS_DOWNLOAD_CHUNK_SIZE
        ).download_to_filename(local_path, timeout=self._GCS_DEFAULT_TIMEOUT)

    def delete_artifacts(self, artifact_path=None):
        (bucket_name, dest_path) = self.parse_gcs_uri(self.artifact_uri)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)

        gcs_bucket = self._get_bucket(bucket_name)
        blobs = gcs_bucket.list_blobs(prefix=f"{dest_path}")
        for blob in blobs:
            blob.delete()
