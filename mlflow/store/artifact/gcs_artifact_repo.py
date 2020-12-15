import os

import posixpath
import urllib.parse

from mlflow.entities import FileInfo
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.utils.file_utils import relative_path_to_artifact_path
from mlflow.exceptions import MlflowException


class GCSArtifactRepository(ArtifactRepository):
    """
    Stores artifacts on Google Cloud Storage.

    Assumes the google credentials are available in the environment,
    see https://google-cloud.readthedocs.io/en/latest/core/auth.html.
    """

    def __init__(self, artifact_uri, client=None):
        if client:
            self.gcs = client
        else:
            from google.cloud import storage as gcs_storage

            self.gcs = gcs_storage
        super().__init__(artifact_uri)

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
        from google.auth.exceptions import DefaultCredentialsError

        try:
            storage_client = self.gcs.Client()
        except DefaultCredentialsError:
            storage_client = self.gcs.Client.create_anonymous_client()
        return storage_client.bucket(bucket)

    def log_artifact(self, local_file, artifact_path=None):
        (bucket, dest_path) = self.parse_gcs_uri(self.artifact_uri)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        dest_path = posixpath.join(dest_path, os.path.basename(local_file))

        gcs_bucket = self._get_bucket(bucket)
        blob = gcs_bucket.blob(dest_path)
        blob.upload_from_filename(local_file)

    def log_artifacts(self, local_dir, artifact_path=None):
        (bucket, dest_path) = self.parse_gcs_uri(self.artifact_uri)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        gcs_bucket = self._get_bucket(bucket)

        local_dir = os.path.abspath(local_dir)
        for (root, _, filenames) in os.walk(local_dir):
            upload_path = dest_path
            if root != local_dir:
                rel_path = os.path.relpath(root, local_dir)
                rel_path = relative_path_to_artifact_path(rel_path)
                upload_path = posixpath.join(dest_path, rel_path)
            for f in filenames:
                path = posixpath.join(upload_path, f)
                gcs_bucket.blob(path).upload_from_filename(os.path.join(root, f))

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
        gcs_bucket.blob(remote_full_path).download_to_filename(local_path)

    def delete_artifacts(self, artifact_path=None):
        raise MlflowException("Not implemented yet")
