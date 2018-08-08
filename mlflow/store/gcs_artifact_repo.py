import os

from six.moves import urllib

from mlflow.entities.file_info import FileInfo
from mlflow.store.artifact_repo import ArtifactRepository
from mlflow.utils.file_utils import build_path, get_relative_path, TempDir


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
        super(GCSArtifactRepository, self).__init__(artifact_uri)

    @staticmethod
    def parse_gcs_uri(uri):
        """Parse an GCS URI, returning (bucket, path)"""
        parsed = urllib.parse.urlparse(uri)
        if parsed.scheme != "gs":
            raise Exception("Not a GCS URI: %s" % uri)
        path = parsed.path
        if path.startswith('/'):
            path = path[1:]
        return parsed.netloc, path

    def log_artifact(self, local_file, artifact_path=None):
        (bucket, dest_path) = self.parse_gcs_uri(self.artifact_uri)
        if artifact_path:
            dest_path = build_path(dest_path, artifact_path)
        dest_path = build_path(dest_path, os.path.basename(local_file))

        gcs_bucket = self.gcs.Client().get_bucket(bucket)
        blob = gcs_bucket.blob(dest_path)
        blob.upload_from_filename(local_file)

    def log_artifacts(self, local_dir, artifact_path=None):
        (bucket, dest_path) = self.parse_gcs_uri(self.artifact_uri)
        if artifact_path:
            dest_path = build_path(dest_path, artifact_path)
        gcs_bucket = self.gcs.Client().get_bucket(bucket)

        local_dir = os.path.abspath(local_dir)
        for (root, _, filenames) in os.walk(local_dir):
            upload_path = dest_path
            if root != local_dir:
                rel_path = get_relative_path(local_dir, root)
                upload_path = build_path(dest_path, rel_path)
            for f in filenames:
                path = build_path(upload_path, f)
                gcs_bucket.blob(path).upload_from_filename(build_path(root, f))

    def list_artifacts(self, path=None):
        (bucket, artifact_path) = self.parse_gcs_uri(self.artifact_uri)
        dest_path = artifact_path
        if path:
            dest_path = build_path(dest_path, path)
        prefix = dest_path + "/"

        bkt = self.gcs.Client().get_bucket(bucket)

        infos = self._list_folders(bkt, prefix, artifact_path)

        results = bkt.list_blobs(prefix=prefix, delimiter="/")
        for result in results:
            blob_path = result.name[len(artifact_path)+1:]
            infos.append(FileInfo(blob_path, False, result.size))

        return sorted(infos, key=lambda f: f.path)

    def _list_folders(self, bkt, prefix, artifact_path):
        results = bkt.list_blobs(prefix=prefix, delimiter="/")
        dir_paths = set()
        for page in results.pages:
            dir_paths.update(page.prefixes)

        return [FileInfo(path[len(artifact_path)+1:-1], True, None)for path in dir_paths]

    def download_artifacts(self, artifact_path):
        with TempDir(remove_on_exit=False) as tmp:
            return self._download_artifacts_into(artifact_path, tmp.path())

    def _download_artifacts_into(self, artifact_path, dest_dir):
        """Private version of download_artifacts that takes a destination directory."""
        basename = os.path.basename(artifact_path)
        local_path = build_path(dest_dir, basename)
        listing = self.list_artifacts(artifact_path)
        if len(listing) > 0:
            # Artifact_path is a directory, so make a directory for it and download everything
            os.mkdir(local_path)
            for file_info in listing:
                self._download_artifacts_into(file_info.path, local_path)
        else:
            (bucket, remote_path) = self.parse_gcs_uri(self.artifact_uri)
            remote_path = build_path(remote_path, artifact_path)
            gcs_bucket = self.gcs.Client().get_bucket(bucket)
            gcs_bucket.get_blob(remote_path).download_to_filename(local_path)
        return local_path
