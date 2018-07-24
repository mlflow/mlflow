from abc import abstractmethod, ABCMeta
import shutil
from six.moves import urllib
from distutils import dir_util
import os

import boto3
from google.cloud import storage as gcs_storage

from mlflow.utils.file_utils import (mkdir, exists, list_all, get_relative_path,
                                     get_file_info, build_path, TempDir)
from mlflow.entities.file_info import FileInfo


class ArtifactRepository:
    """
    Defines how to upload (log) and download potentially large artifacts from different
    storage backends.
    """

    __metaclass__ = ABCMeta

    def __init__(self, artifact_uri):
        self.artifact_uri = artifact_uri

    @abstractmethod
    def log_artifact(self, local_file, artifact_path=None):
        """
        Logs a local file as an artifact, optionally taking an ``artifact_path`` to place it in
        within the run's artifacts. Run artifacts can be organized into directories, so you can
        place the artifact in a directory this way.
        :param local_file: Path to artifact to log
        :param artifact_path: Directory within the run's artifact directory in which to log the
                              artifact
        """
        pass

    @abstractmethod
    def log_artifacts(self, local_dir, artifact_path=None):
        """
        Logs the files in the specified local directory as artifacts, optionally taking
        an ``artifact_path`` to place them in within the run's artifacts.
        :param local_dir: Directory of local artifacts to log
        :param artifact_path: Directory within the run's artifact directory in which to log the
                              artifacts
        """
        pass

    @abstractmethod
    def list_artifacts(self, path):
        """
        Return all the artifacts for this run_uuid directly under path.
        :param path: Relative source path that contain desired artifacts
        :return: List of artifacts as FileInfo listed directly under path.
        """
        pass

    @abstractmethod
    def download_artifacts(self, artifact_path):
        """
        Download an artifact file or directory to a local directory if applicable, and return a
        local path for it.
        :param path: Relative source path to the desired artifact
        :return: Full path desired artifact.
        """
        # TODO: Probably need to add a more efficient method to stream just a single artifact
        # without downloading it, or to get a pre-signed URL for cloud storage.
        pass

    @staticmethod
    def from_artifact_uri(artifact_uri):
        """
        Given an artifact URI for an Experiment Run (e.g., /local/file/path or s3://my/bucket),
        returns an ArtifactReposistory instance capable of logging and downloading artifacts
        on behalf of this URI.
        """
        if artifact_uri.startswith("s3:/"):
            return S3ArtifactRepository(artifact_uri)
        elif artifact_uri.startswith("gs:/"):
            return GCSArtifactRepository(artifact_uri)
        else:
            return LocalArtifactRepository(artifact_uri)


class LocalArtifactRepository(ArtifactRepository):
    """Stores artifacts as files in a local directory."""

    def log_artifact(self, local_file, artifact_path=None):
        artifact_dir = build_path(self.artifact_uri, artifact_path) \
            if artifact_path else self.artifact_uri
        if not exists(artifact_dir):
            mkdir(artifact_dir)
        shutil.copy(local_file, artifact_dir)

    def log_artifacts(self, local_dir, artifact_path=None):
        artifact_dir = build_path(self.artifact_uri, artifact_path) \
            if artifact_path else self.artifact_uri
        if not exists(artifact_dir):
            mkdir(artifact_dir)
        dir_util.copy_tree(src=local_dir, dst=artifact_dir)

    def list_artifacts(self, path=None):
        artifact_dir = self.artifact_uri
        list_dir = build_path(artifact_dir, path) if path else artifact_dir
        artifact_files = list_all(list_dir, full_path=True)
        infos = [get_file_info(f, get_relative_path(artifact_dir, f)) for f in artifact_files]
        return sorted(infos, key=lambda f: f.path)

    def download_artifacts(self, artifact_path):
        """Since this is a local file store, just return the artifacts' local path."""
        return build_path(self.artifact_uri, artifact_path)


class S3ArtifactRepository(ArtifactRepository):
    """Stores artifacts on Amazon S3."""

    @staticmethod
    def parse_s3_uri(uri):
        """Parse an S3 URI, returning (bucket, path)"""
        parsed = urllib.parse.urlparse(uri)
        if parsed.scheme != "s3":
            raise Exception("Not an S3 URI: %s" % uri)
        path = parsed.path
        if path.startswith('/'):
            path = path[1:]
        return parsed.netloc, path

    def log_artifact(self, local_file, artifact_path=None):
        (bucket, dest_path) = self.parse_s3_uri(self.artifact_uri)
        if artifact_path:
            dest_path = build_path(dest_path, artifact_path)
        dest_path = build_path(dest_path, os.path.basename(local_file))

        boto3.client('s3').upload_file(local_file, bucket, dest_path)

    def log_artifacts(self, local_dir, artifact_path=None):
        (bucket, dest_path) = self.parse_s3_uri(self.artifact_uri)
        if artifact_path:
            dest_path = build_path(dest_path, artifact_path)
        s3 = boto3.client('s3')
        local_dir = os.path.abspath(local_dir)
        for (root, _, filenames) in os.walk(local_dir):
            upload_path = dest_path
            if root != local_dir:
                rel_path = get_relative_path(local_dir, root)
                upload_path = build_path(dest_path, rel_path)
            for f in filenames:
                s3.upload_file(build_path(root, f), bucket, build_path(upload_path, f))

    def list_artifacts(self, path=None):
        (bucket, artifact_path) = self.parse_s3_uri(self.artifact_uri)
        dest_path = artifact_path
        if path:
            dest_path = build_path(dest_path, path)
        infos = []
        prefix = dest_path + "/"
        paginator = boto3.client('s3').get_paginator("list_objects_v2")
        results = paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter='/')
        for result in results:
            # Subdirectories will be listed as "common prefixes" due to the way we made the request
            for obj in result.get("CommonPrefixes", []):
                subdir = obj.get("Prefix")[len(artifact_path)+1:]
                if subdir.endswith("/"):
                    subdir = subdir[:-1]
                infos.append(FileInfo(subdir, True, None))
            # Objects listed directly will be files
            for obj in result.get('Contents', []):
                name = obj.get("Key")[len(artifact_path)+1:]
                size = int(obj.get('Size'))
                infos.append(FileInfo(name, False, size))
        return sorted(infos, key=lambda f: f.path)

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
            (bucket, s3_path) = self.parse_s3_uri(self.artifact_uri)
            s3_path = build_path(s3_path, artifact_path)
            boto3.client('s3').download_file(bucket, s3_path, local_path)
        return local_path


class GCSArtifactRepository(ArtifactRepository):
    """Stores artifacts on Google Cloud Storage.
       Assumes the google credentials are available in the environment,
       see https://google-cloud.readthedocs.io/en/latest/core/auth.html """

    def __init__(self, artifact_uri, client=gcs_storage):
        self.gcs = client
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

        infos = self._list_folders(bkt, prefix)

        results = bkt.list_blobs(prefix=prefix, delimiter="/")
        for result in results:
            blob_path = result.name[len(artifact_path)+1:]
            infos.append(FileInfo(blob_path, False, result.size))

        return sorted(infos, key=lambda f: f.path)

    def _list_folders(self, bkt, prefix):
        results = bkt.list_blobs(prefix=prefix, delimiter="/")
        dir_paths = set()
        for page in results.pages:
            dir_paths.update(page.prefixes)

        return [FileInfo(path[len(prefix):-1], True, None)for path in dir_paths]

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
