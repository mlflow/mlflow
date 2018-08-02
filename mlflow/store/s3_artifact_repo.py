import os

import boto3
from six.moves import urllib

from mlflow.entities.file_info import FileInfo
from mlflow.store.artifact_repo import ArtifactRepository
from mlflow.utils.file_utils import build_path, get_relative_path, TempDir


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
