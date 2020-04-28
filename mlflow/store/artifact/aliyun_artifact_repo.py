import os

import posixpath
from six.moves import urllib

from mlflow import data
from mlflow.entities import FileInfo
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.utils.file_utils import relative_path_to_artifact_path


class AliyunArtifactRepository(ArtifactRepository):
    """Stores artifacts on Aliyun OSS."""

    def __init__(self, artifact_uri, oss_bucket=None):
        super(AliyunArtifactRepository, self).__init__(artifact_uri)

        if oss_bucket is not None:
            self.oss_bucket = oss_bucket
            return

        import oss2
        self.oss_endpoint_url = os.environ.get('MLFLOW_OSS_ENDPOINT_URL')
        oss_access_key_id = os.environ.get('MLFLOW_OSS_KEY_ID')
        oss_access_key_secret = os.environ.get('MLFLOW_OSS_KEY_SECRET')
        self.auth = oss2.Auth(oss_access_key_id, oss_access_key_secret)
        self.oss_bucket = None


    @staticmethod
    def parse_oss_uri(uri):
        """Parse an OSS URI, returning (bucket, path)"""
        parsed = urllib.parse.urlparse(uri)
        if parsed.scheme != "oss":
            raise Exception("Not an OSS URI: %s" % uri)
        path = parsed.path
        if path.startswith('/'):
            path = path[1:]
        return parsed.netloc, path


    def _get_oss_bucket(self, bucket):
        import oss2
        if self.oss_bucket is not None: return self.oss_bucket
        return oss2.Bucket(self.auth, self.oss_endpoint_url, bucket)
 

    def log_artifact(self, local_file, artifact_path=None):
        (bucket, dest_path) = self.parse_oss_uri(self.artifact_uri)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        dest_path = posixpath.join(
            dest_path, os.path.basename(local_file))
        self._get_oss_bucket(bucket)
        self.oss_bucket.put_object_from_file(dest_path, local_file)
        

    def log_artifacts(self, local_dir, artifact_path=None):
        (bucket, dest_path) = self.parse_oss_uri(self.artifact_uri)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        oss_bucket = self._get_oss_bucket(bucket)
        local_dir = os.path.abspath(local_dir)
        for (root, _, filenames) in os.walk(local_dir):
            upload_path = dest_path
            if root != local_dir:
                rel_path = os.path.relpath(root, local_dir)
                rel_path = relative_path_to_artifact_path(rel_path)
                upload_path = posixpath.join(dest_path, rel_path)
            for f in filenames:
                self.oss_bucket.put_object_from_file(posixpath.join(upload_path, f), os.path.join(root, f))
    
    def list_artifacts(self, path=None):
        import oss2
        (bucket, artifact_path) = self.parse_oss_uri(self.artifact_uri)
        dest_path = artifact_path
        if path:
            dest_path = posixpath.join(dest_path, path)
        infos = []
        prefix = dest_path + "/" if dest_path else ""
        oss_bucket = self._get_oss_bucket(bucket)
        results = self.oss_bucket.list_objects(prefix=prefix, delimiter = '/').object_list

        for result in results:
            if result.is_prefix():
                subdir_path = result.key
                self._verify_listed_object_contains_artifact_path_prefix(
                    listed_object_path=subdir_path, artifact_path=artifact_path)
                subdir_rel_path = posixpath.relpath(path=subdir_path, start=artifact_path)
                infos.append(FileInfo(subdir_rel_path, True, None))
            else:
                file_path = result.key
                self._verify_listed_object_contains_artifact_path_prefix(
                    listed_object_path=file_path, artifact_path=artifact_path)
                file_rel_path = posixpath.relpath(path=file_path, start=artifact_path)
                file_size = result.size
                infos.append(FileInfo(file_rel_path, False, file_size))

        return sorted(infos, key=lambda f: f.path)

    @staticmethod
    def _verify_listed_object_contains_artifact_path_prefix(listed_object_path, artifact_path):
        if not listed_object_path.startswith(artifact_path):
            raise MlflowException(
                "The path of the listed oss object does not begin with the specified"
                " artifact path. Artifact path: {artifact_path}. Object path:"
                " {object_path}.".format(
                    artifact_path=artifact_path, object_path=listed_object_path))

    def _download_file(self, remote_file_path, local_path):
        (bucket, oss_root_path) = self.parse_oss_uri(self.artifact_uri)
        oss_full_path = posixpath.join(oss_root_path, remote_file_path)
        self._get_oss_bucket(bucket)
        self.oss_bucket.get_object_to_file(oss_full_path, local_path)

    def delete_artifacts(self, artifact_path=None):
        raise MlflowException('Not implemented yet')
