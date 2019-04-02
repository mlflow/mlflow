import distutils.dir_util as dir_util
import shutil

import mlflow.tracking

from mlflow.store.artifact_repo import ArtifactRepository
from mlflow.utils.file_utils import mkdir, list_all, get_file_info, parse_path
from mlflow.utils.validation import path_not_unique, bad_path_message


class LocalArtifactRepository(ArtifactRepository):
    """Stores artifacts as files in a local directory."""
    def __init__(self, *args, **kwargs):
        super(LocalArtifactRepository, self).__init__(*args, **kwargs)

        if mlflow.tracking.utils._is_local_uri(self.artifact_uri):
            mkdir(parse_path(self.artifact_location))

    @property
    def artifact_location(self):
        return parse_path(self.artifact_uri)

    def get_path_module(self):
        import os
        return os.path

    def log_artifact(self, local_file, artifact_path=None):
        if artifact_path and path_not_unique(artifact_path):
            raise Exception("Invalid artifact path: '%s'. %s" % (artifact_path,
                                                                 bad_path_message(artifact_path)))
        artifact_location = self.get_path_module().join(self.artifact_location, artifact_path) \
            if artifact_path else self.artifact_location

        if not self.get_path_module().exists(artifact_location):
            mkdir(artifact_location)
        shutil.copy(local_file, artifact_location)

    def log_artifacts(self, local_dir, artifact_path=None):
        if artifact_path and path_not_unique(artifact_path):
            raise Exception("Invalid artifact path: '%s'. %s" % (artifact_path,
                                                                 bad_path_message(artifact_path)))
        artifact_location = self.get_path_module().join(self.artifact_location, artifact_path) \
            if artifact_path else self.artifact_location
        if not self.get_path_module().exists(artifact_location):
            mkdir(artifact_location)
        dir_util.copy_tree(src=local_dir, dst=artifact_location)

    def list_artifacts(self, path=None):
        list_dir = self.get_path_module().join(self.artifact_location, path) \
            if path else self.artifact_location
        if self.get_path_module().isdir(list_dir):
            artifact_files = list_all(list_dir, full_path=True)
            infos = [get_file_info(f, self.get_path_module().relpath(f, self.artifact_location))
                     for f in artifact_files]
            return sorted(infos, key=lambda f: f.path)
        else:
            return []

    def _download_file(self, remote_file_path, local_path):
        shutil.copyfile(
            self.get_path_module().join(self.artifact_location, remote_file_path), local_path)
