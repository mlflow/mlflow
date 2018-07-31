import distutils.dir_util as dir_util
import shutil

from mlflow.store.artifact_repo import ArtifactRepository
from mlflow.utils.file_utils import (build_path, exists, mkdir, list_all, get_file_info,
                                     get_relative_path)


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
