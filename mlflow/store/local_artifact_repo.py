import distutils.dir_util as dir_util
import shutil

from mlflow.store.artifact_repo import ArtifactRepository, verify_artifact_path
from mlflow.utils.file_utils import mkdir, list_all, get_file_info


class LocalArtifactRepository(ArtifactRepository):
    """Stores artifacts as files in a local directory."""

    def get_path_module(self):
        import os
        return os.path

    def log_artifact(self, local_file, artifact_path=None):

        verify_artifact_path(artifact_path)
        artifact_dir = self._resolve_path(artifact_path)

        if not self.get_path_module().exists(artifact_dir):
            mkdir(artifact_dir)
        shutil.copy(local_file, artifact_dir)

    def log_artifacts(self, local_dir, artifact_path=None):

        verify_artifact_path(artifact_path)
        artifact_dir = self._resolve_path(artifact_path)

        if not self.get_path_module().exists(artifact_dir):
            mkdir(artifact_dir)
        dir_util.copy_tree(src=local_dir, dst=artifact_dir)

    def list_artifacts(self, path=None):

        list_dir = self._resolve_path(path)

        artifact_dir = self.artifact_uri

        if self.get_path_module().isdir(list_dir):
            artifact_files = list_all(list_dir, full_path=True)
            infos = [get_file_info(f, self.get_path_module().relpath(f, artifact_dir))
                     for f in artifact_files]
            return sorted(infos, key=lambda f: f.path)
        else:
            return []

    def _download_file(self, remote_file_path, local_path):
        shutil.copyfile(
            self.get_path_module().join(self.artifact_uri, remote_file_path), local_path)

    def _resolve_path(self, artifact_path):
        artifact_dir = self.get_path_module().join(self.artifact_uri, artifact_path) \
            if artifact_path else self.artifact_uri
        return artifact_dir
