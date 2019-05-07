import distutils.dir_util as dir_util
import os
import shutil

from mlflow.store.artifact_repo import ArtifactRepository, verify_artifact_path
from mlflow.utils.file_utils import mkdir, list_all, get_file_info, local_file_uri_to_path, \
    relative_path_to_artifact_path


class LocalArtifactRepository(ArtifactRepository):
    """Stores artifacts as files in a local directory."""

    def __init__(self, *args, **kwargs):
        super(LocalArtifactRepository, self).__init__(*args, **kwargs)
        self.artifact_dir = local_file_uri_to_path(self.artifact_uri)

    def log_artifact(self, local_file, artifact_path=None):
        verify_artifact_path(artifact_path)
        # NOTE: The artifact_path is expected to be in posix format.
        # Posix paths work fine on windows but just in case we normalize it here.
        if artifact_path:
            artifact_path = os.path.normpath(artifact_path)

        artifact_dir = os.path.join(self.artifact_dir, artifact_path) if artifact_path else \
            self.artifact_dir
        if not os.path.exists(artifact_dir):
            mkdir(artifact_dir)
        shutil.copy(local_file, artifact_dir)

    def log_artifacts(self, local_dir, artifact_path=None):
        verify_artifact_path(artifact_path)
        # NOTE: The artifact_path is expected to be in posix format.
        # Posix paths work fine on windows but just in case we normalize it here.
        if artifact_path:
            artifact_path = os.path.normpath(artifact_path)
        artifact_dir = os.path.join(self.artifact_dir, artifact_path) if artifact_path else \
            self.artifact_dir
        if not os.path.exists(artifact_dir):
            mkdir(artifact_dir)
        dir_util.copy_tree(src=local_dir, dst=artifact_dir)

    def list_artifacts(self, path=None):
        # NOTE: The path is expected to be in posix format.
        # Posix paths work fine on windows but just in case we normalize it here.
        if path:
            path = os.path.normpath(path)
        list_dir = os.path.join(self.artifact_dir, path) if path else self.artifact_dir
        if os.path.isdir(list_dir):
            artifact_files = list_all(list_dir, full_path=True)
            infos = [get_file_info(f,
                                   relative_path_to_artifact_path(
                                       os.path.relpath(f, self.artifact_dir)))
                     for f in artifact_files]
            return sorted(infos, key=lambda f: f.path)
        else:
            return []

    def _download_file(self, remote_file_path, local_path):
        # NOTE: The remote_file_path is expected to be in posix format.
        # Posix paths work fine on windows but just in case we normalize it here.
        remote_file_path = os.path.join(self.artifact_dir, os.path.normpath(remote_file_path))
        shutil.copyfile(remote_file_path, local_path)
