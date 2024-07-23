import os
import shutil
from typing import Any, Dict

from mlflow.store.artifact.artifact_repo import (
    ArtifactRepository,
    try_read_trace_data,
    verify_artifact_path,
)
from mlflow.tracing.artifact_utils import TRACE_DATA_FILE_NAME
from mlflow.utils.file_utils import (
    get_file_info,
    list_all,
    local_file_uri_to_path,
    mkdir,
    relative_path_to_artifact_path,
)
from mlflow.utils.uri import validate_path_is_safe


class LocalArtifactRepository(ArtifactRepository):
    """Stores artifacts as files in a local directory."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._artifact_dir = local_file_uri_to_path(self.artifact_uri)

    @property
    def artifact_dir(self):
        return self._artifact_dir

    def log_artifact(self, local_file, artifact_path=None):
        verify_artifact_path(artifact_path)
        # NOTE: The artifact_path is expected to be in posix format.
        # Posix paths work fine on windows but just in case we normalize it here.
        if artifact_path:
            artifact_path = os.path.normpath(artifact_path)

        artifact_dir = (
            os.path.join(self.artifact_dir, artifact_path) if artifact_path else self.artifact_dir
        )
        if not os.path.exists(artifact_dir):
            mkdir(artifact_dir)
        try:
            shutil.copy2(local_file, os.path.join(artifact_dir, os.path.basename(local_file)))
        except shutil.SameFileError:
            pass

    def _is_directory(self, artifact_path):
        # NOTE: The path is expected to be in posix format.
        # Posix paths work fine on windows but just in case we normalize it here.
        path = os.path.normpath(artifact_path) if artifact_path else ""
        list_dir = os.path.join(self.artifact_dir, path) if path else self.artifact_dir
        return os.path.isdir(list_dir)

    def log_artifacts(self, local_dir, artifact_path=None):
        verify_artifact_path(artifact_path)
        # NOTE: The artifact_path is expected to be in posix format.
        # Posix paths work fine on windows but just in case we normalize it here.
        if artifact_path:
            artifact_path = os.path.normpath(artifact_path)
        artifact_dir = (
            os.path.join(self.artifact_dir, artifact_path) if artifact_path else self.artifact_dir
        )
        if not os.path.exists(artifact_dir):
            mkdir(artifact_dir)
        shutil.copytree(src=local_dir, dst=artifact_dir, dirs_exist_ok=True)

    def download_artifacts(self, artifact_path, dst_path=None):
        """
        Artifacts tracked by ``LocalArtifactRepository`` already exist on the local filesystem.
        If ``dst_path`` is ``None``, the absolute filesystem path of the specified artifact is
        returned. If ``dst_path`` is not ``None``, the local artifact is copied to ``dst_path``.

        Args:
            artifact_path: Relative source path to the desired artifacts.
            dst_path: Absolute path of the local filesystem destination directory to which to
                download the specified artifacts. This directory must already exist. If
                unspecified, the absolute path of the local artifact will be returned.

        Returns:
            Absolute path of the local filesystem location containing the desired artifacts.
        """
        if dst_path:
            return super().download_artifacts(artifact_path, dst_path)
        # NOTE: The artifact_path is expected to be a relative path in posix format.
        # Posix paths work fine on windows but just in case we normalize it here.
        artifact_path = validate_path_is_safe(artifact_path)
        local_artifact_path = os.path.join(self.artifact_dir, os.path.normpath(artifact_path))
        if not os.path.exists(local_artifact_path):
            raise OSError(f"No such file or directory: '{local_artifact_path}'")
        return os.path.abspath(local_artifact_path)

    def list_artifacts(self, path=None):
        # NOTE: The path is expected to be in posix format.
        # Posix paths work fine on windows but just in case we normalize it here.
        if path:
            path = os.path.normpath(path)
        list_dir = os.path.join(self.artifact_dir, path) if path else self.artifact_dir
        if os.path.isdir(list_dir):
            artifact_files = list_all(list_dir, full_path=True)
            infos = [
                get_file_info(
                    f, relative_path_to_artifact_path(os.path.relpath(f, self.artifact_dir))
                )
                for f in artifact_files
            ]
            return sorted(infos, key=lambda f: f.path)
        else:
            return []

    def _download_file(self, remote_file_path, local_path):
        # NOTE: The remote_file_path is expected to be a relative path in posix format.
        # Posix paths work fine on windows but just in case we normalize it here.
        remote_file_path = validate_path_is_safe(remote_file_path)
        remote_file_path = os.path.join(self.artifact_dir, os.path.normpath(remote_file_path))
        shutil.copy2(remote_file_path, local_path)

    def delete_artifacts(self, artifact_path=None):
        artifact_path = local_file_uri_to_path(
            os.path.join(self._artifact_dir, artifact_path) if artifact_path else self._artifact_dir
        )

        if os.path.exists(artifact_path):
            shutil.rmtree(artifact_path)

    def download_trace_data(self) -> Dict[str, Any]:
        """
        Download the trace data.

        Returns:
            The trace data as a dictionary.

        Raises:
            - `MlflowTraceDataNotFound`: The trace data is not found.
            - `MlflowTraceDataCorrupted`: The trace data is corrupted.
        """
        trace_data_path = os.path.join(self.artifact_dir, TRACE_DATA_FILE_NAME)
        return try_read_trace_data(trace_data_path)
