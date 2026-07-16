import os
import shutil
import tempfile
import threading
from contextlib import suppress
from typing import Any, BinaryIO, Callable

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.store.artifact.artifact_repo import (
    ARTIFACT_STREAM_CHUNK_SIZE,
    ArtifactRepository,
    StreamUploadMixin,
    try_read_trace_data,
    verify_artifact_path,
)
from mlflow.tracing.utils.artifact_utils import TRACE_DATA_FILE_NAME
from mlflow.utils.file_utils import (
    get_file_info,
    list_all,
    local_file_uri_to_path,
    mkdir,
    relative_path_to_artifact_path,
    shutil_copytree_without_file_permissions,
)
from mlflow.utils.uri import validate_path_is_safe, validate_path_within_directory

_TEMP_ARTIFACT_PREFIX = ".artifact.uploading."
_UMASK_LOCK = threading.Lock()


def _set_default_file_mode(file_path: str) -> None:
    """Set file permissions to match the default mode produced by ``open()``."""
    if os.name == "nt":
        return

    with _UMASK_LOCK:
        current_umask = os.umask(0)
        os.umask(current_umask)
    os.chmod(file_path, 0o666 & ~current_umask)


def _verify_artifact_file_name(artifact_file_name: str) -> None:
    if os.path.basename(artifact_file_name).startswith(_TEMP_ARTIFACT_PREFIX):
        raise MlflowException.invalid_parameter_value(
            f"Invalid artifact file name: '{artifact_file_name}'. "
            f"Artifact names starting with '{_TEMP_ARTIFACT_PREFIX}' are reserved."
        )


class LocalArtifactRepository(ArtifactRepository, StreamUploadMixin):
    """Stores artifacts as files in a local directory."""

    def __init__(
        self, artifact_uri: str, tracking_uri: str | None = None, registry_uri: str | None = None
    ) -> None:
        super().__init__(artifact_uri, tracking_uri, registry_uri)
        self._artifact_dir = local_file_uri_to_path(self.artifact_uri)

    @property
    def artifact_dir(self):
        return self._artifact_dir

    def get_local_path(self, artifact_path: str) -> str:
        artifact_path = validate_path_is_safe(artifact_path)
        local_artifact_path = os.path.join(self.artifact_dir, os.path.normpath(artifact_path))
        validate_path_within_directory(self.artifact_dir, local_artifact_path)
        if not os.path.exists(local_artifact_path):
            raise MlflowException(
                f"No such artifact: '{artifact_path}'",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        return os.path.abspath(local_artifact_path)

    def _get_or_create_artifact_dir(self, artifact_path: str | None = None) -> str:
        verify_artifact_path(artifact_path)
        # NOTE: The artifact_path is expected to be in posix format.
        # Posix paths work fine on windows but just in case we normalize it here.
        if artifact_path:
            artifact_path = os.path.normpath(artifact_path)

        artifact_dir = (
            os.path.join(self.artifact_dir, artifact_path) if artifact_path else self.artifact_dir
        )
        validate_path_within_directory(self.artifact_dir, artifact_dir)
        if not os.path.exists(artifact_dir):
            mkdir(artifact_dir)
        return artifact_dir

    def _get_destination_artifact_path(
        self, artifact_file_name: str, artifact_path: str | None = None
    ) -> tuple[str, str]:
        artifact_dir = self._get_or_create_artifact_dir(artifact_path)
        _verify_artifact_file_name(artifact_file_name)
        destination_file_path = os.path.join(artifact_dir, os.path.basename(artifact_file_name))
        validate_path_within_directory(self.artifact_dir, destination_file_path)
        return artifact_dir, destination_file_path

    def _write_to_destination_path(
        self,
        artifact_file_name: str,
        writer: Callable[[BinaryIO], None],
        artifact_path: str | None = None,
        finalize_temp_path: Callable[[str], None] | None = None,
    ) -> None:
        artifact_dir, destination_file_path = self._get_destination_artifact_path(
            artifact_file_name, artifact_path
        )
        # Write to a hidden temp file in the destination directory, then atomically
        # replace the target path so readers never see a partially-written artifact.
        temp_file_descriptor, temp_file_path = tempfile.mkstemp(
            dir=artifact_dir, prefix=_TEMP_ARTIFACT_PREFIX
        )
        published = False
        try:
            temp_file = os.fdopen(temp_file_descriptor, "wb")
            temp_file_descriptor = None
            with temp_file:
                writer(temp_file)
            if finalize_temp_path is not None:
                finalize_temp_path(temp_file_path)
            os.replace(temp_file_path, destination_file_path)
            published = True
        finally:
            if temp_file_descriptor is not None:
                os.close(temp_file_descriptor)
            if not published:
                with suppress(FileNotFoundError):
                    os.remove(temp_file_path)

    def log_artifact(self, local_file, artifact_path=None):
        _, destination_file_path = self._get_destination_artifact_path(local_file, artifact_path)
        try:
            if os.path.samefile(local_file, destination_file_path):
                return
        except FileNotFoundError:
            pass

        def _write_file(temp_file: BinaryIO) -> None:
            with open(local_file, "rb") as local_artifact_file:
                shutil.copyfileobj(
                    local_artifact_file, temp_file, length=ARTIFACT_STREAM_CHUNK_SIZE
                )

        self._write_to_destination_path(
            local_file,
            _write_file,
            artifact_path,
            finalize_temp_path=lambda temp_file_path: shutil.copystat(local_file, temp_file_path),
        )

    def log_artifact_from_stream(
        self,
        stream: BinaryIO,
        artifact_file_name: str,
        artifact_path: str | None = None,
    ) -> None:
        def _write_stream(temp_file: BinaryIO) -> None:
            shutil.copyfileobj(stream, temp_file, length=ARTIFACT_STREAM_CHUNK_SIZE)

        self._write_to_destination_path(
            artifact_file_name,
            _write_stream,
            artifact_path,
            finalize_temp_path=_set_default_file_mode,
        )

    def _is_directory(self, artifact_path):
        # NOTE: The path is expected to be in posix format.
        # Posix paths work fine on windows but just in case we normalize it here.
        path = os.path.normpath(artifact_path) if artifact_path else ""
        list_dir = os.path.join(self.artifact_dir, path) if path else self.artifact_dir
        return os.path.isdir(list_dir)

    def log_artifacts(self, local_dir, artifact_path=None):
        artifact_dir = self._get_or_create_artifact_dir(artifact_path)
        shutil_copytree_without_file_permissions(local_dir, artifact_dir)

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
        return self.get_local_path(artifact_path)

    def list_artifacts(self, path=None):
        if path:
            path = os.path.normpath(path)
        list_dir = os.path.join(self.artifact_dir, path) if path else self.artifact_dir
        validate_path_within_directory(self.artifact_dir, list_dir)
        if os.path.isdir(list_dir):
            artifact_files = list_all(list_dir, full_path=True)
            infos = [
                get_file_info(
                    f, relative_path_to_artifact_path(os.path.relpath(f, self.artifact_dir))
                )
                for f in artifact_files
                if not os.path.basename(f).startswith(_TEMP_ARTIFACT_PREFIX)
            ]
            return sorted(infos, key=lambda f: f.path)
        else:
            return []

    def _download_file(self, remote_file_path, local_path):
        remote_file_path = validate_path_is_safe(remote_file_path)
        remote_file_path = os.path.join(self.artifact_dir, os.path.normpath(remote_file_path))
        validate_path_within_directory(self.artifact_dir, remote_file_path)
        if not os.path.exists(remote_file_path):
            raise MlflowException(
                f"No such artifact: '{os.path.relpath(remote_file_path, self.artifact_dir)}'",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )
        shutil.copy2(remote_file_path, local_path)

    def delete_artifacts(self, artifact_path=None):
        artifact_path = local_file_uri_to_path(
            os.path.join(self._artifact_dir, artifact_path) if artifact_path else self._artifact_dir
        )

        if os.path.exists(artifact_path):
            if os.path.isfile(artifact_path):
                os.remove(artifact_path)
            else:
                shutil.rmtree(artifact_path)

    def download_trace_data(self) -> dict[str, Any]:
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
