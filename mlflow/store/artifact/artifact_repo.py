import json
import logging
import os
import posixpath
import tempfile
import traceback
from abc import ABC, ABCMeta, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

from mlflow.entities.file_info import FileInfo
from mlflow.entities.multipart_upload import (
    CreateMultipartUploadResponse,
    MultipartUploadPart,
)
from mlflow.exceptions import (
    MlflowException,
    MlflowTraceDataCorrupted,
    MlflowTraceDataNotFound,
)
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.tracing.utils.artifact_utils import TRACE_DATA_FILE_NAME
from mlflow.utils.annotations import developer_stable
from mlflow.utils.async_logging.async_artifacts_logging_queue import (
    AsyncArtifactsLoggingQueue,
)
from mlflow.utils.file_utils import ArtifactProgressBar, create_tmp_dir
from mlflow.utils.validation import bad_path_message, path_not_unique

# Constants used to determine max level of parallelism to use while uploading/downloading artifacts.
# Max threads to use for parallelism.
_NUM_MAX_THREADS = 20
# Max threads per CPU
_NUM_MAX_THREADS_PER_CPU = 2
assert _NUM_MAX_THREADS >= _NUM_MAX_THREADS_PER_CPU
assert _NUM_MAX_THREADS_PER_CPU > 0
# Default number of CPUs to assume on the machine if unavailable to fetch it using os.cpu_count()
_NUM_DEFAULT_CPUS = _NUM_MAX_THREADS // _NUM_MAX_THREADS_PER_CPU
_logger = logging.getLogger(__name__)


def _truncate_error(err: str, max_length: int = 10_000) -> str:
    if len(err) <= max_length:
        return err
    half = max_length // 2
    return err[:half] + "\n\n*** Error message is too long, truncated ***\n\n" + err[-half:]


def _retry_with_new_creds(try_func, creds_func, orig_creds=None):
    """
    Attempt the try_func with the original credentials (og_creds) if provided, or by generating the
    credentials using creds_func. If the try_func throws, then try again with new credentials
    provided by creds_func.
    """
    try:
        first_creds = creds_func() if orig_creds is None else orig_creds
        return try_func(first_creds)
    except Exception as e:
        _logger.info(
            f"Failed to complete request, possibly due to credential expiration (Error: {e})."
            " Refreshing credentials and trying again..."
        )
        new_creds = creds_func()
        return try_func(new_creds)


@developer_stable
class ArtifactRepository:
    """
    Abstract artifact repo that defines how to upload (log) and download potentially large
    artifacts from different storage backends.
    """

    __metaclass__ = ABCMeta

    def __init__(self, artifact_uri: str, tracking_uri: Optional[str] = None) -> None:
        self.artifact_uri = artifact_uri
        self.tracking_uri = tracking_uri
        # Limit the number of threads used for artifact uploads/downloads. Use at most
        # constants._NUM_MAX_THREADS threads or 2 * the number of CPU cores available on the
        # system (whichever is smaller)
        self.thread_pool = self._create_thread_pool()

        def log_artifact_handler(filename, artifact_path=None, artifact=None):
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = os.path.join(tmp_dir, filename)
                if artifact is not None:
                    # User should already have installed PIL to log a PIL image
                    from PIL import Image

                    if isinstance(artifact, Image.Image):
                        artifact.save(tmp_path)
                self.log_artifact(tmp_path, artifact_path)

        self._async_logging_queue = AsyncArtifactsLoggingQueue(log_artifact_handler)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"artifact_uri={self.artifact_uri!r}, "
            f"tracking_uri={self.tracking_uri!r}"
            f")"
        )

    def _create_thread_pool(self):
        return ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix=f"Mlflow{self.__class__.__name__}"
        )

    def flush_async_logging(self):
        """
        Flushes the async logging queue, ensuring that all pending logging operations have
        completed.
        """
        if self._async_logging_queue._is_activated:
            self._async_logging_queue.flush()

    @abstractmethod
    def log_artifact(self, local_file, artifact_path=None):
        """
        Log a local file as an artifact, optionally taking an ``artifact_path`` to place it in
        within the run's artifacts. Run artifacts can be organized into directories, so you can
        place the artifact in a directory this way.

        Args:
            local_file: Path to artifact to log.
            artifact_path: Directory within the run's artifact directory in which to log the
                artifact.
        """

    def _log_artifact_async(self, filename, artifact_path=None, artifact=None):
        """
        Asynchronously log a local file as an artifact, optionally taking an ``artifact_path`` to
        place it within the run's artifacts. Run artifacts can be organized into directory, so you
        can place the artifact in the directory this way. Cleanup tells the function whether to
        cleanup the local_file after running log_artifact, since it could be a Temporary
        Directory.

        Args:
            filename: Filename of the artifact to be logged.
            artifact_path: Directory within the run's artifact directory in which to log the
                artifact.
            artifact: The artifact to be logged.

        Returns:
            An :py:class:`mlflow.utils.async_logging.run_operations.RunOperations` instance
            that represents future for logging operation.
        """

        if not self._async_logging_queue.is_active():
            self._async_logging_queue.activate()

        return self._async_logging_queue.log_artifacts_async(
            filename=filename, artifact_path=artifact_path, artifact=artifact
        )

    @abstractmethod
    def log_artifacts(self, local_dir, artifact_path=None):
        """
        Log the files in the specified local directory as artifacts, optionally taking
        an ``artifact_path`` to place them in within the run's artifacts.

        Args:
            local_dir: Directory of local artifacts to log.
            artifact_path: Directory within the run's artifact directory in which to log the
                artifacts.
        """

    @abstractmethod
    def list_artifacts(self, path: Optional[str] = None) -> list[FileInfo]:
        """
        Return all the artifacts for this run_id directly under path. If path is a file, returns
        an empty list. Will error if path is neither a file nor directory.

        Args:
            path: Relative source path that contains desired artifacts.

        Returns:
            List of artifacts as FileInfo listed directly under path.
        """

    def _is_directory(self, artifact_path):
        listing = self.list_artifacts(artifact_path)
        return len(listing) > 0

    def _create_download_destination(self, src_artifact_path, dst_local_dir_path=None):
        """
        Creates a local filesystem location to be used as a destination for downloading the artifact
        specified by `src_artifact_path`. The destination location is a subdirectory of the
        specified `dst_local_dir_path`, which is determined according to the structure of
        `src_artifact_path`. For example, if `src_artifact_path` is `dir1/file1.txt`, then the
        resulting destination path is `<dst_local_dir_path>/dir1/file1.txt`. Local directories are
        created for the resulting destination location if they do not exist.

        Args:
            src_artifact_path: A relative, POSIX-style path referring to an artifact stored
                within the repository's artifact root location. `src_artifact_path` should be
                specified relative to the repository's artifact root location.
            dst_local_dir_path: The absolute path to a local filesystem directory in which the
                local destination path will be contained. The local destination path may be
                contained in a subdirectory of `dst_root_dir` if `src_artifact_path` contains
                subdirectories.

        Returns:
            The absolute path to a local filesystem location to be used as a destination
            for downloading the artifact specified by `src_artifact_path`.
        """
        src_artifact_path = src_artifact_path.rstrip("/")  # Ensure correct dirname for trailing '/'
        dirpath = posixpath.dirname(src_artifact_path)
        local_dir_path = os.path.join(dst_local_dir_path, dirpath)
        local_file_path = os.path.join(dst_local_dir_path, src_artifact_path)
        if not os.path.exists(local_dir_path):
            os.makedirs(local_dir_path, exist_ok=True)
        return local_file_path

    def _iter_artifacts_recursive(self, path):
        dir_content = [
            file_info
            for file_info in self.list_artifacts(path)
            # prevent infinite loop, sometimes the dir is recursively included
            if file_info.path not in [".", path]
        ]
        # Empty directory
        if not dir_content:
            yield FileInfo(path=path, is_dir=True, file_size=None)
            return

        for file_info in dir_content:
            if file_info.is_dir:
                yield from self._iter_artifacts_recursive(file_info.path)
            else:
                yield file_info

    def download_artifacts(self, artifact_path, dst_path=None):
        """
        Download an artifact file or directory to a local directory if applicable, and return a
        local path for it.
        The caller is responsible for managing the lifecycle of the downloaded artifacts.

        Args:
            artifact_path: Relative source path to the desired artifacts.
            dst_path: Absolute path of the local filesystem destination directory to which to
                download the specified artifacts. This directory must already exist.
                If unspecified, the artifacts will either be downloaded to a new
                uniquely-named directory on the local filesystem or will be returned
                directly in the case of the LocalArtifactRepository.

        Returns:
            Absolute path of the local filesystem location containing the desired artifacts.
        """
        if dst_path:
            dst_path = os.path.abspath(dst_path)
            if not os.path.exists(dst_path):
                raise MlflowException(
                    message=(
                        "The destination path for downloaded artifacts does not"
                        f" exist! Destination path: {dst_path}"
                    ),
                    error_code=RESOURCE_DOES_NOT_EXIST,
                )
            elif not os.path.isdir(dst_path):
                raise MlflowException(
                    message=(
                        "The destination path for downloaded artifacts must be a directory!"
                        f" Destination path: {dst_path}"
                    ),
                    error_code=INVALID_PARAMETER_VALUE,
                )
        else:
            dst_path = create_tmp_dir()

        def _download_file(src_artifact_path, dst_local_dir_path):
            dst_local_file_path = self._create_download_destination(
                src_artifact_path=src_artifact_path, dst_local_dir_path=dst_local_dir_path
            )
            return self.thread_pool.submit(
                self._download_file,
                remote_file_path=src_artifact_path,
                local_path=dst_local_file_path,
            )

        # Submit download tasks
        futures = {}
        if self._is_directory(artifact_path):
            for file_info in self._iter_artifacts_recursive(artifact_path):
                if file_info.is_dir:  # Empty directory
                    os.makedirs(os.path.join(dst_path, file_info.path), exist_ok=True)
                else:
                    fut = _download_file(file_info.path, dst_path)
                    futures[fut] = file_info.path
        else:
            fut = _download_file(artifact_path, dst_path)
            futures[fut] = artifact_path

        # Wait for downloads to complete and collect failures
        failed_downloads = {}
        tracebacks = {}
        with ArtifactProgressBar.files(desc="Downloading artifacts", total=len(futures)) as pbar:
            for f in as_completed(futures):
                try:
                    f.result()
                    pbar.update()
                except Exception as e:
                    path = futures[f]
                    failed_downloads[path] = e
                    tracebacks[path] = traceback.format_exc()

        if failed_downloads:
            if _logger.isEnabledFor(logging.DEBUG):
                template = "##### File {path} #####\n{error}\nTraceback:\n{traceback}\n"
            else:
                template = "##### File {path} #####\n{error}"

            failures = "\n".join(
                template.format(path=path, error=error, traceback=tracebacks[path])
                for path, error in failed_downloads.items()
            )
            raise MlflowException(
                message=(
                    "The following failures occurred while downloading one or more"
                    f" artifacts from {self.artifact_uri}:\n{_truncate_error(failures)}"
                )
            )

        return os.path.join(dst_path, artifact_path)

    @abstractmethod
    def _download_file(self, remote_file_path, local_path):
        """
        Download the file at the specified relative remote path and saves
        it at the specified local path.

        Args:
            remote_file_path: Source path to the remote file, relative to the root
                directory of the artifact repository.
            local_path: The path to which to save the downloaded file.
        """

    def delete_artifacts(self, artifact_path=None):
        """
        Delete the artifacts at the specified location.
        Supports the deletion of a single file or of a directory. Deletion of a directory
        is recursive.

        Args:
            artifact_path: Path of the artifact to delete.
        """

    @property
    def max_workers(self) -> int:
        """Compute the number of workers to use for multi-threading."""
        num_cpus = os.cpu_count() or _NUM_DEFAULT_CPUS
        return min(num_cpus * _NUM_MAX_THREADS_PER_CPU, _NUM_MAX_THREADS)

    def download_trace_data(self) -> dict[str, Any]:
        """
        Download the trace data.

        Returns:
            The trace data as a dictionary.

        Raises:
            - `MlflowTraceDataNotFound`: The trace data is not found.
            - `MlflowTraceDataCorrupted`: The trace data is corrupted.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir, TRACE_DATA_FILE_NAME)
            try:
                self._download_file(TRACE_DATA_FILE_NAME, temp_file)
            except Exception as e:
                # `MlflowTraceDataNotFound` is caught in `TrackingServiceClient.search_traces` and
                # is used to filter out traces with failed trace data download.
                raise MlflowTraceDataNotFound(artifact_path=TRACE_DATA_FILE_NAME) from e
            return try_read_trace_data(temp_file)

    def upload_trace_data(self, trace_data: str) -> None:
        """
        Upload the trace data.

        Args:
            trace_data: The json-serialized trace data to upload.
        """
        with write_local_temp_trace_data_file(trace_data) as temp_file:
            self.log_artifact(temp_file)


@contextmanager
def write_local_temp_trace_data_file(trace_data: str):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir, TRACE_DATA_FILE_NAME)
        temp_file.write_text(trace_data, encoding="utf-8")
        yield temp_file


def try_read_trace_data(trace_data_path):
    if not os.path.exists(trace_data_path):
        raise MlflowTraceDataNotFound(artifact_path=trace_data_path)
    with open(trace_data_path, encoding="utf-8") as f:
        data = f.read()
    if not data:
        raise MlflowTraceDataNotFound(artifact_path=trace_data_path)
    try:
        return json.loads(data)
    except json.decoder.JSONDecodeError as e:
        raise MlflowTraceDataCorrupted(artifact_path=trace_data_path) from e


class MultipartUploadMixin(ABC):
    @abstractmethod
    def create_multipart_upload(
        self, local_file: str, num_parts: int, artifact_path: Optional[str] = None
    ) -> CreateMultipartUploadResponse:
        """
        Initiate a multipart upload and retrieve the pre-signed upload URLS and upload id.

        Args:
            local_file: Path of artifact to upload.
            num_parts: Number of parts to upload. Only required by S3 and GCS.
            artifact_path: Directory within the run's artifact directory in which to upload the
                artifact.

        """

    @abstractmethod
    def complete_multipart_upload(
        self,
        local_file: str,
        upload_id: str,
        parts: list[MultipartUploadPart],
        artifact_path: Optional[str] = None,
    ) -> None:
        """
        Complete a multipart upload.

        Args:
            local_file: Path of artifact to upload.
            upload_id: The upload ID. Only required by S3 and GCS.
            parts: A list containing the metadata of each part that has been uploaded.
            artifact_path: Directory within the run's artifact directory in which to upload the
                artifact.

        """

    @abstractmethod
    def abort_multipart_upload(
        self,
        local_file: str,
        upload_id: str,
        artifact_path: Optional[str] = None,
    ) -> None:
        """
        Abort a multipart upload.

        Args:
            local_file: Path of artifact to upload.
            upload_id: The upload ID. Only required by S3 and GCS.
            artifact_path: Directory within the run's artifact directory in which to upload the
                artifact.

        """


def verify_artifact_path(artifact_path):
    if artifact_path and path_not_unique(artifact_path):
        raise MlflowException(
            f"Invalid artifact path: '{artifact_path}'. {bad_path_message(artifact_path)}"
        )
