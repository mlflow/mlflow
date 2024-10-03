import logging
import math
import os
import posixpath
import time
from abc import abstractmethod
from collections import namedtuple
from concurrent.futures import as_completed

from mlflow.environment_variables import (
    _MLFLOW_MPD_NUM_RETRIES,
    _MLFLOW_MPD_RETRY_INTERVAL_SECONDS,
    MLFLOW_ENABLE_MULTIPART_DOWNLOAD,
    MLFLOW_MULTIPART_DOWNLOAD_CHUNK_SIZE,
    MLFLOW_MULTIPART_DOWNLOAD_MINIMUM_FILE_SIZE,
    MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE,
)
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.utils import chunk_list
from mlflow.utils.file_utils import (
    ArtifactProgressBar,
    parallelized_download_file_using_http_uri,
    relative_path_to_artifact_path,
    remove_on_error,
)
from mlflow.utils.request_utils import download_chunk
from mlflow.utils.uri import is_fuse_or_uc_volumes_uri

_logger = logging.getLogger(__name__)
_ARTIFACT_UPLOAD_BATCH_SIZE = (
    50  # Max number of artifacts for which to fetch write credentials at once.
)
_AWS_MIN_CHUNK_SIZE = 5 * 1024**2  # 5 MB is the minimum chunk size for S3 multipart uploads
_AWS_MAX_CHUNK_SIZE = 5 * 1024**3  # 5 GB is the maximum chunk size for S3 multipart uploads


def _readable_size(size: int) -> str:
    return f"{size / 1024**2:.2f} MB"


def _validate_chunk_size_aws(chunk_size: int) -> None:
    """
    Validates the specified chunk size in bytes is in valid range for AWS multipart uploads.
    """
    if chunk_size < _AWS_MIN_CHUNK_SIZE or chunk_size > _AWS_MAX_CHUNK_SIZE:
        raise MlflowException(
            message=(
                f"Multipart chunk size {_readable_size(chunk_size)} must be in range: "
                f"{_readable_size(_AWS_MIN_CHUNK_SIZE)} to {_readable_size(_AWS_MAX_CHUNK_SIZE)}."
            )
        )


def _compute_num_chunks(local_file: os.PathLike, chunk_size: int) -> int:
    """
    Computes the number of chunks to use for a multipart upload of the specified file.
    """
    return math.ceil(os.path.getsize(local_file) / chunk_size)


def _complete_futures(futures_dict, file):
    """
    Waits for the completion of all the futures in the given dictionary and returns
    a tuple of two dictionaries. The first dictionary contains the results of the
    futures (unordered) and the second contains the errors (unordered) that occurred
    during the execution of the futures.
    """
    results = {}
    errors = {}

    with ArtifactProgressBar.chunks(
        os.path.getsize(file),
        f"Uploading {file}",
        MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE.get(),
    ) as pbar:
        for future in as_completed(futures_dict):
            key = futures_dict[future]
            try:
                results[key] = future.result()
                pbar.update()
            except Exception as e:
                errors[key] = repr(e)

    return results, errors


def _retry_with_new_creds(try_func, creds_func, og_creds=None):
    """
    Attempt the try_func with the original credentials (og_creds) if provided, or by generating the
    credentials using creds_func. If the try_func throws, then try again with new credentials
    provided by creds_func.
    """
    try:
        first_creds = creds_func() if og_creds is None else og_creds
        return try_func(first_creds)
    except Exception as e:
        _logger.info(
            "Failed to complete request, possibly due to credential expiration."
            f" Refreshing credentials and trying again... (Error: {e})"
        )
        new_creds = creds_func()
        return try_func(new_creds)


StagedArtifactUpload = namedtuple(
    "StagedArtifactUpload",
    [
        # Local filesystem path of the source file to upload
        "src_file_path",
        # Base artifact URI-relative path specifying the upload destination
        "artifact_file_path",
    ],
)


class CloudArtifactRepository(ArtifactRepository):
    def __init__(self, artifact_uri):
        super().__init__(artifact_uri)
        # Use an isolated thread pool executor for chunk uploads/downloads to avoid a deadlock
        # caused by waiting for a chunk-upload/download task within a file-upload/download task.
        # See https://superfastpython.com/threadpoolexecutor-deadlock/#Deadlock_1_Submit_and_Wait_for_a_Task_Within_a_Task
        # for more details
        self.chunk_thread_pool = self._create_thread_pool()

    # Write APIs

    def log_artifacts(self, local_dir, artifact_path=None):
        """
        Parallelized implementation of `log_artifacts`.
        """

        artifact_path = artifact_path or ""

        staged_uploads = []
        for dirpath, _, filenames in os.walk(local_dir):
            artifact_subdir = artifact_path
            if dirpath != local_dir:
                rel_path = os.path.relpath(dirpath, local_dir)
                rel_path = relative_path_to_artifact_path(rel_path)
                artifact_subdir = posixpath.join(artifact_path, rel_path)
            for name in filenames:
                src_file_path = os.path.join(dirpath, name)
                src_file_name = os.path.basename(src_file_path)
                staged_uploads.append(
                    StagedArtifactUpload(
                        src_file_path=src_file_path,
                        artifact_file_path=posixpath.join(artifact_subdir, src_file_name),
                    )
                )

        # Join futures to ensure that all artifacts have been uploaded prior to returning
        failed_uploads = {}

        # For each batch of files, upload them in parallel and wait for completion
        # TODO: change to class method
        def upload_artifacts_iter():
            for staged_upload_chunk in chunk_list(staged_uploads, _ARTIFACT_UPLOAD_BATCH_SIZE):
                write_credential_infos = self._get_write_credential_infos(
                    remote_file_paths=[
                        staged_upload.artifact_file_path for staged_upload in staged_upload_chunk
                    ],
                )

                inflight_uploads = {}
                for staged_upload, write_credential_info in zip(
                    staged_upload_chunk, write_credential_infos
                ):
                    upload_future = self.thread_pool.submit(
                        self._upload_to_cloud,
                        cloud_credential_info=write_credential_info,
                        src_file_path=staged_upload.src_file_path,
                        artifact_file_path=staged_upload.artifact_file_path,
                    )
                    inflight_uploads[staged_upload.src_file_path] = upload_future

                yield from inflight_uploads.items()

        with ArtifactProgressBar.files(
            desc="Uploading artifacts", total=len(staged_uploads)
        ) as pbar:
            for src_file_path, upload_future in upload_artifacts_iter():
                try:
                    upload_future.result()
                    pbar.update()
                except Exception as e:
                    failed_uploads[src_file_path] = repr(e)

        if len(failed_uploads) > 0:
            raise MlflowException(
                message=(
                    "The following failures occurred while uploading one or more artifacts"
                    f" to {self.artifact_uri}: {failed_uploads}"
                )
            )

    @abstractmethod
    def _get_write_credential_infos(self, remote_file_paths):
        """
        Retrieve write credentials for a batch of remote file paths, including presigned URLs.

        Args:
            remote_file_paths: List of file paths in the remote artifact repository.

        Returns:
            List of ArtifactCredentialInfo objects corresponding to each file path.
        """

    @abstractmethod
    def _upload_to_cloud(self, cloud_credential_info, src_file_path, artifact_file_path):
        """
        Upload a single file to the cloud.

        Args:
            cloud_credential_info: ArtifactCredentialInfo object with presigned URL for the file.
            src_file_path: Local source file path for the upload.
            artifact_file_path: Path in the artifact repository where the artifact will be logged.

        """

    # Read APIs

    def _extract_headers_from_credentials(self, headers):
        """
        Returns:
            A python dictionary of http headers converted from the protobuf credentials.
        """
        return {header.name: header.value for header in headers}

    def _parallelized_download_from_cloud(self, file_size, remote_file_path, local_path):
        read_credentials = self._get_read_credential_infos([remote_file_path])
        # Read credentials for only one file were requested. So we expected only one value in
        # the response.
        assert len(read_credentials) == 1
        cloud_credential_info = read_credentials[0]

        with remove_on_error(local_path):
            parallel_download_subproc_env = os.environ.copy()
            failed_downloads = parallelized_download_file_using_http_uri(
                thread_pool_executor=self.chunk_thread_pool,
                http_uri=cloud_credential_info.signed_uri,
                download_path=local_path,
                remote_file_path=remote_file_path,
                file_size=file_size,
                uri_type=cloud_credential_info.type,
                chunk_size=MLFLOW_MULTIPART_DOWNLOAD_CHUNK_SIZE.get(),
                env=parallel_download_subproc_env,
                headers=self._extract_headers_from_credentials(cloud_credential_info.headers),
            )
            num_retries = _MLFLOW_MPD_NUM_RETRIES.get()
            interval = _MLFLOW_MPD_RETRY_INTERVAL_SECONDS.get()
            failed_downloads = list(failed_downloads)
            while failed_downloads and num_retries > 0:
                self._refresh_credentials()
                new_cloud_creds = self._get_read_credential_infos([remote_file_path])[0]
                new_signed_uri = new_cloud_creds.signed_uri
                new_headers = self._extract_headers_from_credentials(new_cloud_creds.headers)

                futures = {
                    self.chunk_thread_pool.submit(
                        download_chunk,
                        range_start=chunk.start,
                        range_end=chunk.end,
                        headers=new_headers,
                        download_path=local_path,
                        http_uri=new_signed_uri,
                    ): chunk
                    for chunk in failed_downloads
                }

                new_failed_downloads = []

                for future in as_completed(futures):
                    chunk = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        _logger.info(
                            f"Failed to download chunk {chunk.index} for {chunk.path}: {e}. "
                            f"The download of this chunk will be retried later."
                        )
                        new_failed_downloads.append(chunk)

                failed_downloads = new_failed_downloads
                num_retries -= 1
                time.sleep(interval)

            if failed_downloads:
                raise MlflowException(
                    message=("All retries have been exhausted. Download has failed.")
                )

    def _download_file(self, remote_file_path, local_path):
        # list_artifacts API only returns a list of FileInfos at the specified path
        # if it's a directory. To get file size, we need to iterate over FileInfos
        # contained by the parent directory. A bad path could result in there being
        # no matching FileInfos (by path), so fall back to None size to prevent
        # parallelized download.
        parent_dir = posixpath.dirname(remote_file_path)
        file_infos = self.list_artifacts(parent_dir)
        file_info = [info for info in file_infos if info.path == remote_file_path]
        file_size = file_info[0].file_size if len(file_info) == 1 else None
        # NB: FUSE mounts do not support file write from a non-0th index seek position.
        # Due to this limitation (writes must start at the beginning of a file),
        # offset writes are disabled if FUSE is the local_path destination.
        if (
            not MLFLOW_ENABLE_MULTIPART_DOWNLOAD.get()
            or not file_size
            or file_size < MLFLOW_MULTIPART_DOWNLOAD_MINIMUM_FILE_SIZE.get()
            or is_fuse_or_uc_volumes_uri(local_path)
        ):
            self._download_from_cloud(remote_file_path, local_path)
        else:
            self._parallelized_download_from_cloud(file_size, remote_file_path, local_path)

    @abstractmethod
    def _get_read_credential_infos(self, remote_file_paths):
        """
        Retrieve read credentials for a batch of remote file paths, including presigned URLs.

        Args:
            remote_file_paths: List of file paths in the remote artifact repository.

        Returns:
            List of ArtifactCredentialInfo objects corresponding to each file path.
        """

    @abstractmethod
    def _download_from_cloud(self, remote_file_path, local_path):
        """
        Download a file from the input `remote_file_path` and save it to `local_path`.

        Args:
            remote_file_path: Path to file in the remote artifact repository.
            local_path: Local path to download file to.

        """

    def _refresh_credentials(self):
        """
        Refresh credentials for user in the case of credential expiration

        Args:
            None
        """
