import logging
import math
import os
import posixpath
from abc import abstractmethod
from collections import namedtuple
from concurrent.futures import as_completed

from mlflow.environment_variables import (
    MLFLOW_ENABLE_MULTIPART_DOWNLOAD,
    MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR,
)
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.utils import chunk_list
from mlflow.utils.file_utils import (
    ArtifactProgressBar,
    parallelized_download_file_using_http_uri,
    relative_path_to_artifact_path,
    download_chunk,
)

_logger = logging.getLogger(__name__)
_DOWNLOAD_CHUNK_SIZE = 100_000_000  # 100 MB
_MULTIPART_DOWNLOAD_MINIMUM_FILE_SIZE = 500_000_000  # 500 MB
_MULTIPART_UPLOAD_CHUNK_SIZE = 10_000_000  # 10 MB
_ARTIFACT_UPLOAD_BATCH_SIZE = (
    50  # Max number of artifacts for which to fetch write credentials at once.
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
        _MULTIPART_UPLOAD_CHUNK_SIZE,
    ) as pbar:
        for future in as_completed(futures_dict):
            key = futures_dict[future]
            try:
                results[key] = future.result()
                pbar.update()
            except Exception as e:
                errors[key] = repr(e)

    return results, errors


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
        StagedArtifactUpload = namedtuple(
            "StagedArtifactUpload",
            [
                # Local filesystem path of the source file to upload
                "src_file_path",
                # Base artifact URI-relative path specifying the upload destination
                "artifact_path",
            ],
        )

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
                        artifact_path=posixpath.join(artifact_subdir, src_file_name),
                    )
                )

        # Join futures to ensure that all artifacts have been uploaded prior to returning
        failed_uploads = {}

        # For each batch of files, upload them in parallel and wait for completion
        def upload_artifacts_iter():
            for staged_upload_chunk in chunk_list(staged_uploads, _ARTIFACT_UPLOAD_BATCH_SIZE):
                write_credential_infos = self._get_write_credential_infos(
                    paths=[staged_upload.artifact_path for staged_upload in staged_upload_chunk],
                )

                inflight_uploads = {}
                for staged_upload, write_credential_info in zip(
                    staged_upload_chunk, write_credential_infos
                ):
                    upload_future = self.thread_pool.submit(
                        self._upload_to_cloud,
                        cloud_credential_info=write_credential_info,
                        src_file_path=staged_upload.src_file_path,
                        artifact_path=staged_upload.artifact_path,
                    )
                    inflight_uploads[staged_upload.src_file_path] = upload_future

                yield from inflight_uploads.items()

        with ArtifactProgressBar.files(
            desc="Uploading artifacts", total=len(staged_uploads)
        ) as pbar:
            if len(staged_uploads) >= 10 and pbar.pbar:
                _logger.info(
                    "The progress bar can be disabled by setting the environment "
                    f"variable {MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR} to false"
                )
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
    def _get_write_credential_infos(self, paths):
        """
        For a batch of local files, get the write credentials for each file, which include
        a presigned URL per file

        Return a list of CredentialInfo objects, one for each file.
        """
        pass

    @abstractmethod
    def _upload_to_cloud(self, cloud_credential_info, src_file_path, artifact_path):
        """
        Upload a single file to the cloud.
        :param cloud_credential_info: ArtifactCredentialInfo containing presigned URL for the current file
                                      Note: in S3 this gets ignored
        :param src_file_path: source file path
        TODO rename this variable and downstream variables to artifact_file_path (that is more accurate)
        :param artifact_path: artifact path, including file name
        :return:
        """
        pass

    # Read APIs

    def _extract_headers_from_credentials(self, headers):
        """
        :return: A python dictionary of http headers converted from the protobuf credentials
        """
        return {header.name: header.value for header in headers}

    def _parallelized_download_from_cloud(self, file_size, remote_file_path, local_path):
        read_credentials = self._get_read_credential_infos([remote_file_path])
        # Read credentials for only one file were requested. So we expected only one value in
        # the response.
        assert len(read_credentials) == 1
        cloud_credential_info = read_credentials[0]
        # from mlflow.utils.databricks_utils import get_databricks_env_vars

        try:
            parallel_download_subproc_env = os.environ.copy()
            # parallel_download_subproc_env.update(
            #     get_databricks_env_vars(self.databricks_profile_uri)
            # )
            failed_downloads = parallelized_download_file_using_http_uri(
                thread_pool_executor=self.chunk_thread_pool,
                http_uri=cloud_credential_info.signed_uri,
                download_path=local_path,
                file_size=file_size,
                uri_type=cloud_credential_info.type,
                chunk_size=_DOWNLOAD_CHUNK_SIZE,
                env=parallel_download_subproc_env,
                headers=self._extract_headers_from_credentials(cloud_credential_info.headers),
            )
            download_errors = [
                e for e in failed_downloads.values() if e["error_status_code"] not in (401, 403)
            ]
            if download_errors:
                raise MlflowException(
                    f"Failed to download artifact {remote_file_path}: " f"{download_errors}"
                )

            if failed_downloads:
                new_cloud_creds = self._get_read_credential_infos([remote_file_path])[0]
                new_signed_uri = new_cloud_creds.signed_uri
                new_headers = self._extract_headers_from_credentials(new_cloud_creds.headers)

                for i in failed_downloads:
                    download_chunk(i, _DOWNLOAD_CHUNK_SIZE, new_headers, local_path, new_signed_uri)
        except Exception as err:
            if os.path.exists(local_path):
                os.remove(local_path)
            raise MlflowException(err)

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
        if (
            not file_size
            or file_size < _MULTIPART_DOWNLOAD_MINIMUM_FILE_SIZE
            or not MLFLOW_ENABLE_MULTIPART_DOWNLOAD.get()
        ):
            print("TEST THIS IS NORMAL")
            self._download_from_cloud(remote_file_path, local_path)
        else:
            print("TEST THIS IS PARALLEL")
            self._parallelized_download_from_cloud(file_size, remote_file_path, local_path)

    @abstractmethod
    def _download_from_cloud(self, remote_file_path, local_path):
        pass
