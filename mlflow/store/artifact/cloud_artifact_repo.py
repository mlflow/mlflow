import os
import posixpath
from abc import abstractmethod
from collections import namedtuple

from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.store.artifact.databricks_artifact_repo import MLFLOW_ENABLE_MULTIPART_DOWNLOAD
from mlflow.utils import chunk_list
from mlflow.utils.file_utils import (
    parallelized_download_file_using_http_uri,
    relative_path_to_artifact_path,
    download_chunk,
)

_DOWNLOAD_CHUNK_SIZE = 100_000_000  # 100 MB
_MULTIPART_DOWNLOAD_MINIMUM_FILE_SIZE = 500_000_000  # 500 MB
_MULTIPART_UPLOAD_CHUNK_SIZE = 10_000_000  # 10 MB
_ARTIFACT_UPLOAD_BATCH_SIZE = (
    50  # Max number of artifacts for which to fetch write credentials at once.
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

    def log_artifacts_parallel(self, local_dir, artifact_path=None):
        """
        Parallelized implementation of `download_artifacts` for Databricks.
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

        # For each batch of files, upload them in parallel
        # and wait for completion
        def get_creds_and_upload(staged_upload_chunk):
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

            for src_file_path, upload_future in inflight_uploads.items():
                try:
                    upload_future.result()
                except Exception as e:
                    failed_uploads[src_file_path] = repr(e)

        # Iterate over batches of files and upload them
        for chunk in chunk_list(staged_uploads, _ARTIFACT_UPLOAD_BATCH_SIZE):
            get_creds_and_upload(chunk)

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
        :param src_file_path:
        :param artifact_path:
        :return:
        """
        pass

    # Read APIs

    def _extract_headers_from_credentials(self, headers):
        """
        :return: A python dictionary of http headers converted from the protobuf credentials
        """
        return {header.name: header.value for header in headers}

    def _parallelized_download_from_cloud(
        self, cloud_credential_info, file_size, dst_local_file_path, dst_run_relative_artifact_path
    ):
        # from mlflow.utils.databricks_utils import get_databricks_env_vars

        try:
            parallel_download_subproc_env = os.environ.copy()
            # parallel_download_subproc_env.update(
            #     get_databricks_env_vars(self.databricks_profile_uri)
            # )
            failed_downloads = parallelized_download_file_using_http_uri(
                thread_pool_executor=self.chunk_thread_pool,
                http_uri=cloud_credential_info.signed_uri,
                download_path=dst_local_file_path,
                file_size=file_size,
                uri_type=cloud_credential_info.type,
                chunk_size=_DOWNLOAD_CHUNK_SIZE,
                env=parallel_download_subproc_env,
                # headers={"x-ms-blob-type": "BlockBlob"},
                headers=self._extract_headers_from_credentials(cloud_credential_info.headers),
            )
            download_errors = [
                e for e in failed_downloads.values() if e["error_status_code"] not in (401, 403)
            ]
            if download_errors:
                raise MlflowException(
                    f"Failed to download artifact {dst_run_relative_artifact_path}: "
                    f"{download_errors}"
                )

            if failed_downloads:
                new_cloud_creds = self._get_read_credential_infos([dst_run_relative_artifact_path])[
                    0
                ]
                new_signed_uri = new_cloud_creds.signed_uri
                new_headers = self._extract_headers_from_credentials(new_cloud_creds.headers)

                for i in failed_downloads:
                    download_chunk(
                        i, _DOWNLOAD_CHUNK_SIZE, new_headers, dst_local_file_path, new_signed_uri
                    )
        except Exception as err:
            if os.path.exists(dst_local_file_path):
                os.remove(dst_local_file_path)
            raise MlflowException(err)

    def _download_file(self, remote_file_path, local_path):
        # TODO add back some logic to handle downloading artifacts from an artifact
        # repo that's not at the root of the run? Don't think it makes sense to have
        # that logic since this repo is unaware of runs.
        # remote_full_path = posixpath.join(
        #     self.base_data_lake_directory, remote_file_path
        # )
        read_credentials = self._get_read_credential_infos(paths=[remote_file_path])
        # Read credentials for only one file were requested. So we expected only one value in
        # the response.
        assert len(read_credentials) == 1

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
            self._download_from_cloud(remote_file_path=remote_file_path, local_path=local_path)
        else:
            self._parallelized_download_from_cloud(
                read_credentials[0], file_size, local_path, remote_file_path
            )

    @abstractmethod
    def _download_from_cloud(self, remote_file_path, local_path):
        pass
