import logging
import math
import os
import posixpath
import time
from abc import abstractmethod
from concurrent.futures import Future, as_completed
from dataclasses import dataclass
from typing import NamedTuple

from mlflow.environment_variables import (
    _MLFLOW_MPD_NUM_RETRIES,
    _MLFLOW_MPD_RETRY_INTERVAL_SECONDS,
    MLFLOW_ENABLE_MULTIPART_DOWNLOAD,
    MLFLOW_MULTIPART_DOWNLOAD_CHUNK_SIZE,
    MLFLOW_MULTIPART_DOWNLOAD_MINIMUM_FILE_SIZE,
    MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE,
    MLFLOW_MULTIPART_UPLOAD_MINIMUM_FILE_SIZE,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_artifacts_pb2 import ArtifactCredentialInfo, ArtifactCredentialType
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
_ARTIFACT_UPLOAD_BATCH_SIZE = 50
_AWS_MIN_CHUNK_SIZE = 5 * 1024**2  # 5 MB
_AWS_MAX_CHUNK_SIZE = 5 * 1024**3  # 5 GB

# Databricks storage proxy delays to avoid race conditions
# When uploading multiple files through the Databricks storage proxy, the proxy maintains
# internal state for credential exchanges and block upload operations. Adding delays between
# file uploads allows the proxy to complete its internal cleanup and prevents 500 errors.
_DATABRICKS_UPLOAD_DELAY_SECONDS = 3.0  # Delay between file uploads
_DATABRICKS_LARGE_FILE_DELAY_SECONDS = 5.0  # Delay after large files (>100MB)


def _readable_size(size: int) -> str:
    return f"{size / 1024**2:.2f} MB"


def _validate_chunk_size_aws(chunk_size: int) -> None:
    if chunk_size < _AWS_MIN_CHUNK_SIZE or chunk_size > _AWS_MAX_CHUNK_SIZE:
        raise MlflowException(
            message=(
                f"Multipart chunk size {_readable_size(chunk_size)} must be in range: "
                f"{_readable_size(_AWS_MIN_CHUNK_SIZE)} to {_readable_size(_AWS_MAX_CHUNK_SIZE)}."
            )
        )


def _compute_num_chunks(local_file: os.PathLike, chunk_size: int) -> int:
    return math.ceil(os.path.getsize(local_file) / chunk_size)


def _complete_futures(futures_dict, file):
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


class StagedArtifactUpload(NamedTuple):
    src_file_path: str  # Local filesystem path
    artifact_file_path: str  # Remote artifact path


@dataclass
class FileUploadPlan:
    """Groups all information needed to upload a single file.

    This avoids brittle parallel lists and caches the file size.
    """

    staged_upload: StagedArtifactUpload
    file_size: int
    credential_info: ArtifactCredentialInfo | None = None

    @property
    def src_path(self) -> str:
        return self.staged_upload.src_file_path

    @property
    def dest_path(self) -> str:
        return self.staged_upload.artifact_file_path


class CloudArtifactRepository(ArtifactRepository):
    def __init__(
        self, artifact_uri: str, tracking_uri: str | None = None, registry_uri: str | None = None
    ) -> None:
        super().__init__(artifact_uri, tracking_uri, registry_uri)
        # Isolated thread pool for chunk operations to avoid deadlocks
        # See: https://superfastpython.com/threadpoolexecutor-deadlock/
        self.chunk_thread_pool = self._create_thread_pool()

    def log_artifacts(self, local_dir, artifact_path=None):
        """Upload all files from local_dir to the cloud artifact store.

        This method handles three cloud providers with different upload strategies:
        - AWS: Small files use simple PUT with presigned URLs, large files use multipart upload
        - Azure/GCP: All files use PUT with SAS tokens (Azure uses block uploads)

        Databricks Storage Proxy Compatibility:
        When uploading through Databricks storage proxies, the proxy maintains internal state
        for credential exchanges and block upload operations. To avoid race conditions:

        1. Batch ALL credential requests upfront (one batch API call, not N individual calls)
           - Concurrent credential requests to the same artifact path can cause proxy conflicts
           - Requesting credentials one-at-a-time while uploads are in progress interferes
             with the proxy's session management

        2. Serialize uploads with adaptive delays
           - Wait between file uploads to allow proxy internal state cleanup
           - Larger files need more cleanup time (3s base, 5s for >100MB files)
           - Azure block uploads in particular need time for put_block_list cleanup

        3. Separate small and large files (AWS only)
           - Small files: batch credentials → parallel upload → wait
           - Delay for proxy cleanup
           - Large files: each gets multipart credentials → upload
           - This avoids mixing simple PUT and multipart operations
        """
        artifact_path = artifact_path or ""

        # Collect all files and compute sizes upfront
        upload_plans = self._collect_upload_plans(local_dir, artifact_path)
        if not upload_plans:
            return

        # Detect cloud provider type from a sample file
        cloud_type = self._detect_cloud_type(upload_plans)

        # Route to cloud-specific upload logic
        failed_uploads = {}
        multipart_threshold = MLFLOW_MULTIPART_UPLOAD_MINIMUM_FILE_SIZE.get()

        for batch in chunk_list(upload_plans, _ARTIFACT_UPLOAD_BATCH_SIZE):
            try:
                if cloud_type == ArtifactCredentialType.AWS_PRESIGNED_URL:
                    batch_failures = self._upload_batch_aws(batch, multipart_threshold)
                elif cloud_type in (
                    ArtifactCredentialType.AZURE_SAS_URI,
                    ArtifactCredentialType.AZURE_ADLS_GEN2_SAS_URI,
                ):
                    batch_failures = self._upload_batch_azure_gcp(batch, cloud_type)
                else:
                    # Unknown cloud type or no credentials - upload with what we have
                    batch_failures = self._upload_batch_generic(batch)

                failed_uploads.update(batch_failures)
            except Exception as e:
                _logger.error(f"Batch upload failed: {e}")
                for plan in batch:
                    failed_uploads[plan.src_path] = repr(e)

        if failed_uploads:
            raise MlflowException(
                message=(
                    f"The following failures occurred while uploading one or more artifacts "
                    f"to {self.artifact_uri}: {failed_uploads}"
                )
            )

    def _collect_upload_plans(self, local_dir: str, artifact_path: str) -> list[FileUploadPlan]:
        """Collect all files to upload and cache their sizes."""
        plans = []
        for dirpath, _, filenames in os.walk(local_dir):
            artifact_subdir = artifact_path
            if dirpath != local_dir:
                rel_path = os.path.relpath(dirpath, local_dir)
                rel_path = relative_path_to_artifact_path(rel_path)
                artifact_subdir = posixpath.join(artifact_path, rel_path)

            for filename in filenames:
                src_path = os.path.join(dirpath, filename)
                dest_path = posixpath.join(artifact_subdir, filename)
                file_size = os.path.getsize(src_path)

                plans.append(
                    FileUploadPlan(
                        staged_upload=StagedArtifactUpload(src_path, dest_path),
                        file_size=file_size,
                    )
                )

        return plans

    def _detect_cloud_type(
        self, upload_plans: list[FileUploadPlan]
    ) -> ArtifactCredentialType | None:
        """Detect cloud provider by fetching credentials for a single sample file.

        We prefer to sample a small file to avoid wasting credentials on AWS large files
        (which will get their own credentials via multipart upload).
        """
        if not upload_plans:
            return None

        # Find a small file if possible
        multipart_threshold = MLFLOW_MULTIPART_UPLOAD_MINIMUM_FILE_SIZE.get()
        sample_plan = None
        for plan in upload_plans:
            if plan.file_size < multipart_threshold:
                sample_plan = plan
                break

        # No small files found, use the first file
        if sample_plan is None:
            sample_plan = upload_plans[0]

        # Fetch credential to determine cloud type
        creds = self._get_write_credential_infos([sample_plan.dest_path])
        if creds and hasattr(creds[0], "type"):
            # Cache the credential so we don't waste it
            sample_plan.credential_info = creds[0]
            return creds[0].type

        return None

    def _upload_batch_aws(
        self, batch: list[FileUploadPlan], multipart_threshold: int
    ) -> dict[str, str]:
        """Upload a batch of files to AWS S3.

        AWS Strategy:
        - Small files (<threshold): Fetch presigned URLs and upload via simple PUT
        - Large files (>=threshold): Use multipart upload (fetches its own credentials)

        Databricks Storage Proxy Serialization:
        When we have both small and large files, we upload small files first (in parallel),
        wait for them to complete, then upload large files. This avoids concurrent multipart
        and simple PUT operations which can cause storage proxy 500 errors.
        """
        small_files = [p for p in batch if p.file_size < multipart_threshold]
        large_files = [p for p in batch if p.file_size >= multipart_threshold]

        # Fetch credentials for small files (large files get credentials via multipart)
        self._fetch_credentials_for_plans(small_files)

        failures = {}

        # If we have both types, serialize: small files first, then large files
        if small_files and large_files:
            # Upload small files in parallel
            failures.update(self._upload_files_parallel(small_files, wait=True))

            # Wait for storage proxy cleanup before starting large file multipart uploads
            # Each large file will request its own multipart credentials, so we need to
            # ensure the proxy has finished cleanup from small file uploads
            _logger.debug(
                f"Completed {len(small_files)} small file(s). Waiting "
                f"{_DATABRICKS_UPLOAD_DELAY_SECONDS}s before starting {len(large_files)} "
                "large file multipart upload(s)."
            )
            time.sleep(_DATABRICKS_UPLOAD_DELAY_SECONDS)

            # Then upload large files (each uses multipart internally)
            failures.update(self._upload_files_parallel(large_files, wait=False))
        else:
            # Only one type - upload in parallel
            failures.update(self._upload_files_parallel(batch, wait=False))

        return failures

    def _upload_batch_azure_gcp(
        self, batch: list[FileUploadPlan], cloud_type: ArtifactCredentialType
    ) -> dict[str, str]:
        """Upload a batch of files to Azure or GCP.

        Azure/GCP Strategy:
        - All files use SAS tokens for upload (no distinction between small/large)
        - Azure uses block uploads with put_block_list to commit blocks

        Databricks Storage Proxy Serialization:
        With multiple files, the Databricks storage proxy cannot handle concurrent credential
        requests or overlapping block upload operations to the same artifact path. We must:
        1. Fetch ALL credentials in ONE batch request (avoid N separate requests)
        2. Upload files serially with adaptive delays for proxy state cleanup
        3. For Azure: Allow extra time after put_block_list operations to complete cleanup
        """
        failures = {}

        # Multi-file case: Batch credential fetch, then serialize uploads with adaptive delays
        if len(batch) > 1:
            # CRITICAL: Fetch ALL credentials in ONE batch request to avoid proxy conflicts
            # The Databricks storage proxy maintains session state per artifact path. Requesting
            # credentials one-at-a-time while uploads are in progress causes state conflicts
            # and 500 errors. This is especially problematic for Azure block uploads.
            self._fetch_credentials_for_plans(batch)

            # Now upload serially with adaptive delays for proxy cleanup
            with ArtifactProgressBar.files(desc="Uploading artifacts", total=len(batch)) as pbar:
                for idx, plan in enumerate(batch):
                    try:
                        # Credential already fetched and cached in plan.credential_info
                        future = self.thread_pool.submit(
                            self._upload_to_cloud,
                            cloud_credential_info=plan.credential_info,
                            src_file_path=plan.src_path,
                            artifact_file_path=plan.dest_path,
                        )
                        future.result()  # Wait for completion (includes put_block_list)
                        pbar.update()

                        # Adaptive delay before next upload to avoid storage proxy race condition
                        # Larger files need more cleanup time, especially for Azure block uploads
                        # where put_block_list operations require additional proxy cleanup time
                        if idx < len(batch) - 1:
                            delay = _DATABRICKS_UPLOAD_DELAY_SECONDS
                            if plan.file_size > 100 * 1024 * 1024:  # >100MB
                                delay = _DATABRICKS_LARGE_FILE_DELAY_SECONDS
                                _logger.debug(
                                    f"Large file ({plan.file_size / 1024**2:.1f}MB) uploaded. "
                                    f"Waiting {delay}s before next upload to ensure proxy cleanup."
                                )
                            time.sleep(delay)

                    except Exception as e:
                        failures[plan.src_path] = repr(e)
        else:
            # Single file - no need to worry about proxy state conflicts
            self._fetch_credentials_for_plans(batch)
            failures.update(self._upload_files_parallel(batch, wait=False))

        return failures

    def _upload_batch_generic(self, batch: list[FileUploadPlan]) -> dict[str, str]:
        """Upload files when cloud type is unknown or credentials aren't needed."""
        self._fetch_credentials_for_plans(batch)
        return self._upload_files_parallel(batch, wait=False)

    def _fetch_credentials_for_plans(self, plans: list[FileUploadPlan]) -> None:
        """Fetch and cache credentials for plans that don't already have them.

        This uses the efficient batch API to fetch multiple credentials at once.
        Plans that already have credentials (from cloud type detection) are skipped.
        """
        plans_needing_creds = [p for p in plans if p.credential_info is None]
        if not plans_needing_creds:
            return

        try:
            creds = self._get_write_credential_infos([p.dest_path for p in plans_needing_creds])
            for plan, cred in zip(plans_needing_creds, creds):
                plan.credential_info = cred
        except Exception as e:
            _logger.warning(f"Failed to fetch credentials: {e}")

    def _upload_files_parallel(self, plans: list[FileUploadPlan], wait: bool) -> dict[str, str]:
        """Upload multiple files in parallel and optionally wait for completion.

        Args:
            plans: Files to upload
            wait: If True, block until all uploads complete before returning

        Returns:
            Dictionary mapping failed file paths to error messages
        """
        failures = {}
        futures: dict[Future, FileUploadPlan] = {}

        for plan in plans:
            future = self.thread_pool.submit(
                self._upload_to_cloud,
                cloud_credential_info=plan.credential_info,
                src_file_path=plan.src_path,
                artifact_file_path=plan.dest_path,
            )
            futures[future] = plan

        if wait:
            # Wait for all uploads to complete
            for future, plan in futures.items():
                try:
                    future.result()
                except Exception as e:
                    failures[plan.src_path] = repr(e)

        return failures

    @abstractmethod
    def _get_write_credential_infos(self, remote_file_paths):
        """Fetch write credentials for a batch of files.

        Args:
            remote_file_paths: List of remote artifact paths

        Returns:
            List of ArtifactCredentialInfo objects (same order as input)
        """

    @abstractmethod
    def _upload_to_cloud(self, cloud_credential_info, src_file_path, artifact_file_path):
        """Upload a single file to cloud storage.

        Args:
            cloud_credential_info: Credential (e.g., presigned URL) or None for multipart
            src_file_path: Local source file path
            artifact_file_path: Remote destination path
        """

    # Read APIs (unchanged from original)

    def _extract_headers_from_credentials(self, headers):
        return {header.name: header.value for header in headers}

    def _parallelized_download_from_cloud(self, file_size, remote_file_path, local_path):
        read_credentials = self._get_read_credential_infos([remote_file_path])
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
                    message="All retries have been exhausted. Download has failed."
                )

    def _download_file(self, remote_file_path, local_path):
        parent_dir = posixpath.dirname(remote_file_path)
        file_infos = self.list_artifacts(parent_dir)
        file_info = [info for info in file_infos if info.path == remote_file_path]
        file_size = file_info[0].file_size if len(file_info) == 1 else None

        if (
            not MLFLOW_ENABLE_MULTIPART_DOWNLOAD.get()
            or not file_size
            or file_size < MLFLOW_MULTIPART_DOWNLOAD_MINIMUM_FILE_SIZE.get()
            or is_fuse_or_uc_volumes_uri(local_path)
            or type(self).__name__ == "DatabricksSDKModelsArtifactRepository"
        ):
            self._download_from_cloud(remote_file_path, local_path)
        else:
            self._parallelized_download_from_cloud(file_size, remote_file_path, local_path)

    @abstractmethod
    def _get_read_credential_infos(self, remote_file_paths):
        """Fetch read credentials for a batch of files."""

    @abstractmethod
    def _download_from_cloud(self, remote_file_path, local_path):
        """Download a file from cloud storage."""

    def _refresh_credentials(self):
        """Refresh credentials after expiration."""
