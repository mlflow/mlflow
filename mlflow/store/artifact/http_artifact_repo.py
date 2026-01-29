import logging
import os
import posixpath
import time
from concurrent.futures import as_completed

import requests
from requests import HTTPError

from mlflow.entities import FileInfo
from mlflow.entities.multipart_upload import (
    CreateMultipartUploadResponse,
    MultipartUploadCredential,
    MultipartUploadPart,
)
from mlflow.entities.presigned_download import PresignedDownloadUrlResponse
from mlflow.environment_variables import (
    _MLFLOW_MPD_NUM_RETRIES,
    _MLFLOW_MPD_RETRY_INTERVAL_SECONDS,
    MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD,
    MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD,
    MLFLOW_MULTIPART_DOWNLOAD_CHUNK_SIZE,
    MLFLOW_MULTIPART_DOWNLOAD_MINIMUM_FILE_SIZE,
    MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE,
    MLFLOW_MULTIPART_UPLOAD_MINIMUM_FILE_SIZE,
)
from mlflow.exceptions import (
    MlflowException,
    _UnsupportedMultipartUploadException,
    _UnsupportedPresignedDownloadException,
)
from mlflow.store.artifact.artifact_repo import (
    ArtifactRepository,
    MultipartUploadMixin,
    verify_artifact_path,
)
from mlflow.store.artifact.cloud_artifact_repo import _complete_futures, _compute_num_chunks
from mlflow.utils.credentials import get_default_host_creds
from mlflow.utils.file_utils import (
    ArtifactProgressBar,
    _yield_chunks,
    read_chunk,
    relative_path_to_artifact_path,
    remove_on_error,
)
from mlflow.utils.mime_type_utils import _guess_mime_type
from mlflow.utils.request_utils import download_chunk
from mlflow.utils.rest_utils import augmented_raise_for_status, http_request
from mlflow.utils.uri import validate_path_is_safe

_logger = logging.getLogger(__name__)


class HttpArtifactRepository(ArtifactRepository, MultipartUploadMixin):
    """Stores artifacts in a remote artifact storage using HTTP requests"""

    def __init__(self, artifact_uri):
        super().__init__(artifact_uri)
        # Use an isolated thread pool executor for chunk downloads to avoid deadlocks
        # caused by waiting for a chunk-download task within a file-download task.
        self.chunk_thread_pool = self._create_thread_pool()

    @property
    def _host_creds(self):
        return get_default_host_creds(self.artifact_uri)

    def log_artifact(self, local_file, artifact_path=None):
        verify_artifact_path(artifact_path)

        # Try to perform multipart upload if the file is large.
        # If the server does not support, or if the upload failed, revert to normal upload.
        if (
            MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD.get()
            and os.path.getsize(local_file) >= MLFLOW_MULTIPART_UPLOAD_MINIMUM_FILE_SIZE.get()
        ):
            try:
                self._try_multipart_upload(local_file, artifact_path)
                return
            except _UnsupportedMultipartUploadException:
                pass

        file_name = os.path.basename(local_file)
        mime_type = _guess_mime_type(file_name)
        paths = (artifact_path, file_name) if artifact_path else (file_name,)
        endpoint = posixpath.join("/", *paths)
        extra_headers = {"Content-Type": mime_type}
        with open(local_file, "rb") as f:
            resp = http_request(
                self._host_creds, endpoint, "PUT", data=f, extra_headers=extra_headers
            )
            augmented_raise_for_status(resp)

    def log_artifacts(self, local_dir, artifact_path=None):
        local_dir = os.path.abspath(local_dir)
        for root, _, filenames in os.walk(local_dir):
            if root == local_dir:
                artifact_dir = artifact_path
            else:
                rel_path = os.path.relpath(root, local_dir)
                rel_path = relative_path_to_artifact_path(rel_path)
                artifact_dir = (
                    posixpath.join(artifact_path, rel_path) if artifact_path else rel_path
                )
            for f in filenames:
                self.log_artifact(os.path.join(root, f), artifact_dir)

    def list_artifacts(self, path=None):
        endpoint = "/mlflow-artifacts/artifacts"
        url, tail = self.artifact_uri.split(endpoint, maxsplit=1)
        root = tail.lstrip("/")
        params = {"path": posixpath.join(root, path) if path else root}
        host_creds = get_default_host_creds(url)
        resp = http_request(host_creds, endpoint, "GET", params=params)
        augmented_raise_for_status(resp)
        file_infos = []
        for f in resp.json().get("files", []):
            validated_path = validate_path_is_safe(f["path"])
            file_info = FileInfo(
                posixpath.join(path, validated_path) if path else validated_path,
                f["is_dir"],
                int(f["file_size"]) if ("file_size" in f) else None,
            )
            file_infos.append(file_info)

        return sorted(file_infos, key=lambda f: f.path)

    def _download_file(self, remote_file_path, local_path):
        # Try multipart download via presigned URL if enabled and file is large enough
        if MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD.get():
            try:
                presigned_response = self._get_presigned_download_url(remote_file_path)
                file_size = presigned_response.file_size
                chunk_size = MLFLOW_MULTIPART_DOWNLOAD_CHUNK_SIZE.get()
                min_size = MLFLOW_MULTIPART_DOWNLOAD_MINIMUM_FILE_SIZE.get()

                if file_size is not None and file_size >= min_size:
                    self._multipart_download(
                        presigned_response=presigned_response,
                        remote_file_path=remote_file_path,
                        local_path=local_path,
                        file_size=file_size,
                        chunk_size=chunk_size,
                    )
                    return
            except _UnsupportedPresignedDownloadException:
                pass  # Fall back to proxied download
            except HTTPError as e:
                # Check if the server doesn't support presigned downloads
                error_message = e.response.json().get("message", "") if e.response else ""
                if isinstance(error_message, str) and error_message.startswith(
                    _UnsupportedPresignedDownloadException.MESSAGE
                ):
                    pass  # Fall back to proxied download
                else:
                    raise

        # Fall back to proxied download through the tracking server
        endpoint = posixpath.join("/", remote_file_path)
        resp = http_request(self._host_creds, endpoint, "GET", stream=True)
        augmented_raise_for_status(resp)
        with open(local_path, "wb") as f:
            chunk_size = 1024 * 1024  # 1 MB
            for chunk in resp.iter_content(chunk_size=chunk_size):
                f.write(chunk)

    def _multipart_download(
        self, presigned_response, remote_file_path, local_path, file_size, chunk_size
    ):
        """
        Download a file in parallel chunks using HTTP Range requests with presigned URLs.

        Args:
            presigned_response: PresignedDownloadUrlResponse containing URL and headers.
            remote_file_path: Path to the remote file (for logging purposes).
            local_path: Local path to save the downloaded file.
            file_size: Size of the file in bytes.
            chunk_size: Size of each chunk to download.
        """
        http_uri = presigned_response.url
        headers = presigned_response.headers

        with remove_on_error(local_path):
            # Create file before parallel downloads so workers can seek to their positions
            with open(local_path, "wb") as f:
                f.truncate(file_size)

            chunks = list(_yield_chunks(remote_file_path, file_size, chunk_size))

            # Submit all chunk downloads to thread pool
            futures = {
                self.chunk_thread_pool.submit(
                    download_chunk,
                    range_start=chunk.start,
                    range_end=chunk.end,
                    headers=headers,
                    download_path=local_path,
                    http_uri=http_uri,
                ): chunk
                for chunk in chunks
            }

            failed_downloads = []
            with ArtifactProgressBar.chunks(
                file_size, f"Downloading {remote_file_path}", chunk_size
            ) as pbar:
                for future in as_completed(futures):
                    chunk = futures[future]
                    try:
                        future.result()
                        pbar.update()
                    except Exception as e:
                        _logger.debug(
                            f"Failed to download chunk {chunk.index} for {chunk.path}: {e}. "
                            f"The download of this chunk will be retried."
                        )
                        failed_downloads.append(chunk)

            # Retry failed downloads
            self._retry_failed_downloads(
                failed_downloads=failed_downloads,
                remote_file_path=remote_file_path,
                local_path=local_path,
                http_uri=http_uri,
                headers=headers,
            )

    def _retry_failed_downloads(
        self, failed_downloads, remote_file_path, local_path, http_uri, headers
    ):
        """
        Retry downloading failed chunks with exponential backoff.
        """
        if not failed_downloads:
            return

        num_retries = _MLFLOW_MPD_NUM_RETRIES.get()
        interval = _MLFLOW_MPD_RETRY_INTERVAL_SECONDS.get()

        while failed_downloads and num_retries > 0:
            _logger.info(
                f"Retrying {len(failed_downloads)} failed chunk(s) for {remote_file_path}. "
                f"Retries remaining: {num_retries}"
            )
            time.sleep(interval)

            # Re-fetch presigned URL in case the old one expired
            try:
                new_presigned_response = self._get_presigned_download_url(remote_file_path)
                http_uri = new_presigned_response.url
                headers = new_presigned_response.headers
            except Exception as e:
                _logger.warning(f"Failed to refresh presigned URL: {e}. Using previous URL.")

            futures = {
                self.chunk_thread_pool.submit(
                    download_chunk,
                    range_start=chunk.start,
                    range_end=chunk.end,
                    headers=headers,
                    download_path=local_path,
                    http_uri=http_uri,
                ): chunk
                for chunk in failed_downloads
            }

            new_failed_downloads = []
            for future in as_completed(futures):
                chunk = futures[future]
                try:
                    future.result()
                except Exception as e:
                    _logger.debug(f"Retry failed for chunk {chunk.index} of {chunk.path}: {e}")
                    new_failed_downloads.append(chunk)

            failed_downloads = new_failed_downloads
            num_retries -= 1

        if failed_downloads:
            raise MlflowException(
                f"Failed to download {len(failed_downloads)} chunk(s) for {remote_file_path} "
                f"after all retries exhausted."
            )

    def _get_presigned_download_url(self, remote_file_path):
        """
        Get a presigned URL for downloading an artifact directly from cloud storage.

        Args:
            remote_file_path: The path to the artifact relative to the artifact URI.

        Returns:
            PresignedDownloadUrlResponse containing the presigned URL and headers.

        Raises:
            _UnsupportedPresignedDownloadException: If the server doesn't support
                presigned downloads.
        """
        uri, path = self.artifact_uri.split("/mlflow-artifacts/artifacts", maxsplit=1)
        path = path.strip("/")
        artifact_path = posixpath.join(path, remote_file_path) if path else remote_file_path
        endpoint = f"/mlflow-artifacts/mpd/presigned/{artifact_path}"
        host_creds = get_default_host_creds(uri)
        resp = http_request(host_creds, endpoint, "GET")
        augmented_raise_for_status(resp)
        return PresignedDownloadUrlResponse.from_dict(resp.json())

    def delete_artifacts(self, artifact_path=None):
        endpoint = posixpath.join("/", artifact_path) if artifact_path else "/"
        resp = http_request(self._host_creds, endpoint, "DELETE", stream=True)
        augmented_raise_for_status(resp)

    def _construct_mpu_uri_and_path(self, base_endpoint, artifact_path):
        uri, path = self.artifact_uri.split("/mlflow-artifacts/artifacts", maxsplit=1)
        path = path.strip("/")
        endpoint = (
            posixpath.join(base_endpoint, path, artifact_path)
            if artifact_path
            else posixpath.join(base_endpoint, path)
        )
        return uri, endpoint

    def create_multipart_upload(self, local_file, num_parts=1, artifact_path=None):
        uri, endpoint = self._construct_mpu_uri_and_path(
            "/mlflow-artifacts/mpu/create", artifact_path
        )
        host_creds = get_default_host_creds(uri)
        params = {
            "path": local_file,
            "num_parts": num_parts,
        }
        resp = http_request(host_creds, endpoint, "POST", json=params)
        augmented_raise_for_status(resp)
        return CreateMultipartUploadResponse.from_dict(resp.json())

    def complete_multipart_upload(self, local_file, upload_id, parts=None, artifact_path=None):
        uri, endpoint = self._construct_mpu_uri_and_path(
            "/mlflow-artifacts/mpu/complete", artifact_path
        )
        host_creds = get_default_host_creds(uri)
        params = {
            "path": local_file,
            "upload_id": upload_id,
            "parts": [part.to_dict() for part in parts],
        }
        resp = http_request(host_creds, endpoint, "POST", json=params)
        augmented_raise_for_status(resp)

    def abort_multipart_upload(self, local_file, upload_id, artifact_path=None):
        uri, endpoint = self._construct_mpu_uri_and_path(
            "/mlflow-artifacts/mpu/abort", artifact_path
        )
        host_creds = get_default_host_creds(uri)
        params = {
            "path": local_file,
            "upload_id": upload_id,
        }
        resp = http_request(host_creds, endpoint, "POST", json=params)
        augmented_raise_for_status(resp)

    @staticmethod
    def _upload_part(credential: MultipartUploadCredential, local_file, size, start_byte):
        data = read_chunk(local_file, size, start_byte)
        response = requests.put(credential.url, data=data, headers=credential.headers)
        augmented_raise_for_status(response)
        return MultipartUploadPart(
            part_number=credential.part_number,
            etag=response.headers.get("ETag", ""),
            url=credential.url,
        )

    def _try_multipart_upload(self, local_file, artifact_path=None):
        """
        Attempts to perform multipart upload to log an artifact.
        Returns if the multipart upload is successful.
        Raises UnsupportedMultipartUploadException if multipart upload is unsupported.
        """
        chunk_size = MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE.get()
        num_parts = _compute_num_chunks(local_file, chunk_size)

        try:
            create = self.create_multipart_upload(local_file, num_parts, artifact_path)
        except HTTPError as e:
            # return False if server does not support multipart upload
            error_message = e.response.json().get("message", "")
            if isinstance(error_message, str) and error_message.startswith(
                _UnsupportedMultipartUploadException.MESSAGE
            ):
                raise _UnsupportedMultipartUploadException()
            raise

        try:
            futures = {}
            for i, credential in enumerate(create.credentials):
                future = self.thread_pool.submit(
                    self._upload_part,
                    credential=credential,
                    local_file=local_file,
                    size=chunk_size,
                    start_byte=chunk_size * i,
                )
                futures[future] = credential.part_number

            parts, errors = _complete_futures(futures, local_file)
            if errors:
                raise MlflowException(
                    f"Failed to upload at least one part of {local_file}. Errors: {errors}"
                )

            parts = sorted(parts.values(), key=lambda part: part.part_number)
            self.complete_multipart_upload(local_file, create.upload_id, parts, artifact_path)
        except Exception as e:
            self.abort_multipart_upload(local_file, create.upload_id, artifact_path)
            _logger.warning(f"Failed to upload file {local_file} using multipart upload: {e}")
            raise
