import logging
import math
import os
import posixpath

import requests
from requests import HTTPError

from mlflow.entities import FileInfo
from mlflow.entities.multipart_upload import CreateMultipartUploadResponse, MultipartUploadPart
from mlflow.environment_variables import (
    MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD,
    MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE,
    MLFLOW_MULTIPART_UPLOAD_MINIMUM_FILE_SIZE,
)
from mlflow.exceptions import _UnsupportedMultipartUploadException
from mlflow.store.artifact.artifact_repo import (
    ArtifactRepository,
    MultipartUploadMixin,
    verify_artifact_path,
)
from mlflow.tracking._tracking_service.utils import _get_default_host_creds
from mlflow.utils.file_utils import relative_path_to_artifact_path
from mlflow.utils.mime_type_utils import _guess_mime_type
from mlflow.utils.rest_utils import augmented_raise_for_status, http_request

_logger = logging.getLogger(__name__)


class HttpArtifactRepository(ArtifactRepository, MultipartUploadMixin):
    """Stores artifacts in a remote artifact storage using HTTP requests"""

    @property
    def _host_creds(self):
        return _get_default_host_creds(self.artifact_uri)

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
        host_creds = _get_default_host_creds(url)
        resp = http_request(host_creds, endpoint, "GET", params=params)
        augmented_raise_for_status(resp)
        file_infos = []
        for f in resp.json().get("files", []):
            file_info = FileInfo(
                posixpath.join(path, f["path"]) if path else f["path"],
                f["is_dir"],
                int(f["file_size"]) if ("file_size" in f) else None,
            )
            file_infos.append(file_info)

        return sorted(file_infos, key=lambda f: f.path)

    def _download_file(self, remote_file_path, local_path):
        endpoint = posixpath.join("/", remote_file_path)
        resp = http_request(self._host_creds, endpoint, "GET", stream=True)
        augmented_raise_for_status(resp)
        with open(local_path, "wb") as f:
            chunk_size = 1024 * 1024  # 1 MB
            for chunk in resp.iter_content(chunk_size=chunk_size):
                f.write(chunk)

    def delete_artifacts(self, artifact_path=None):
        endpoint = posixpath.join("/", artifact_path) if artifact_path else "/"
        resp = http_request(self._host_creds, endpoint, "DELETE", stream=True)
        augmented_raise_for_status(resp)

    def create_multipart_upload(self, local_file, num_parts=1, artifact_path=None):
        url, _ = self.artifact_uri.split("/mlflow-artifacts", maxsplit=1)
        host_creds = _get_default_host_creds(url)
        base_endpoint = "/mlflow-artifacts/mpu/create"
        endpoint = posixpath.join(base_endpoint, artifact_path) if artifact_path else base_endpoint
        params = {
            "path": local_file,
            "num_parts": num_parts,
        }
        resp = http_request(host_creds, endpoint, "POST", json=params)
        augmented_raise_for_status(resp)
        return CreateMultipartUploadResponse.from_dict(resp.json())

    def complete_multipart_upload(self, local_file, upload_id, parts=None, artifact_path=None):
        url, _ = self.artifact_uri.split("/mlflow-artifacts", maxsplit=1)
        host_creds = _get_default_host_creds(url)
        base_endpoint = "/mlflow-artifacts/mpu/complete"
        endpoint = posixpath.join(base_endpoint, artifact_path) if artifact_path else base_endpoint
        params = {
            "path": local_file,
            "upload_id": upload_id,
            "parts": [{"part_number": part.part_number, "etag": part.etag} for part in parts],
        }
        resp = http_request(host_creds, endpoint, "POST", json=params)
        augmented_raise_for_status(resp)

    def abort_multipart_upload(self, local_file, upload_id, artifact_path=None):
        url, _ = self.artifact_uri.split("/mlflow-artifacts", maxsplit=1)
        host_creds = _get_default_host_creds(url)
        base_endpoint = "/mlflow-artifacts/mpu/abort"
        endpoint = posixpath.join(base_endpoint, artifact_path) if artifact_path else base_endpoint
        params = {
            "path": local_file,
            "upload_id": upload_id,
        }
        resp = http_request(host_creds, endpoint, "POST", json=params)
        augmented_raise_for_status(resp)

    def _try_multipart_upload(self, local_file, artifact_path=None):
        """
        Attempts to perform multipart upload to log an artifact.
        Returns if the multipart upload is successful.
        Raises UnsupportedMultipartUploadException if multipart upload is unsupported.
        """
        parts = []
        chunk_size = MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE.get()
        size = os.path.getsize(local_file)
        num_parts = math.ceil(size / chunk_size)

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
            with open(local_file, "rb") as f:
                for credential in create.credentials:
                    chunk = f.read(chunk_size)
                    response = requests.put(credential.url, data=chunk)
                    augmented_raise_for_status(response)
                    parts.append(
                        MultipartUploadPart(
                            part_number=credential.part_number,
                            etag=response.headers["ETag"],
                        )
                    )

            self.complete_multipart_upload(local_file, create.upload_id, parts, artifact_path)
        except Exception as e:
            self.abort_multipart_upload(local_file, create.upload_id, artifact_path)
            _logger.warning(f"Failed to upload file {local_file} using multipart upload: {e}")
            raise
