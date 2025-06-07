import logging
import os
import posixpath
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from mlflow.entities import FileInfo
from mlflow.environment_variables import (
    MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE,
    MLFLOW_MULTIPART_UPLOAD_MINIMUM_FILE_SIZE,
)
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import _NUM_MAX_THREADS
from mlflow.store.artifact.cloud_artifact_repo import CloudArtifactRepository
from mlflow.utils.file_utils import _compute_num_chunks, _complete_futures, read_chunk

DOWNLOAD_CHUNK_SIZE = 1024 * 1024 * 1024

_logger = logging.getLogger(__name__)


def _get_databricks_workspace_client():
    from databricks.sdk import WorkspaceClient

    return WorkspaceClient()


class DatabricksSDKModelsArtifactRepository(CloudArtifactRepository):
    """
    Stores and retrieves model artifacts via Databricks SDK, agnostic to the underlying cloud
    that stores the model artifacts.
    """

    def __init__(self, model_name, model_version):
        self.model_name = model_name
        self.model_version = model_version
        self.model_base_path = f"/Models/{model_name.replace('.', '/')}/{model_version}"
        self.client = _get_databricks_workspace_client()
        super().__init__(self.model_base_path)
        # Initialize thread pool for parallel uploads
        self.chunk_thread_pool = ThreadPoolExecutor(max_workers=_NUM_MAX_THREADS)

    def list_artifacts(self, path: Optional[str] = None) -> list[FileInfo]:
        dest_path = self.model_base_path
        if path:
            dest_path = posixpath.join(dest_path, path)

        file_infos = []

        # check if dest_path is file, if so return empty dir
        if not self._is_dir(dest_path):
            return file_infos

        resp = self.client.files.list_directory_contents(dest_path)
        for directory_entry in resp:
            relative_path = posixpath.relpath(directory_entry.path, self.model_base_path)
            file_infos.append(
                FileInfo(
                    path=relative_path,
                    is_dir=directory_entry.is_directory,
                    file_size=directory_entry.file_size,
                )
            )

        return sorted(file_infos, key=lambda f: f.path)

    def _is_dir(self, artifact_path):
        from databricks.sdk.errors.platform import NotFound

        try:
            self.client.files.get_directory_metadata(artifact_path)
        except NotFound:
            return False
        return True

    def _upload_to_cloud(self, cloud_credential_info, src_file_path, artifact_file_path=None):
        dest_path = self.model_base_path
        if artifact_file_path:
            dest_path = posixpath.join(dest_path, artifact_file_path)

        with open(src_file_path, "rb") as f:
            self.client.files.upload(dest_path, f, overwrite=True)

    def log_artifact(self, local_file, artifact_path=None):
        self._upload_to_cloud(
            cloud_credential_info=None,
            src_file_path=local_file,
            artifact_file_path=artifact_path,
        )

    def _download_from_cloud(self, remote_file_path, local_path):
        dest_path = self.model_base_path
        if remote_file_path:
            dest_path = posixpath.join(dest_path, remote_file_path)

        resp = self.client.files.download(dest_path)
        contents = resp.contents

        with open(local_path, "wb") as f:
            while chunk := contents.read(DOWNLOAD_CHUNK_SIZE):
                f.write(chunk)

    def _get_write_credential_infos(self, remote_file_paths):
        # Databricks sdk based model download/upload don't need any extra credentials
        return [None] * len(remote_file_paths)

    def _get_read_credential_infos(self, remote_file_paths):
        # Databricks sdk based model download/upload don't need any extra credentials
        return [None] * len(remote_file_paths)
    def _create_multipart_upload(self, run_id: str, path: str, num_parts: int) -> str:
        """Create a multipart upload and return the upload ID."""
        response = self.client.files.create_multipart_upload(
            run_id=run_id,
            path=path,
            num_parts=num_parts,
        )
        return response.upload_id

    def _upload_part(self, run_id: str, path: str, upload_id: str, part_number: int, data: bytes) -> str:
        """Upload a part of the file and return its ETag."""
        response = self.client.files.upload_part(
            run_id=run_id,
            path=path,
            upload_id=upload_id,
            part_number=part_number,
            data=data,
        )
        return response.etag

    def _complete_multipart_upload(
        self, run_id: str, path: str, upload_id: str, parts: list[tuple[int, str]]
    ) -> None:
        """Complete the multipart upload."""
        self.client.files.complete_multipart_upload(
            run_id=run_id,
            path=path,
            upload_id=upload_id,
            parts=[{"part_number": part_number, "etag": etag} for part_number, etag in parts],
        )

    def _log_artifact_mpu(self, local_file: str, artifact_path: Optional[str] = None) -> None:
        """Log an artifact to the repository using multipart upload."""
        file_size = os.path.getsize(local_file)
        chunk_size = MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE.get()
        min_file_size = MLFLOW_MULTIPART_UPLOAD_MINIMUM_FILE_SIZE.get()

        if file_size < min_file_size:
            # For small files, use a simple upload
            with open(local_file, "rb") as f:
                self.client.files.upload(
                    run_id=self.run_id,
                    path=artifact_path or os.path.basename(local_file),
                    data=f.read(),
                )
            return

        # For large files, use multipart upload
        num_chunks = _compute_num_chunks(file_size, chunk_size)
        upload_id = self._create_multipart_upload(
            run_id=self.run_id,
            path=artifact_path or os.path.basename(local_file),
            num_parts=num_chunks,
        )

        try:
            # Upload parts in parallel using futures
            futures = {}
            for part_number in range(1, num_chunks + 1):
                start_byte = (part_number - 1) * chunk_size
                future = self.chunk_thread_pool.submit(
                    self._upload_part,
                    run_id=self.run_id,
                    path=artifact_path or os.path.basename(local_file),
                    upload_id=upload_id,
                    part_number=part_number,
                    data=read_chunk(local_file, chunk_size, start_byte),
                )
                futures[future] = part_number

            results, errors = _complete_futures(futures, local_file)
            if errors:
                raise MlflowException(
                    f"Failed to upload at least one part of {local_file}. Errors: {errors}"
                )

            # Sort parts by part number and complete the upload
            parts = [(part_number, results[part_number]) for part_number in sorted(results)]
            self._complete_multipart_upload(
                run_id=self.run_id,
                path=artifact_path or os.path.basename(local_file),
                upload_id=upload_id,
                parts=parts,
            )
        except Exception as e:
            _logger.warning(
                f"Encountered an unexpected error during multipart upload: {e}, aborting"
            )
            # TODO: Implement abort_multipart_upload when it's available in the Files API
            raise e

