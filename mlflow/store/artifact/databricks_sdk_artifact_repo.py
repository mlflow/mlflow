import importlib.metadata
import logging
import posixpath
from concurrent.futures import Future
from pathlib import Path
from typing import TYPE_CHECKING

from packaging.version import Version

from mlflow.entities import FileInfo
from mlflow.environment_variables import MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository

if TYPE_CHECKING:
    from databricks.sdk.service.files import FilesAPI


def _sdk_supports_large_file_uploads() -> bool:
    # https://github.com/databricks/databricks-sdk-py/commit/7ca3fb7e8643126b74c9f5779dc01fb20c1741fb
    return Version(importlib.metadata.version("databricks-sdk")) >= Version("0.45.0")


_logger = logging.getLogger(__name__)


# TODO: The following artifact repositories should use this class. Migrate them.
#   - databricks_sdk_models_artifact_repo.py
class DatabricksSdkArtifactRepository(ArtifactRepository):
    def __init__(
        self, artifact_uri: str, tracking_uri: str | None = None, registry_uri: str | None = None
    ) -> None:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.config import Config

        super().__init__(artifact_uri, tracking_uri, registry_uri)
        supports_large_file_uploads = _sdk_supports_large_file_uploads()
        wc = WorkspaceClient(
            config=(
                Config(enable_experimental_files_api_client=True)
                if supports_large_file_uploads
                else None
            )
        )
        if supports_large_file_uploads:
            # `Config` has a `multipart_upload_min_stream_size` parameter but the constructor
            # doesn't set it. This is a bug in databricks-sdk.
            # >>> from databricks.sdk.config import Config
            # >>> config = Config(multipart_upload_chunk_size=123)
            # >>> assert config.multipart_upload_chunk_size != 123
            try:
                wc.files._config.multipart_upload_chunk_size = (
                    MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE.get()
                )
            except AttributeError:
                _logger.debug("Failed to set multipart_upload_chunk_size in Config", exc_info=True)
        self.wc = wc

    @property
    def files_api(self) -> "FilesAPI":
        return self.wc.files

    def _is_dir(self, path: str) -> bool:
        from databricks.sdk.errors.platform import NotFound

        try:
            self.files_api.get_directory_metadata(path)
        except NotFound:
            return False
        return True

    def full_path(self, artifact_path: str | None) -> str:
        return f"{self.artifact_uri}/{artifact_path}" if artifact_path else self.artifact_uri

    def log_artifact(self, local_file: str, artifact_path: str | None = None) -> None:
        if Path(local_file).stat().st_size > 5 * (1024**3) and not _sdk_supports_large_file_uploads:
            raise MlflowException.invalid_parameter_value(
                "Databricks SDK version < 0.41.0 does not support uploading files larger than 5GB. "
                "Please upgrade the databricks-sdk package to version >= 0.41.0."
            )

        with open(local_file, "rb") as f:
            name = Path(local_file).name
            self.files_api.upload(
                self.full_path(posixpath.join(artifact_path, name) if artifact_path else name),
                f,
                overwrite=True,
            )

    def log_artifacts(self, local_dir: str, artifact_path: str | None = None) -> None:
        local_dir = Path(local_dir).resolve()
        futures: list[Future[None]] = []
        with self._create_thread_pool() as executor:
            for f in local_dir.rglob("*"):
                if not f.is_file():
                    continue

                paths: list[str] = []
                if artifact_path:
                    paths.append(artifact_path)
                if f.parent != local_dir:
                    paths.append(str(f.parent.relative_to(local_dir)))

                fut = executor.submit(
                    self.log_artifact,
                    local_file=f,
                    artifact_path=posixpath.join(*paths) if paths else None,
                )
                futures.append(fut)

        for fut in futures:
            fut.result()

    def list_artifacts(self, path: str | None = None) -> list[FileInfo]:
        dest_path = self.full_path(path)
        if not self._is_dir(dest_path):
            return []

        file_infos: list[FileInfo] = []
        for directory_entry in self.files_api.list_directory_contents(dest_path):
            relative_path = posixpath.relpath(directory_entry.path, self.artifact_uri)
            file_infos.append(
                FileInfo(
                    path=relative_path,
                    is_dir=directory_entry.is_directory,
                    file_size=directory_entry.file_size,
                )
            )

        return sorted(file_infos, key=lambda f: f.path)

    def _download_file(self, remote_file_path: str, local_path: str) -> None:
        download_resp = self.files_api.download(self.full_path(remote_file_path))
        with open(local_path, "wb") as f:
            while chunk := download_resp.contents.read(10 * 1024 * 1024):
                f.write(chunk)
