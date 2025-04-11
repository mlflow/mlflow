import posixpath
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from databricks.sdk import WorkspaceClient
from databricks.sdk.config import Config
from databricks.sdk.errors.platform import NotFound
from databricks.sdk.service.files import FilesAPI

from mlflow.entities import FileInfo
from mlflow.store.artifact.artifact_repo import ArtifactRepository


# TODO: Migrate the following artifact repository to use this class.
#   - uc_volume_artifact_repo.py
#   - databricks_sdk_models_artifact_repo.py
class DatabricksSdkArtifactRepository(ArtifactRepository):
    def __init__(self, artifact_uri: str) -> None:
        super().__init__(artifact_uri)
        self.wc = WorkspaceClient(config=Config(enable_experimental_files_api_client=True))

    @property
    def files_api(self) -> FilesAPI:
        return self.wc.files

    def _is_dir(self, path: str) -> bool:
        try:
            self.files_api.get_directory_metadata(path)
        except NotFound:
            return False
        return True

    def full_path(self, artifact_path: str) -> str:
        return f"{self.root_path}/{artifact_path}" if artifact_path else self.root_path

    def log_artifact(self, local_file: str, artifact_path: Optional[str] = None) -> None:
        with open(local_file, "rb") as f:
            self.files_api.upload(self.full_path(artifact_path), f, overwrite=True)

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        futures: list[Future[None]] = []
        with ThreadPoolExecutor() as executor:
            for f in Path(local_dir).rglob("*"):
                if f.is_file():
                    fut = executor.submit(self.log_artifact, f, artifact_path)
                    futures.append(fut)

        for fut in futures:
            fut.result()

    def list_artifacts(self, path: Optional[str] = None) -> list[FileInfo]:
        dest_path = self.full_path(path)
        if not self._is_dir(dest_path):
            return []

        file_infos: list[FileInfo] = []
        for directory_entry in self.files_api.list_directory_contents(dest_path):
            relative_path = posixpath.relpath(directory_entry.path, self.root_path)
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
