import posixpath
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import NotFound
from databricks.sdk.service.files import FilesAPI

from mlflow.entities import FileInfo
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository


class DatabricksLoggedModelArtifactRepository(ArtifactRepository):
    # Matches URIs of the form:
    # databricks/mlflow-tracking/<experiment_id>/logged_models/<model_id>/<relative_path>
    _URI_REGEX = re.compile(
        r"databricks/mlflow-tracking/(?P<experiment_id>[^/]+)/logged_models/(?P<model_id>[^/]+)(?P<relative_path>/.*)?$"
    )

    @staticmethod
    def is_logged_model_uri(artifact_uri: str) -> bool:
        return bool(DatabricksLoggedModelArtifactRepository._URI_REGEX.search(artifact_uri))

    def __init__(self, artifact_uri: str) -> None:
        super().__init__(artifact_uri)
        self.wc = WorkspaceClient()
        m = self._URI_REGEX.search(artifact_uri)
        if not m:
            raise MlflowException.invalid_parameter_value(
                f"Invalid artifact URI: {artifact_uri}. Expected URI of the form "
                f"databricks/mlflow-tracking/<EXP_ID>/logged_models/<MODEL_ID>"
            )
        experiment_id = m.group("experiment_id")
        model_id = m.group("model_id")
        relative_path = m.group("relative_path") or ""
        self.root_path = f"/Mlflow/Artifacts/{experiment_id}/LoggedModels/{model_id}{relative_path}"

    @property
    def files_api(self) -> FilesAPI:
        return self.wc.files

    def _is_dir(self, path: str) -> bool:
        try:
            self.files_api.get_directory_metadata(path)
        except NotFound:
            return False
        return True

    def path(self, artifact_path: str) -> str:
        """
        Construct the full path to the artifact, given the artifact's relative path.
        """
        return f"{self.root_path}/{artifact_path}" if artifact_path else self.root

    def log_artifact(self, local_file: str, artifact_path: Optional[str] = None) -> None:
        with open(local_file, "rb") as f:
            self.files_api.upload(self.path(artifact_path), f, overwrite=True)

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        futures = []
        with ThreadPoolExecutor() as executor:
            for f in Path(local_dir).rglob("*"):
                if f.is_file():
                    fut = executor.submit(self.log_artifact, f, artifact_path)
                    futures.append(fut)

        for fut in futures:
            fut.result()

    def list_artifacts(self, path: Optional[str] = None) -> list[FileInfo]:
        file_infos: list[FileInfo] = []

        dest_path = self.path(path)
        if not self._is_dir(dest_path):
            return file_infos

        for directory_entry in self.files_api.list_directory_contents(dest_path):
            relative_path = posixpath.relpath(directory_entry.path, self.root)
            file_infos.append(
                FileInfo(
                    path=relative_path,
                    is_dir=directory_entry.is_directory,
                    file_size=directory_entry.file_size,
                )
            )

        return sorted(file_infos, key=lambda f: f.path)

    def _download_file(self, remote_file_path: str, local_path: str) -> None:
        download_resp = self.files_api.download(self.path(remote_file_path))
        with open(local_path, "wb") as f:
            while chunk := download_resp.contents.read(10 * 1024 * 1024):
                f.write(chunk)
