import logging
import re
from functools import cached_property
from typing import Optional

from mlflow.entities import FileInfo
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.store.artifact.databricks_artifact_repo import DatabricksArtifactRepository
from mlflow.store.artifact.databricks_sdk_artifact_repo import DatabricksSdkArtifactRepository

_logger = logging.getLogger(__name__)


class DatabricksLoggedModelArtifactRepository(ArtifactRepository):
    """
    Artifact repository for interacting with logged model artifacts in a Databricks workspace.
    If operations using the Databricks SDK fail for any reason, this repository automatically
    falls back to using the `DatabricksArtifactRepository`, ensuring operational resilience.
    """

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
        m = self._URI_REGEX.search(artifact_uri)
        if not m:
            raise MlflowException.invalid_parameter_value(
                f"Invalid artifact URI: {artifact_uri}. Expected URI of the form "
                f"databricks/mlflow-tracking/<EXP_ID>/logged_models/<MODEL_ID>"
            )
        experiment_id = m.group("experiment_id")
        model_id = m.group("model_id")
        relative_path = m.group("relative_path") or ""
        self.root_path = (
            f"/WorkspaceInternal/Mlflow/Artifacts/{experiment_id}/LoggedModels/{model_id}"
            f"{relative_path}"
        )
        self.databricks_sdk_repo = DatabricksSdkArtifactRepository(artifact_uri)

    @cached_property
    def databricks_artifact_repo(self) -> DatabricksArtifactRepository:
        return DatabricksArtifactRepository(self.artifact_uri)

    def log_artifact(self, local_file: str, artifact_path: Optional[str] = None) -> None:
        try:
            self.databricks_sdk_repo.log_artifact(local_file, artifact_path)
        except Exception:
            _logger.debug(
                (
                    "Failed to log artifact using Databricks SDK, falling back to "
                    "DatabricksArtifactRepository"
                ),
                exc_info=True,
            )
            return self.databricks_artifact_repo.log_artifact(local_file, artifact_path)

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        try:
            self.databricks_sdk_repo.log_artifacts(local_dir, artifact_path)
        except Exception:
            _logger.debug(
                (
                    "Failed to log artifacts using Databricks SDK, falling back to "
                    "DatabricksArtifactRepository"
                ),
                exc_info=True,
            )
            self.databricks_artifact_repo.log_artifacts(local_dir, artifact_path)

    def list_artifacts(self, path: Optional[str] = None) -> list[FileInfo]:
        try:
            return self.databricks_sdk_repo.list_artifacts(path)
        except Exception:
            _logger.debug(
                (
                    "Failed to list artifacts using Databricks SDK, falling back to "
                    "DatabricksArtifactRepository"
                ),
                exc_info=True,
            )
            return self.databricks_artifact_repo.list_artifacts(path)

    def _download_file(self, remote_file_path: str, local_path: str) -> None:
        try:
            self.databricks_sdk_repo._download_file(remote_file_path, local_path)
        except Exception:
            _logger.debug(
                (
                    "Failed to download file using Databricks SDK, falling back to "
                    "DatabricksArtifactRepository"
                ),
                exc_info=True,
            )
            self.databricks_artifact_repo._download_file(remote_file_path, local_path)
