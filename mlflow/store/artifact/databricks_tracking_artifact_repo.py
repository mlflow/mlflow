import logging
import re
from abc import ABC, abstractmethod

from mlflow.entities import FileInfo
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.store.artifact.databricks_artifact_repo import DatabricksArtifactRepository
from mlflow.store.artifact.databricks_sdk_artifact_repo import DatabricksSdkArtifactRepository

_logger = logging.getLogger(__name__)


_FALLBACK_MESSAGE_TEMPLATE = (
    "Failed to perform {operation} operation using Databricks SDK, falling back to "
    "DatabricksArtifactRepository. Original error: %s"
)


class DatabricksTrackingArtifactRepository(ArtifactRepository, ABC):
    """
    Base artifact repository for interacting with tracking artifacts in a Databricks workspace.
    If operations using the Databricks SDK fail for any reason, this repository automatically
    falls back to using the `DatabricksArtifactRepository`, ensuring operational resilience.

    This is an abstract base class that should be extended by specific tracking artifact
    repositories (e.g., for runs, logged models, etc.).
    """

    def __init__(
        self, artifact_uri: str, tracking_uri: str | None = None, registry_uri: str | None = None
    ) -> None:
        super().__init__(artifact_uri, tracking_uri, registry_uri)
        m = self._get_uri_regex().search(artifact_uri)
        if not m:
            raise MlflowException.invalid_parameter_value(
                f"Invalid artifact URI: {artifact_uri}. Expected URI of the form "
                f"{self._get_expected_uri_format()}"
            )
        experiment_id = m.group("experiment_id")
        relative_path = m.group("relative_path") or ""
        root_path = self._build_root_path(experiment_id, m, relative_path)
        self.databricks_sdk_repo = DatabricksSdkArtifactRepository(root_path)
        self.databricks_artifact_repo = DatabricksArtifactRepository(artifact_uri)

    @abstractmethod
    def _get_uri_regex(self) -> re.Pattern[str]:
        """Return the regex pattern for matching URIs of this type."""

    @abstractmethod
    def _get_expected_uri_format(self) -> str:
        """Return a description of the expected URI format."""

    @abstractmethod
    def _build_root_path(self, experiment_id: str, match: re.Match, relative_path: str) -> str:
        """Build the root path for the Databricks SDK repository."""

    def log_artifact(self, local_file: str, artifact_path: str | None = None) -> None:
        try:
            self.databricks_sdk_repo.log_artifact(local_file, artifact_path)
        except Exception as e:
            _logger.debug(
                _FALLBACK_MESSAGE_TEMPLATE.format(operation="log_artifact") % str(e),
                exc_info=True,
            )
            self.databricks_artifact_repo.log_artifact(local_file, artifact_path)

    def log_artifacts(self, local_dir: str, artifact_path: str | None = None) -> None:
        try:
            self.databricks_sdk_repo.log_artifacts(local_dir, artifact_path)
        except Exception as e:
            _logger.debug(
                _FALLBACK_MESSAGE_TEMPLATE.format(operation="log_artifacts") % str(e),
                exc_info=True,
            )
            self.databricks_artifact_repo.log_artifacts(local_dir, artifact_path)

    def list_artifacts(self, path: str | None = None) -> list[FileInfo]:
        try:
            return self.databricks_sdk_repo.list_artifacts(path)
        except Exception as e:
            _logger.debug(
                _FALLBACK_MESSAGE_TEMPLATE.format(operation="list_artifacts") % str(e),
                exc_info=True,
            )
            return self.databricks_artifact_repo.list_artifacts(path)

    def _download_file(self, remote_file_path: str, local_path: str) -> None:
        try:
            self.databricks_sdk_repo._download_file(remote_file_path, local_path)
        except Exception as e:
            _logger.debug(
                _FALLBACK_MESSAGE_TEMPLATE.format(operation="download_file") % str(e),
                exc_info=True,
            )
            self.databricks_artifact_repo._download_file(remote_file_path, local_path)
