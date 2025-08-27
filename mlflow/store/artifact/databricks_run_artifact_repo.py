import logging
import re

from mlflow.store.artifact.databricks_tracking_artifact_repo import (
    DatabricksTrackingArtifactRepository,
)

_logger = logging.getLogger(__name__)


class DatabricksRunArtifactRepository(DatabricksTrackingArtifactRepository):
    """
    Artifact repository for interacting with run artifacts in a Databricks workspace.
    If operations using the Databricks SDK fail for any reason, this repository automatically
    falls back to using the `DatabricksArtifactRepository`, ensuring operational resilience.
    """

    # Matches URIs of the form:
    # databricks/mlflow-tracking/<experiment_id>/<run_id>/<relative_path>
    # But excludes trace URIs (run_id starting with "tr-") and logged_models
    _URI_REGEX = re.compile(
        r"databricks/mlflow-tracking/(?P<experiment_id>[^/]+)/(?P<run_id>(?!tr-|logged_models)[^/]+)(?P<relative_path>/.*)?$"
    )

    def _get_uri_regex(self) -> re.Pattern:
        _logger.info("[RUN_REPO_DEBUG] Returning URI regex for run artifacts")
        return self._URI_REGEX

    def _get_expected_uri_format(self) -> str:
        _logger.info(
            "[RUN_REPO_DEBUG] Expected URI format: databricks/mlflow-tracking/<EXP_ID>/<RUN_ID>"
        )
        return "databricks/mlflow-tracking/<EXP_ID>/<RUN_ID>"

    def _build_root_path(self, experiment_id: str, match: re.Match, relative_path: str) -> str:
        run_id = match.group("run_id")
        root_path = (
            f"/WorkspaceInternal/Mlflow/Artifacts/{experiment_id}/Runs/{run_id}{relative_path}"
        )
        _logger.info(
            f"[RUN_REPO_DEBUG] Built root path: experiment_id={experiment_id}, "
            f"run_id={run_id}, relative_path={relative_path} -> {root_path}"
        )
        return root_path

    @staticmethod
    def is_run_uri(artifact_uri: str) -> bool:
        is_run = bool(DatabricksRunArtifactRepository._URI_REGEX.search(artifact_uri))
        _logger.info(f"[RUN_REPO_DEBUG] is_run_uri check for '{artifact_uri}': {is_run}")
        return is_run
