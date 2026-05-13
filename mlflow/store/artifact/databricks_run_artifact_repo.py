import re

from mlflow.store.artifact.databricks_tracking_artifact_repo import (
    DatabricksTrackingArtifactRepository,
)


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

    def _get_uri_regex(self) -> re.Pattern[str]:
        return self._URI_REGEX

    def _get_expected_uri_format(self) -> str:
        return "databricks/mlflow-tracking/<EXPERIMENT_ID>/<RUN_ID>"

    def _build_root_path(self, experiment_id: str, match: re.Match, relative_path: str) -> str:
        run_id = match.group("run_id")
        return f"/WorkspaceInternal/Mlflow/Artifacts/{experiment_id}/Runs/{run_id}{relative_path}"

    @staticmethod
    def is_run_uri(artifact_uri: str) -> bool:
        return bool(DatabricksRunArtifactRepository._URI_REGEX.search(artifact_uri))
