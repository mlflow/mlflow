import re

from mlflow.store.artifact.databricks_tracking_artifact_repo import (
    DatabricksTrackingArtifactRepository,
)


class DatabricksLoggedModelArtifactRepository(DatabricksTrackingArtifactRepository):
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

    def _get_uri_regex(self) -> re.Pattern[str]:
        return self._URI_REGEX

    def _get_expected_uri_format(self) -> str:
        return "databricks/mlflow-tracking/<EXP_ID>/logged_models/<MODEL_ID>"

    def _build_root_path(self, experiment_id: str, match: re.Match, relative_path: str) -> str:
        model_id = match.group("model_id")
        return (
            f"/WorkspaceInternal/Mlflow/Artifacts/{experiment_id}/LoggedModels/{model_id}"
            f"{relative_path}"
        )

    @staticmethod
    def is_logged_model_uri(artifact_uri: str) -> bool:
        return bool(DatabricksLoggedModelArtifactRepository._URI_REGEX.search(artifact_uri))
