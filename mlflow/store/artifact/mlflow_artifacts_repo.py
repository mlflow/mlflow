from mlflow.store.artifact.http_artifact_repo import HttpArtifactRepository
from mlflow.tracking._tracking_service.utils import get_tracking_uri


class MlflowArtifactsRepository(HttpArtifactRepository):
    """Scheme wrapper around HttpArtifactRepository for mlflow-artifacts server functionality"""

    def __init__(self, artifact_uri):
        tracking_uri = get_tracking_uri()
        resolved_artifacts_uri = artifact_uri.replace("mlflow-artifacts:", f"{tracking_uri}")

        super().__init__(resolved_artifacts_uri)
