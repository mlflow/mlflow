from urllib import parse
from collections import namedtuple
from mlflow.store.artifact.http_artifact_repo import HttpArtifactRepository
from mlflow.tracking._tracking_service.utils import get_tracking_uri


def _resolve_connection_params(artifact_uri):
    ParsedURI = namedtuple("ParsedURI", "scheme host port path")
    parsed_uri = parse.urlparse(artifact_uri)
    return ParsedURI(parsed_uri.scheme, parsed_uri.hostname, parsed_uri.port, parsed_uri.path)


class MlflowArtifactsRepository(HttpArtifactRepository):
    """Scheme wrapper around HttpArtifactRepository for mlflow-artifacts server functionality"""

    def __init__(self, artifact_uri):
        parsed = _resolve_connection_params(artifact_uri)
        tracking_uri = get_tracking_uri()
        resolved_artifacts_uri = (
            artifact_uri.replace("mlflow-artifacts:", f"{tracking_uri}")
            .replace(f"mlflow-artifacts:{parsed.host}", f"{tracking_uri}")
            .replace(f"mlflow-artifacts:{parsed.host}:{parsed.port}", f"{tracking_uri}")
        )

        super().__init__(resolved_artifacts_uri)
