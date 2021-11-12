import os
import requests
import posixpath

import mlflow
from mlflow.entities import FileInfo
from mlflow.store.artifact.http_artifact_repo import HttpArtifactRepository
from mlflow.utils.file_utils import relative_path_to_artifact_path
import urllib.parse


def _resolve_connection_params(artifact_uri):
    # Taken from mlflow/store/artifact/hdfs_artifact_repo.py
    parsed = urllib.parse.urlparse(artifact_uri)

    return parsed.scheme, parsed.hostname, parsed.port, parsed.path


class MlflowArtifactsRepository(HttpArtifactRepository):
    def __init__(self, artifact_uri):
        # Not sure if this really works
        scheme, host, port, path = _resolve_connection_params(artifact_uri)
        tracking_uri = mlflow.tracking.get_tracking_uri()
        resolved_artifact_uri = artifact_uri.replace(
            "mlflow-artifacts:", f"{tracking_uri}/api/2.0/mlflow-artifacts/artifacts/{path}"
        )
        super().__init__(resolved_artifact_uri)
