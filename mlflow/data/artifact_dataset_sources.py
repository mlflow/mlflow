import json
import warnings

from typing import TypeVar, Any
from urllib.parse import urlparse

from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.store.artifact.artifact_repository_registry import _artifact_repository_registry


def register_artifact_dataset_sources():
    from mlflow.data.dataset_source import DatasetSource
    from mlflow.data.dataset_source_registry import dataset_source_registry

    registered_source_schemes = set()
    artifact_schemes_to_exclude = ["http", "https", "runs", "models", "mlflow-artifacts"]
    schemes_to_artifact_repos = _artifact_repository_registry.get_registered_artifact_repositories()
    for scheme, artifact_repo in schemes_to_artifact_repos.items():
        if scheme in artifact_schemes_to_exclude or scheme in registered_source_schemes:
            continue

        if "ArtifactRepository" in artifact_repo.__name__:
            # Artifact repository name is something like "LocalArtifactRepository",
            # "S3ArtifactRepository", etc. To preserve capitalization, strip ArtifactRepository
            # and replace it with DatasetSource
            dataset_source_name = artifact_repo.__name__.replace("ArtifactRepository", "DatasetSource")
        else:
            # Artifact repository name has some other form, e.g. "dbfs_artifact_repo_factory".
            # In this case, generate the name by capitalizing the first letter of the scheme and
            # appending ArtifactRepository
            scheme = str(scheme)

            def camelcase_scheme(scheme, separator):
                return "".join([part.capitalize() for part in scheme.split(separator)])

            if "-" in scheme:
                source_name_prefix = camelcase_scheme(scheme, "-")
            elif "_" in scheme:
                source_name_prefix = camelcase_scheme(scheme, "_")
            else:
                source_name_prefix = scheme.capitalize()

            dataset_source_name = scheme + "ArtifactRepository"

        try:
            registered_source_schemes.add(scheme)
            dataset_source = _create_dataset_source_for_artifact_repo(scheme=scheme, dataset_source_name=dataset_source_name, artifact_repo=artifact_repo)
            dataset_source_registry.register(dataset_source)
        except Exception as e:
            warnings.warn(f"Failed to register a dataset source for URIs with scheme '{scheme}': {e}", stacklevel=2)


def _create_dataset_source_for_artifact_repo(scheme: str, dataset_source_name: str, artifact_repo: ArtifactRepository):
    from mlflow.data.dataset_source import DatasetSource

    DatasetForArtifactRepoSourceType = TypeVar(dataset_source_name)

    class ArtifactRepoSource(DatasetSource):

        def __init__(self, uri: str):
            self.uri = uri

        @staticmethod
        def _get_source_type() -> str:
            if scheme:
                return scheme
            else:
                # An empty scheme indicates a file / directory on the local filesystem
                return "file"

        def download(self) -> str:
            return download_artifacts(self.uri)

        @staticmethod
        def _can_resolve(raw_source: str):
            if not isinstance(raw_source, str):
                return False

            try:
                parsed_source = urlparse(raw_source)
                return parsed_source.scheme == scheme
            except Exception:
                return False

        @classmethod
        def _resolve(cls, raw_source: str) -> DatasetForArtifactRepoSourceType:
            return cls(raw_source)

        def to_json(self):
            return json.dumps({
                "uri": self.uri,
            })

        @classmethod
        def _from_json(cls, source_json: str):
            parsed_json = json.loads(source_json)
            if not isinstance(parsed_json, dict):
                raise MlflowException(
                    f"Failed to parse {dataset_source_name} from JSON. Expected a JSON dictionary, but received: {source_json}",
                    INVALID_PARAMETER_VALUE
                )

            uri = parsed_json.get("uri")
            if uri is None:
                raise MlflowException(
                    f"Failed to parse {dataset_source_name} from JSON. Missing expected key: \"uri\"",
                    INVALID_PARAMETER_VALUE
                )

            return cls(uri=uri)

    setattr(ArtifactRepoSource, "__name__", dataset_source_name)
    setattr(ArtifactRepoSource, "__qualname__", dataset_source_name)
    return ArtifactRepoSource
