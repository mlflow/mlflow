import json
import warnings

from typing import TypeVar, Any, Union
from urllib.parse import urlparse

from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.store.artifact.artifact_repository_registry import _artifact_repository_registry

from mlflow.data.dataset_source import DatasetSource


class HuggingFace(DatasetSource):
    def __init__(self, uri: str, split: str = None):
        self.uri = uri
        self.split = split

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
        return json.dumps(
            {
                "uri": self.uri,
            }
        )

    @classmethod
    def _from_json(cls, source_json: str):
        parsed_json = json.loads(source_json)
        if not isinstance(parsed_json, dict):
            raise MlflowException(
                f"Failed to parse {dataset_source_name} from JSON. Expected a JSON dictionary, but received: {source_json}",
                INVALID_PARAMETER_VALUE,
            )

        uri = parsed_json.get("uri")
        if uri is None:
            raise MlflowException(
                f'Failed to parse {dataset_source_name} from JSON. Missing expected key: "uri"',
                INVALID_PARAMETER_VALUE,
            )

        return cls(uri=uri)

setattr(ArtifactRepoSource, "__name__", dataset_source_name)
setattr(ArtifactRepoSource, "__qualname__", dataset_source_name)
return ArtifactRepoSource
