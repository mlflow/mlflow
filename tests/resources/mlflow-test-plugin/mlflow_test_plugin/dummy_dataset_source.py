import json

from typing import Any
from urllib.parse import urlparse

from mlflow.artifacts import download_artifacts
from mlflow.data.dataset_source import DatasetSource
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE


class DummyDatasetSource(DatasetSource):
    def __init__(self, uri):
        self._uri = uri

    @property
    def uri(self):
        return self._uri

    @staticmethod
    def _get_source_type() -> str:
        return "dummy"

    def download(self) -> str:
        # Ignore the "dummy" URI scheme and download the local path
        parsed_uri = urlparse(self._uri)
        return download_artifacts(parsed_uri.path)

    @staticmethod
    def _can_resolve(raw_source: Any) -> bool:
        if not isinstance(raw_source, str):
            return False

        try:
            parsed_source = urlparse(raw_source)
            return parsed_source.scheme == "dummy"
        except Exception:
            return False

    @classmethod
    def _resolve(cls, raw_source: Any) -> DatasetSource:
        return cls(raw_source)

    def to_json(self) -> str:
        return json.dumps({"uri": self.uri})

    @classmethod
    def _from_json(cls, source_json: str) -> DatasetSource:
        parsed_json = json.loads(source_json)
        if not isinstance(parsed_json, dict):
            raise MlflowException(
                f"Failed to parse dummy dataset source from JSON. Expected a JSON dictionary, but received: {source_json}",
                INVALID_PARAMETER_VALUE,
            )

        uri = parsed_json.get("uri")
        if uri is None:
            raise MlflowException(
                f'Failed to parse dummy dataset source from JSON. Missing expected key: "uri"',
                INVALID_PARAMETER_VALUE,
            )

        return cls(uri=uri)
