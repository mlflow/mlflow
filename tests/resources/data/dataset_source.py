from typing import Any
from urllib.parse import urlparse

from mlflow.artifacts import download_artifacts
from mlflow.data.dataset_source import DatasetSource
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE


class SampleDatasetSource(DatasetSource):
    def __init__(self, uri):
        self._uri = uri

    @property
    def uri(self):
        return self._uri

    @staticmethod
    def _get_source_type() -> str:
        return "test"

    def load(self) -> str:
        # Ignore the "test" URI scheme and download the local path
        parsed_uri = urlparse(self._uri)
        return download_artifacts(parsed_uri.path)

    @staticmethod
    def _can_resolve(raw_source: Any) -> bool:
        if not isinstance(raw_source, str):
            return False

        try:
            parsed_source = urlparse(raw_source)
            return parsed_source.scheme == "test"
        except Exception:
            return False

    @classmethod
    def _resolve(cls, raw_source: Any) -> DatasetSource:
        return cls(raw_source)

    def to_dict(self) -> dict[Any, Any]:
        return {"uri": self.uri}

    @classmethod
    def from_dict(cls, source_dict: dict[Any, Any]) -> DatasetSource:
        uri = source_dict.get("uri")
        if uri is None:
            raise MlflowException(
                'Failed to parse dummy dataset source. Missing expected key: "uri"',
                INVALID_PARAMETER_VALUE,
            )

        return cls(uri=uri)
