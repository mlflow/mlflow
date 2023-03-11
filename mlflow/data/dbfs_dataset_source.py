import json
import os

from typing import TypeVar, Any
from urllib.parse import urlparse

from mlflow.artifacts import download_artifacts
from mlflow.data.filesystem_dataset_source import FileSystemDatasetSource
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.databricks_utils import is_in_databricks_runtime
from mlflow.utils.uri import is_valid_dbfs_uri


DBFSDatasetSourceType = TypeVar("DBFSDatasetSourceType", bound="DBFSDatasetSource")


class DBFSDatasetSource(FileSystemDatasetSource):

    def __init__(self, uri):
        self._uri = uri

    @property
    def uri(self):
        """
        :return: The dbfs:/ URI of the dataset or the /dbfs FUSE path of the dataset.
        """
        return self._uri

    @staticmethod
    def _get_source_type() -> str:
        return "dbfs"

    def download(self) -> str:
        """
        :return: The downloaded source, e.g. a local filesystem path, a Spark
                 DataFrame, etc.
        """
        return download_artifacts(self._uri)

    @staticmethod
    def _can_resolve(raw_source: Any) -> bool:
        """
        :param raw_source: A dbfs:/ URI or a /dbfs FUSE path.
        :return: True if this DatsetSource can resolve the raw source, False otherwise.
        """
        if not isinstance(raw_source, str):
            return False

        try:
            if is_valid_dbfs_uri(raw_source):
                return True

            parsed_source = urlparse(raw_source)
            return is_in_databricks_runtime() and not parsed_source.scheme and os.path.normpath(parsed_source.path).startswith("/dbfs")
        except Exception:
            return False

    @classmethod
    def _resolve(cls, raw_source: Any) -> DBFSDatasetSourceType:
        """
        :param raw_source: A dbfs:/ URI or a /dbfs FUSE path.
        """
        return cls(raw_source)

    def to_json(self) -> str:
        """
        :return: A JSON string representation of the DBFSDatasetSourceType.
        """
        # TODO: Include workspace information in the source
        return json.dumps({
            "uri": self.uri
        })

    @classmethod
    def _from_json(cls, source_json: str) -> DBFSDatasetSourceType:
        """
        :param json: A JSON string representation of the FileSystemDatasetSource.
        """
        parsed_json = json.loads(source_json)
        if not isinstance(parsed_json, dict):
            raise MlflowException(
                f"Failed to parse DBFS dataset source from JSON. Expected a JSON dictionary, but received: {source_json}",
                INVALID_PARAMETER_VALUE,
            )

        uri = parsed_json.get("uri")
        if uri is None:
            raise MlflowException(
                f'Failed to parse DBFS dataset source from JSON. Missing expected key: "uri"',
                INVALID_PARAMETER_VALUE,
            )

        return cls(uri=uri)
