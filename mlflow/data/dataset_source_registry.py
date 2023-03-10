from typing import Any

from mlflow.exceptions import MlflowException
from mlflow.data.dataset_source import DatasetSource
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE


class DatasetSourceRegistry:

    def __init__(self):
        self._sources = {}

    def register(self, source: DatasetSource):
        """
        Registers the DatasetSource
        """
        self._sources[source.source_type] = source

    def register_entrypoints(self):
        """
        Registers dataset sources defined as Python entrypoints. For reference, see
        https://mlflow.org/docs/latest/plugins.html#defining-a-plugin.
        """

    def resolve(self, raw_source: Any) -> DatasetSource:
        """
        :param raw_source: The raw source, e.g. a string like
                           "s3://mybucket/path/to/iris/data".
        """
        for source in self._sources.values():
            if source.can_resolve(raw_source):
                return source.resolve(raw_source)

    def get_source_from_json(self, source_json: str, source_type: str) -> DatasetSource:
        source = self._sources.get(source_type)
        if source is not None:
            return source.from_json(source_json)
        else:
            raise MlflowException(
                f"Could not parse dataset source from JSON due to unrecognized source type: {source_type}",
                INVALID_PARAMETER_VALUE
            )


_dataset_source_registry = DatasetSourceRegistry()
# _dataset_source_registry.register(DeltaTableDatasetSource)
# _dataset_source_registry.register(HuggingFaceHubDatasetSource)
# _dataset_source_registry.register(S3DatasetSource)
# _dataset_source_registry.register_entrypoints()


def resolve_dataset_source(self, raw_source: Any) -> DatasetSource:
    return _dataset_source_registry.resolve(raw_source)


def get_dataset_source_from_json(self, source_json: str, source_type: str) -> DatasetSource:
    return _dataset_source_registry.get_source_from_json(source_json=source_json, source_type=source_type)
