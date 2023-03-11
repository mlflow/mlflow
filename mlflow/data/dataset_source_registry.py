import warnings
import entrypoints
from typing import Any, List

from mlflow.exceptions import MlflowException
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.dbfs_dataset_source import DBFSDatasetSource
from mlflow.data.huggingface_dataset_source import HuggingFaceDatasetSource
from mlflow.data.artifact_dataset_sources import register_artifact_dataset_sources
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST


class DatasetSourceRegistry:
    def __init__(self):
        self._sources = {}

    def register(self, source: DatasetSource):
        """
        Registers the DatasetSource
        """
        self._sources[source._get_source_type] = source

    def register_entrypoints(self):
        """
        Registers dataset sources defined as Python entrypoints. For reference, see
        https://mlflow.org/docs/latest/plugins.html#defining-a-plugin.
        """
        for entrypoint in entrypoints.get_group_all("mlflow.dataset_source"):
            try:
                self.register(entrypoint.load())
            except (AttributeError, ImportError) as exc:
                warnings.warn(
                    f'Failure attempting to register dataset source with source type "{entrypoint.source_type}": {exc}',
                    stacklevel=2,
                )

    def resolve(
        self, raw_source: Any, candidate_sources: List[DatasetSource] = None
    ) -> DatasetSource:
        """
        :param raw_source: The raw source, e.g. a string like "s3://mybucket/path/to/iris/data".
        """
        matching_sources = []
        for source in self._sources.values():
            if candidate_sources and not any(
                [issubclass(source, candidate_src) for candidate_src in candidate_sources]
            ):
                continue
            if source._can_resolve(raw_source):
                matching_sources.append(source)

        if len(matching_sources) > 1:
            source_types_str = ", ".join([source._get_source_type() for source in matching_sources])
            warnings.warn(
                f"The specified dataset source can be interpreted in multiple ways: {source_types_str}. MLflow will assume that this is a {matching_sources[0]._get_source_type()} source",
                stacklevel=2,
            )

        if len(matching_sources) >= 1:
            return matching_sources[-1]._resolve(raw_source)
        else:
            # TODO: Support passing in dataset info to add context to this error
            raise MlflowException(
                f"Could not find a source information resolver for the specified dataset source: {raw_source}",
                RESOURCE_DOES_NOT_EXIST,
            )

    def get_source_from_json(self, source_json: str, source_type: str) -> DatasetSource:
        source = self._sources.get(source_type)
        if source is not None:
            return source._from_json(source_json)
        else:
            raise MlflowException(
                f"Could not parse dataset source from JSON due to unrecognized source type: {source_type}",
                RESOURCE_DOES_NOT_EXIST,
            )


def register_dataset_source(source: DatasetSource):
    """
    Registers a DatasetSource for use with MLflow.

    :param source: The DatasetSource to register.
    """
    _dataset_source_registry.register(source)


def resolve_dataset_source(
    raw_source: Any, candidate_sources: List[DatasetSource] = None
) -> DatasetSource:
    """
    Resolves a raw source object, such as a string URI, to a DatasetSource for use with MLflow.

    :param raw_source: The raw source, e.g. a string like "s3://mybucket/path/to/iris/data" or a
                       HuggingFace `datasets.Dataset` object.
    :param candidate_sources: A list of DatasetSource classes to consider as potential sources
                              when resolving the raw source. Subclasses of the specified candidate
                              sources are also considered. If unspecified, all registered sources
                              are considered.
    :throws: MlflowException if no DatasetSource class can resolve the raw source.
    :return: The resolved DatasetSource.
    """
    return _dataset_source_registry.resolve(
        raw_source=raw_source, candidate_sources=candidate_sources
    )


def get_dataset_source_from_json(source_json: str, source_type: str) -> DatasetSource:
    """
    Parses and returns a DatasetSource object from its JSON representation.

    :param source_json: The JSON representation of the DatasetSource.
    :param source_type: The string type of the DatasetSource, which indicates how to parse the
                        source JSON.
    """
    return _dataset_source_registry.get_source_from_json(
        source_json=source_json, source_type=source_type
    )


# NB: The ordering here is important. The last dataset source to be registered takes precedence
# when resolving dataset information for a raw source (e.g. a string like "s3://mybucket/my/path").
# Dataset sources derived from artifact repositories are the most generic / provide the most
# general information about dataset source locations, so they are registered first. More specific
# source information is provided by specialized dataset platform sources like
# HuggingFaceDatasetSource, so these sources are registered next. Finally, externally-defined
# dataset sources are registered last because externally-defined behavior should take precedence
# over any internally-defined generic behavior
_dataset_source_registry = DatasetSourceRegistry()
register_artifact_dataset_sources()
_dataset_source_registry.register(DBFSDatasetSource)
_dataset_source_registry.register(HuggingFaceDatasetSource)
_dataset_source_registry.register_entrypoints()
