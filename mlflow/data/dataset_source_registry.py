import warnings
from typing import Any, List, Optional

import entrypoints

from mlflow.data.artifact_dataset_sources import register_artifact_dataset_sources
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.http_dataset_source import HTTPDatasetSource
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST


class DatasetSourceRegistry:
    def __init__(self):
        self.sources = []

    def register(self, source: DatasetSource):
        """Registers a DatasetSource for use with MLflow Tracking.

        Args:
            source: The DatasetSource to register.
        """
        self.sources.append(source)

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
                    "Failure attempting to register dataset constructor"
                    + f' "{entrypoint}": {exc}',
                    stacklevel=2,
                )

    def resolve(
        self, raw_source: Any, candidate_sources: Optional[List[DatasetSource]] = None
    ) -> DatasetSource:
        """Resolves a raw source object, such as a string URI, to a DatasetSource for use with
        MLflow Tracking.

        Args:
            raw_source: The raw source, e.g. a string like "s3://mybucket/path/to/iris/data" or a
                HuggingFace :py:class:`datasets.Dataset` object.
            candidate_sources: A list of DatasetSource classes to consider as potential sources
                when resolving the raw source. Subclasses of the specified candidate sources are
                also considered. If unspecified, all registered sources are considered.

        Raises:
            MlflowException: If no DatasetSource class can resolve the raw source.

        Returns:
            The resolved DatasetSource.
        """
        matching_sources = []
        for source in self.sources:
            if candidate_sources and not any(
                issubclass(source, candidate_src) for candidate_src in candidate_sources
            ):
                continue
            try:
                if source._can_resolve(raw_source):
                    matching_sources.append(source)
            except Exception as e:
                warnings.warn(
                    f"Failed to determine whether {source.__name__} can resolve source"
                    f" information for '{raw_source}'. Exception: {e}",
                    stacklevel=2,
                )
                continue

        if len(matching_sources) > 1:
            source_class_names_str = ", ".join([source.__name__ for source in matching_sources])
            warnings.warn(
                f"The specified dataset source can be interpreted in multiple ways:"
                f" {source_class_names_str}. MLflow will assume that this is a"
                f" {matching_sources[-1].__name__} source.",
                stacklevel=2,
            )

        for matching_source in reversed(matching_sources):
            try:
                return matching_source._resolve(raw_source)
            except Exception as e:
                warnings.warn(
                    f"Encountered an unexpected error while using {matching_source.__name__} to"
                    f" resolve source information for '{raw_source}'. Exception: {e}",
                    stacklevel=2,
                )
                continue

        raise MlflowException(
            f"Could not find a source information resolver for the specified"
            f" dataset source: {raw_source}.",
            RESOURCE_DOES_NOT_EXIST,
        )

    def get_source_from_json(self, source_json: str, source_type: str) -> DatasetSource:
        """Parses and returns a DatasetSource object from its JSON representation.

        Args:
            source_json: The JSON representation of the DatasetSource.
            source_type: The string type of the DatasetSource, which indicates how to parse the
                source JSON.
        """
        for source in reversed(self.sources):
            if source._get_source_type() == source_type:
                return source.from_json(source_json)

        raise MlflowException(
            f"Could not parse dataset source from JSON due to unrecognized"
            f" source type: {source_type}.",
            RESOURCE_DOES_NOT_EXIST,
        )


def register_dataset_source(source: DatasetSource):
    """Registers a DatasetSource for use with MLflow Tracking.

    Args:
        source: The DatasetSource to register.
    """
    _dataset_source_registry.register(source)


def resolve_dataset_source(
    raw_source: Any, candidate_sources: Optional[List[DatasetSource]] = None
) -> DatasetSource:
    """Resolves a raw source object, such as a string URI, to a DatasetSource for use with
    MLflow Tracking.

    Args:
        raw_source: The raw source, e.g. a string like "s3://mybucket/path/to/iris/data" or a
            HuggingFace :py:class:`datasets.Dataset` object.
        candidate_sources: A list of DatasetSource classes to consider as potential sources
            when resolving the raw source. Subclasses of the specified candidate
            sources are also considered. If unspecified, all registered sources
            are considered.

    Raises:
        MlflowException: If no DatasetSource class can resolve the raw source.

    Returns:
        The resolved DatasetSource.
    """
    return _dataset_source_registry.resolve(
        raw_source=raw_source, candidate_sources=candidate_sources
    )


def get_dataset_source_from_json(source_json: str, source_type: str) -> DatasetSource:
    """Parses and returns a DatasetSource object from its JSON representation.

    Args:
        source_json: The JSON representation of the DatasetSource.
        source_type: The string type of the DatasetSource, which indicates how to parse the
            source JSON.
    """
    return _dataset_source_registry.get_source_from_json(
        source_json=source_json, source_type=source_type
    )


def get_registered_sources() -> List[DatasetSource]:
    """Obtains the registered dataset sources.

    Returns:
        A list of registered dataset sources.

    """
    return _dataset_source_registry.sources


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
_dataset_source_registry.register(HTTPDatasetSource)
_dataset_source_registry.register_entrypoints()

try:
    from mlflow.data.huggingface_dataset_source import HuggingFaceDatasetSource

    _dataset_source_registry.register(HuggingFaceDatasetSource)
except ImportError:
    pass
try:
    from mlflow.data.spark_dataset_source import SparkDatasetSource

    _dataset_source_registry.register(SparkDatasetSource)
except ImportError:
    pass
try:
    from mlflow.data.delta_dataset_source import DeltaDatasetSource

    _dataset_source_registry.register(DeltaDatasetSource)
except ImportError:
    pass
try:
    from mlflow.data.code_dataset_source import CodeDatasetSource

    _dataset_source_registry.register(CodeDatasetSource)
except ImportError:
    pass
try:
    from mlflow.data.uc_volume_dataset_source import UCVolumeDatasetSource

    _dataset_source_registry.register(UCVolumeDatasetSource)
except ImportError:
    pass
