import json

from abc import abstractmethod
from typing import Any, Dict

from mlflow.utils.annotations import experimental


@experimental
class DatasetSource:
    """
    Represents the source of a dataset used in MLflow Tracking, providing information such as
    cloud storage location, delta table name / version, etc.
    """

    @staticmethod
    @abstractmethod
    def _get_source_type() -> str:
        """
        Obtains a string representing the source type of the dataset.

        :return: A string representing the source type of the dataset, e.g. "s3", "delta_table", ...
        """

    @abstractmethod
    def load(self) -> Any:
        """
        Loads files / objects referred to by the DatasetSource. For example, depending on the type
        of :py:class:`DatasetSource <mlflow.data.dataset_source.DatasetSource>`, this may download
        source CSV files from S3 to the local filesystem, load a source Delta Table as a Spark
        DataFrame, etc.

        :return: The downloaded source, e.g. a local filesystem path, a Spark DataFrame, etc.
        """

    @staticmethod
    @abstractmethod
    def _can_resolve(raw_source: Any) -> bool:
        """
        Determines whether this type of DatasetSource can be resolved from a specified raw source
        object. For example, an S3DatasetSource can be resolved from an S3 URI like
        "s3://mybucket/path/to/iris/data" but not from an Azure Blob Storage URI like
        "wasbs:/account@host.blob.core.windows.net".

        :param raw_source: The raw source, e.g. a string like "s3://mybucket/path/to/iris/data".
        :return: True if this DatsetSource can resolve the raw source, False otherwise.
        """

    @classmethod
    @abstractmethod
    def _resolve(cls, raw_source: Any) -> "DatasetSource":
        """
        Constructs an instance of the DatasetSource from a raw source object, such as a
        string URI like "s3://mybucket/path/to/iris/data" or a delta table identifier
        like "my.delta.table@2".

        :param raw_source: The raw source, e.g. a string like "s3://mybucket/path/to/iris/data".
        :return: A DatasetSource instance derived from the raw_source.
        """

    @abstractmethod
    def _to_dict(self) -> Dict[str, Any]:
        """
        Obtains a JSON-compatible dictionary representation of the DatasetSource.

        :return: A JSON-compatible dictionary representation of the DatasetSource.
        """

    def to_json(self) -> str:
        """
        Obtains a JSON string representation of the
        :py:class:`DatasetSource <mlflow.data.dataset_source.DatasetSource>`.

        :return: A JSON string representation of the
                 :py:class:`DatasetSource <mlflow.data.dataset_source.DatasetSource>`.

        """
        return json.dumps(self._to_dict())

    @classmethod
    @abstractmethod
    def _from_dict(cls, source_dict: Dict[Any, Any]) -> "DatasetSource":
        """
        Constructs an instance of the DatasetSource from a dictionary representation.

        :param source_dict: A dictionary representation of the DatasetSource.
        :return: A DatasetSource instance.
        """

    @classmethod
    def from_json(cls, source_json: str) -> "DatasetSource":
        """
        Constructs an instance of the DatasetSource from a JSON string representation.

        :param source_dict: A JSON string representation of the DatasetSource.
        :return: A DatasetSource instance.
        """
        return cls._from_dict(json.loads(source_json))
