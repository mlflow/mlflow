from abc import abstractmethod
from typing import TypeVar, Any

from mlflow.data.filesystem_dataset_source import FileSystemDatasetSource


DBFSDatasetSourceType = TypeVar("DBFSDatasetSourceType", bound="DBFSDatasetSource")


class DBFSDatasetSource(FileSystemDatasetSource):

    @property
    @abstractmethod
    def uri(self):
        """
        :return: The location of the dataset within /dbfs.
        """

    @staticmethod
    @abstractmethod
    def _get_source_type() -> str:
        return "dbfs_fuse"

    @abstractmethod
    def download() -> Any:
        """
        :return: The downloaded source, e.g. a local filesystem path, a Spark
                 DataFrame, etc.
        """

    @staticmethod
    @abstractmethod
    def _can_resolve(raw_source: Any) -> bool:
        """
        :param raw_source: The raw source, e.g. a string like
                           "s3://mybucket/path/to/iris/data".
        :return: True if this resolver can resolve the source to a
        """

    @classmethod
    @abstractmethod
    def _resolve(cls, raw_source: Any) -> FileSystemDatasetSourceType:
        """
        :param raw_source: The raw source, e.g. a string like "s3://mybucket/path/to/iris/data".
        """

    @abstractmethod
    def to_json() -> str:
        """
        :return: A JSON string representation of the FileSystemDatasetSource.
        """

    @classmethod
    @abstractmethod
    def _from_json(cls, source_json: str) -> FileSystemDatasetSourceType:
        """
        :param json: A JSON string representation of the FileSystemDatasetSource.
        """
