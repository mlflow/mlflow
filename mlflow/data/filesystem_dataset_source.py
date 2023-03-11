from abc import abstractmethod
from typing import TypeVar, Any

from mlflow.data.dataset_source import DatasetSource


FileSystemDatasetSourceType = TypeVar("FileSystemDatasetSourceType", bound="FileSystemDatasetSource")


class FileSystemDatasetSource(DatasetSource):

    @property
    @abstractmethod
    def uri(self):
        """
        :return: The URI referring to the dataset filesystem location,
                 e.g "s3://mybucket/path/to/mydataset", "/tmp/path/to/my/dataset" etc.
        """

    @staticmethod
    @abstractmethod
    def _get_source_type() -> str:
        """
        :return: A string describing the filesystem containing the dataset, e.g. "local", "s3", ...
        """

    @abstractmethod
    def download(self) -> str:
        """
        Downloads the dataset to the local filesystem.

        :return: The path to the downloaded dataset on the local filesystem.
        """

    @staticmethod
    @abstractmethod
    def _can_resolve(raw_source: Any) -> bool:
        """
        :param raw_source: The raw source, e.g. a string like
                           "s3://mybucket/path/to/iris/data".
        :return: True if this DatsetSource can resolve the raw source, False otherwise.
        """

    @classmethod
    @abstractmethod
    def _resolve(cls, raw_source: Any) -> FileSystemDatasetSourceType:
        """
        :param raw_source: The raw source, e.g. a string like "s3://mybucket/path/to/iris/data".
        """

    @abstractmethod
    def to_json(self) -> str:
        """
        :return: A JSON string representation of the FileSystemDatasetSource.
        """

    @classmethod
    @abstractmethod
    def _from_json(cls, source_json: str) -> FileSystemDatasetSourceType:
        """
        :param json: A JSON string representation of the FileSystemDatasetSource.
        """
