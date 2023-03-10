from abc import abstractmethod
from typing import TypeVar, Any


DatasetSourceType = TypeVar("DatasetSourceType", bound="DatasetSource")


class DatasetSource:

    @staticmethod
    @abstractmethod
    def source_type() -> str:
        """
        :return: A string representing the source type of the dataset,
                 e.g. "s3"
        """

    @abstractmethod
    def download() -> Any:
        """
        :return: The downloaded source, e.g. a local filesystem path, a Spark
                 DataFrame, etc.
        """


    @staticmethod
    @abstractmethod
    def can_resolve(raw_source: Any) -> bool:
        """
        :param raw_source: The raw source, e.g. a string like
                           "s3://mybucket/path/to/iris/data".
        :return: True if this resolver can resolve the source to a
        """

    @classmethod
    @abstractmethod
    def resolve(cls, raw_source: Any) -> DatasetSourceType:
        """
        :param raw_source: The raw source, e.g. a string like
                           "s3://mybucket/path/to/iris/data".
        """

    @abstractmethod
    def to_json() -> str:
        """
        :return: A JSON string representation of the DatasetSource.
        """

    @classmethod
    @abstractmethod
    def from_json(cls, json: str) -> DatasetSourceType:
        """
        :param json: A JSON string representation of the DatasetSource.
        """
