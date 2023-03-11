from abc import abstractmethod
from typing import TypeVar, Any, Dict


DatasetSourceType = TypeVar("DatasetSourceType", bound="DatasetSource")


class DatasetSource:
    @staticmethod
    @abstractmethod
    def _get_source_type() -> str:
        """
        :return: A string representing the source type of the dataset,
                 e.g. "s3"
        """

    @abstractmethod
    def download(self) -> Any:
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
        :return: True if this DatsetSource can resolve the raw source, False otherwise.
        """

    @classmethod
    @abstractmethod
    def _resolve(cls, raw_source: Any) -> DatasetSourceType:
        """
        :param raw_source: The raw source, e.g. a string like
                           "s3://mybucket/path/to/iris/data".
        """

    @abstractmethod
    def to_dict(self) -> str:
        """
        :return: A string dictionary representation of the DatasetSource.
        """

    @classmethod
    @abstractmethod
    def _from_dict(cls, source_dict: Dict[str, str]) -> DatasetSourceType:
        """
        :param source_dict: A string dictionary representation of the DatasetSource.
        """
