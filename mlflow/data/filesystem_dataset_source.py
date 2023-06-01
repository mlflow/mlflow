from abc import abstractmethod
from typing import Any, Dict

from mlflow.data.dataset_source import DatasetSource
from mlflow.utils.annotations import experimental


@experimental
class FileSystemDatasetSource(DatasetSource):
    """
    Represents the source of a dataset stored on a filesystem, e.g. a local UNIX filesystem,
    blob storage services like S3, etc.
    """

    @property
    @abstractmethod
    def uri(self):
        """
        The URI referring to the dataset source filesystem location.

        :return: The URI referring to the dataset source filesystem location,
                 e.g "s3://mybucket/path/to/mydataset", "/tmp/path/to/my/dataset" etc.
        """

    @staticmethod
    @abstractmethod
    def _get_source_type() -> str:
        """
        :return: A string describing the filesystem containing the dataset, e.g. "local", "s3", ...
        """

    @abstractmethod
    def load(self, dst_path=None) -> str:
        """
        Downloads the dataset source to the local filesystem.

        :param dst_path: Path of the local filesystem destination directory to which to download the
                         dataset source. If the directory does not exist, it is created. If
                         unspecified, the dataset source is downloaded to a new uniquely-named
                         directory on the local filesystem, unless the dataset source already
                         exists on the local filesystem, in which case its local path is returned
                         directly.
        :return: The path to the downloaded dataset source on the local filesystem.
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
    def _resolve(cls, raw_source: Any) -> "FileSystemDatasetSource":
        """
        :param raw_source: The raw source, e.g. a string like "s3://mybucket/path/to/iris/data".
        """

    @abstractmethod
    def _to_dict(self) -> Dict[Any, Any]:
        """
        :return: A JSON-compatible dictionary representation of the FileSystemDatasetSource.
        """

    @classmethod
    @abstractmethod
    def _from_dict(cls, source_dict: Dict[Any, Any]) -> "FileSystemDatasetSource":
        """
        :param source_dict: A dictionary representation of the FileSystemDatasetSource.
        """
