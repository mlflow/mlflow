import os
import re
from typing import Any, Dict
from urllib.parse import urlparse

from mlflow.data.dataset_source import DatasetSource
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.file_utils import create_tmp_dir
from mlflow.utils.rest_utils import augmented_raise_for_status, cloud_storage_http_request


def _is_path(filename: str) -> bool:
    """
    Return True if `filename` is a path, False otherwise. For example,
    "foo/bar" is a path, but "bar" is not.
    """
    return os.path.basename(filename) != filename


class HTTPDatasetSource(DatasetSource):
    """
    Represents the source of a dataset stored at a web location and referred to
    by an HTTP or HTTPS URL.
    """

    def __init__(self, url):
        self._url = url

    @property
    def url(self):
        """
        The HTTP/S URL referring to the dataset source location.

        :return: The HTTP/S URL referring to the dataset source location.
        """
        return self._url

    @staticmethod
    def _get_source_type() -> str:
        return "http"

    def _extract_filename(self, response) -> str:
        """
        Extracts a filename from the Content-Disposition header or the URL's path.
        """
        if content_disposition := response.headers.get("Content-Disposition"):
            for match in re.finditer(r"filename=(.+)", content_disposition):
                filename = match[1].strip("'\"")
                if _is_path(filename):
                    raise MlflowException.invalid_parameter_value(
                        f"Invalid filename in Content-Disposition header: {filename}. "
                        "It must be a file name, not a path."
                    )
                return filename

        # Extract basename from URL if no valid filename in Content-Disposition
        return os.path.basename(urlparse(self.url).path)

    def load(self, dst_path=None) -> str:
        """
        Downloads the dataset source to the local filesystem.

        :param dst_path: Path of the local filesystem destination directory to which to download the
                         dataset source. If the directory does not exist, it is created. If
                         unspecified, the dataset source is downloaded to a new uniquely-named
                         directory on the local filesystem.
        :return: The path to the downloaded dataset source on the local filesystem.
        """
        resp = cloud_storage_http_request(
            method="GET",
            url=self.url,
            stream=True,
        )
        augmented_raise_for_status(resp)

        basename = self._extract_filename(resp)

        if not basename:
            basename = "dataset_source"

        if dst_path is None:
            dst_path = create_tmp_dir()

        dst_path = os.path.join(dst_path, basename)
        with open(dst_path, "wb") as f:
            chunk_size = 1024 * 1024  # 1 MB
            for chunk in resp.iter_content(chunk_size=chunk_size):
                f.write(chunk)

        return dst_path

    @staticmethod
    def _can_resolve(raw_source: Any) -> bool:
        """
        :param raw_source: The raw source, e.g. a string like "http://mysite/mydata.tar.gz".
        :return: True if this DatsetSource can resolve the raw source, False otherwise.
        """
        if not isinstance(raw_source, str):
            return False

        try:
            parsed_source = urlparse(str(raw_source))
            return parsed_source.scheme in ["http", "https"]
        except Exception:
            return False

    @classmethod
    def _resolve(cls, raw_source: Any) -> "HTTPDatasetSource":
        """
        :param raw_source: The raw source, e.g. a string like "http://mysite/mydata.tar.gz".
        """
        return HTTPDatasetSource(raw_source)

    def _to_dict(self) -> Dict[Any, Any]:
        """
        :return: A JSON-compatible dictionary representation of the HTTPDatasetSource.
        """
        return {
            "url": self.url,
        }

    @classmethod
    def _from_dict(cls, source_dict: Dict[Any, Any]) -> "HTTPDatasetSource":
        """
        :param source_dict: A dictionary representation of the HTTPDatasetSource.
        """
        url = source_dict.get("url")
        if url is None:
            raise MlflowException(
                'Failed to parse HTTPDatasetSource. Missing expected key: "url"',
                INVALID_PARAMETER_VALUE,
            )

        return cls(url=url)
