import os
import posixpath
import re
import tempfile

from typing import Any, Dict
from urllib.parse import urlparse

from mlflow.data.dataset_source import DatasetSource
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.rest_utils import cloud_storage_http_request, augmented_raise_for_status


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

        path = urlparse(self.url).path
        content_disposition = resp.headers.get("Content-Disposition")
        if content_disposition is not None and (
            file_name := next(re.finditer(r"filename=(.+)", content_disposition), None)
        ):
            # NB: If the filename is quoted, unquote it
            basename = file_name[1].strip("'\"")
        elif path is not None and len(posixpath.basename(path)) > 0:
            basename = posixpath.basename(path)
        else:
            basename = "dataset_source"

        if dst_path is None:
            dst_path = tempfile.mkdtemp()

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
