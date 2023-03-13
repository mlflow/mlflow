from typing import Any, Dict, Optional
from urllib.parse import urlparse

import pytest

from mlflow.artifacts import download_artifacts
from mlflow.data.dataset import Dataset 
from mlflow.data.dataset_registry import DatasetRegistry
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.dataset_source_registry import DatasetSourceRegistry
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

#
# class TestDatasetSource(DatasetSource):
#     def __init__(self, uri):
#         self._uri = uri
#
#     @property
#     def uri(self):
#         return self._uri
#
#     @staticmethod
#     def _get_source_type() -> str:
#         return "test"
#
#     def download(self) -> str:
#         # Ignore the "test" URI scheme and download the local path
#         parsed_uri = urlparse(self._uri)
#         return download_artifacts(parsed_uri.path)
#
#     @staticmethod
#     def _can_resolve(raw_source: Any) -> bool:
#         if not isinstance(raw_source, str):
#             return False
#
#         try:
#             parsed_source = urlparse(raw_source)
#             return parsed_source.scheme == "test"
#         except Exception:
#             return False
#
#     @classmethod
#     def _resolve(cls, raw_source: Any) -> DatasetSource:
#         return cls(raw_source)
#
#     def _to_dict(self) -> Dict[str, str]:
#         return {"uri": self.uri}
#
#     @classmethod
#     def _from_dict(cls, source_dict: Dict[str, str]) -> DatasetSource:
#         uri = source_dict.get("uri")
#         if uri is None:
#             raise MlflowException(
#                 'Failed to parse dummy dataset source. Missing expected key: "uri"',
#                 INVALID_PARAMETER_VALUE,
#             )
#
#         return cls(uri=uri)


def test_register_constructor_function_performs_validation():
    registry = DatasetRegistry()

    def from_good_function(path: str, name: Optional[str] = None, digest: Optional[str] = None) -> Dataset:
        pass

    registry.register_constructor(from_good_function)

    def bad_name_fn(name: Optional[str] = None, digest: Optional[str] = None) -> Dataset:
        pass

    with pytest.raises(MlflowException, match="Constructor name must start with"):
        registry.register_constructor(bad_name_fn)

    with pytest.raises(MlflowException, match="Constructor name must start with"):
        registry.register_constructor(constructor_fn=from_good_function, constructor_name="bad_name")

    def from_no_name_fn(digest: Optional[str] = None) -> Dataset:
        pass

    with pytest.raises(MlflowException, match="must define an optional parameter named 'name'"):
        registry.register_constructor(from_no_name_fn)

    def from_no_digest_fn(name: Optional[str] = None) -> Dataset:
        pass

    with pytest.raises(MlflowException, match="must define an optional parameter named 'digest'"):
        registry.register_constructor(from_no_digest_fn)

    def from_bad_return_type_fn(path: str, name: Optional[str] = None, digest: Optional[str] = None) -> str:
        pass

    with pytest.raises(MlflowException, match="must have a return type annotation.*Dataset"):
        registry.register_constructor(from_bad_return_type_fn)

    def from_no_return_type_fn(path: str, name: Optional[str] = None, digest: Optional[str] = None):
        pass

    with pytest.raises(MlflowException, match="must have a return type annotation.*Dataset"):
        registry.register_constructor(from_no_return_type_fn)


def test_register_constructor_from_entrypoints_and_call():
    pass
