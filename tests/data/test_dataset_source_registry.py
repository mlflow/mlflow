from typing import Any, Dict
from urllib.parse import urlparse

import pytest

from mlflow.artifacts import download_artifacts
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.dataset_source_registry import DatasetSourceRegistry
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE


class TestDatasetSource(DatasetSource):
    def __init__(self, uri):
        self._uri = uri

    @property
    def uri(self):
        return self._uri

    @staticmethod
    def _get_source_type() -> str:
        return "test"

    def download(self) -> str:
        # Ignore the "test" URI scheme and download the local path
        parsed_uri = urlparse(self._uri)
        return download_artifacts(parsed_uri.path)

    @staticmethod
    def _can_resolve(raw_source: Any) -> bool:
        if not isinstance(raw_source, str):
            return False

        try:
            parsed_source = urlparse(raw_source)
            return parsed_source.scheme == "test"
        except Exception:
            return False

    @classmethod
    def _resolve(cls, raw_source: Any) -> DatasetSource:
        return cls(raw_source)

    def _to_dict(self) -> Dict[str, str]:
        return {"uri": self.uri}

    @classmethod
    def _from_dict(cls, source_dict: Dict[str, str]) -> DatasetSource:
        uri = source_dict.get("uri")
        if uri is None:
            raise MlflowException(
                'Failed to parse dummy dataset source. Missing expected key: "uri"',
                INVALID_PARAMETER_VALUE,
            )

        return cls(uri=uri)


def test_register_entrypoints_and_resolve(tmp_path):
    """This test requires the package in tests/resources/mlflow-test-plugin to be installed"""

    from mlflow_test_plugin.dummy_dataset_source import DummyDatasetSource

    registry = DatasetSourceRegistry()
    registry.register_entrypoints()

    uri = "dummy:" + str(tmp_path)
    resolved_source = registry.resolve(uri)
    assert isinstance(resolved_source, DummyDatasetSource)
    # Verify that the DummyDatasetSource is constructed with the correct URI
    assert resolved_source.uri == uri


def test_register_dataset_source_and_resolve(tmp_path):
    registry = DatasetSourceRegistry()
    registry.register(TestDatasetSource)

    uri = "test:" + str(tmp_path)
    resolved_source = registry.resolve(uri)
    assert isinstance(resolved_source, TestDatasetSource)
    # Verify that the TestDatasetSource is constructed with the correct URI
    assert resolved_source.uri == uri


def test_register_dataset_source_and_load_from_json(tmp_path):
    registry = DatasetSourceRegistry()
    registry.register(TestDatasetSource)
    resolved_source = registry.resolve("test:" + str(tmp_path))
    resolved_source_json = resolved_source.to_json()
    source_from_json = registry.get_source_from_json(
        source_json=resolved_source_json, source_type="test"
    )
    assert source_from_json.uri == resolved_source.uri


def test_resolve_dataset_only_considers_candidates_if_specified_using_inheritance(tmp_path):
    class CandidateDatasetSource1(TestDatasetSource):
        @staticmethod
        def _get_source_type() -> str:
            return "candidate1"

        @staticmethod
        def _can_resolve(raw_source: Any) -> bool:
            return raw_source.startswith("candidate1")

    class CandidateDatasetSource2(CandidateDatasetSource1):
        @staticmethod
        def _get_source_type() -> str:
            return "candidate2"

        @staticmethod
        def _can_resolve(raw_source: Any) -> bool:
            return raw_source.startswith("candidate2")

    registry = DatasetSourceRegistry()
    registry.register(TestDatasetSource)
    registry.register(CandidateDatasetSource1)
    registry.register(CandidateDatasetSource2)

    registry.resolve("test:" + str(tmp_path))
    registry.resolve("test:" + str(tmp_path), candidate_sources=[TestDatasetSource])
    with pytest.raises(MlflowException, match="Could not find a source information resolver"):
        # TestDatasetSource is the only source that can resolve raw sources with scheme "test",
        # and TestDatasetSource is not a subclass of CandidateDatasetSource1
        registry.resolve("test:" + str(tmp_path), candidate_sources=[CandidateDatasetSource1])

    registry.resolve("candidate1:" + str(tmp_path))
    registry.resolve("candidate1:" + str(tmp_path), candidate_sources=[CandidateDatasetSource1])
    # CandidateDatasetSource1 is a subclass of TestDatasetSource and is therefore considered
    # as a candidate for resolution
    registry.resolve("candidate1:" + str(tmp_path), candidate_sources=[TestDatasetSource])
    with pytest.raises(MlflowException, match="Could not find a source information resolver"):
        # CandidateDatasetSource2 is not a superclass of CandidateDatasetSource1 or
        # TestDatasetSource and cannot resolve raw sources with scheme "candidate1"
        registry.resolve("candidate1:" + str(tmp_path), candidate_sources=[CandidateDatasetSource2])


def test_load_from_json_throws_for_unrecognized_source_type(tmp_path):
    registry = DatasetSourceRegistry()
    registry.register(TestDatasetSource)

    with pytest.raises(MlflowException, match="unrecognized source type: foo"):
        registry.get_source_from_json(source_json='{"bar": "123"}', source_type="foo")
