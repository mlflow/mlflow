from typing import Optional
from unittest import mock

import pytest

import mlflow.data
from mlflow.data.dataset import Dataset
from mlflow.data.dataset_registry import DatasetRegistry, register_constructor
from mlflow.data.dataset_source_registry import DatasetSourceRegistry, resolve_dataset_source
from mlflow.exceptions import MlflowException

from tests.resources.data.dataset import SampleDataset
from tests.resources.data.dataset_source import SampleDatasetSource


@pytest.fixture
def dataset_source_registry():
    registry = DatasetSourceRegistry()
    with mock.patch("mlflow.data.dataset_source_registry._dataset_source_registry", wraps=registry):
        yield registry


@pytest.fixture
def dataset_registry():
    registry = DatasetRegistry()
    with mock.patch("mlflow.data.dataset_registry._dataset_registry", wraps=registry):
        yield registry


def test_register_constructor_function_performs_validation():
    registry = DatasetRegistry()

    def from_good_function(
        path: str,
        name: Optional[str] = None,
        digest: Optional[str] = None,
    ) -> Dataset:
        pass

    registry.register_constructor(from_good_function)

    def bad_name_fn(
        name: Optional[str] = None,
        digest: Optional[str] = None,
    ) -> Dataset:
        pass

    with pytest.raises(MlflowException, match="Constructor name must start with"):
        registry.register_constructor(bad_name_fn)

    with pytest.raises(MlflowException, match="Constructor name must start with"):
        registry.register_constructor(
            constructor_fn=from_good_function, constructor_name="bad_name"
        )

    def from_no_name_fn(
        digest: Optional[str] = None,
    ) -> Dataset:
        pass

    with pytest.raises(MlflowException, match="must define an optional parameter named 'name'"):
        registry.register_constructor(from_no_name_fn)

    def from_no_digest_fn(
        name: Optional[str] = None,
    ) -> Dataset:
        pass

    with pytest.raises(MlflowException, match="must define an optional parameter named 'digest'"):
        registry.register_constructor(from_no_digest_fn)

    def from_bad_return_type_fn(
        path: str,
        name: Optional[str] = None,
        digest: Optional[str] = None,
    ) -> str:
        pass

    with pytest.raises(MlflowException, match="must have a return type annotation.*Dataset"):
        registry.register_constructor(from_bad_return_type_fn)

    def from_no_return_type_fn(
        path: str,
        name: Optional[str] = None,
        digest: Optional[str] = None,
    ):
        pass

    with pytest.raises(MlflowException, match="must have a return type annotation.*Dataset"):
        registry.register_constructor(from_no_return_type_fn)


def test_register_constructor_from_entrypoints_and_call(dataset_registry, tmp_path):
    """This test requires the package in tests/resources/mlflow-test-plugin to be installed"""

    from mlflow_test_plugin.dummy_dataset import DummyDataset

    dataset_registry.register_entrypoints()

    dataset = mlflow.data.from_dummy(
        data_list=[1, 2, 3],
        # Use a DummyDatasetSource URI from mlflow_test_plugin.dummy_dataset_source, which
        # is registered as an entrypoint whenever mlflow-test-plugin is installed
        source="dummy:" + str(tmp_path),
        name="dataset_name",
        digest="foo",
    )
    assert isinstance(dataset, DummyDataset)
    assert dataset.data_list == [1, 2, 3]
    assert dataset.name == "dataset_name"
    assert dataset.digest == "foo"


def test_register_constructor_and_call(dataset_registry, dataset_source_registry, tmp_path):
    dataset_source_registry.register(SampleDatasetSource)

    def from_test(data_list, source, name=None, digest=None) -> SampleDataset:
        resolved_source: SampleDatasetSource = resolve_dataset_source(
            source, candidate_sources=[SampleDatasetSource]
        )
        return SampleDataset(data_list=data_list, source=resolved_source, name=name, digest=digest)

    register_constructor(constructor_fn=from_test)
    register_constructor(constructor_name="from_test_2", constructor_fn=from_test)

    dataset1 = mlflow.data.from_test(
        data_list=[1, 2, 3],
        # Use a SampleDatasetSourceURI
        source="test:" + str(tmp_path),
        name="name1",
        digest="digest1",
    )
    assert isinstance(dataset1, SampleDataset)
    assert dataset1.data_list == [1, 2, 3]
    assert dataset1.name == "name1"
    assert dataset1.digest == "digest1"

    dataset2 = mlflow.data.from_test_2(
        data_list=[4, 5, 6],
        # Use a SampleDatasetSourceURI
        source="test:" + str(tmp_path),
        name="name2",
        digest="digest2",
    )
    assert isinstance(dataset2, SampleDataset)
    assert dataset2.data_list == [4, 5, 6]
    assert dataset2.name == "name2"
    assert dataset2.digest == "digest2"


def test_dataset_source_registration_failure(dataset_source_registry):
    with mock.patch.object(dataset_source_registry, "register", side_effect=ImportError("Error")):
        with pytest.warns(UserWarning, match="Failure attempting to register dataset constructor"):
            dataset_source_registry.register_entrypoints()
