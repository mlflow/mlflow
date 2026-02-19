import json

import numpy as np
import pandas as pd
import pytest

import mlflow.data
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.evaluation_dataset import EvaluationDataset
from mlflow.data.filesystem_dataset_source import FileSystemDatasetSource
from mlflow.data.numpy_dataset import NumpyDataset
from mlflow.data.pyfunc_dataset_mixin import PyFuncInputsOutputs
from mlflow.data.schema import TensorDatasetSchema
from mlflow.types.utils import _infer_schema

from tests.resources.data.dataset_source import SampleDatasetSource


def test_conversion_to_json():
    source_uri = "test:/my/test/uri"
    source = SampleDatasetSource._resolve(source_uri)
    dataset = NumpyDataset(features=np.array([1, 2, 3]), source=source, name="testname")

    dataset_json = dataset.to_json()
    parsed_json = json.loads(dataset_json)
    assert parsed_json.keys() <= {"name", "digest", "source", "source_type", "schema", "profile"}
    assert parsed_json["name"] == dataset.name
    assert parsed_json["digest"] == dataset.digest
    assert parsed_json["source"] == dataset.source.to_json()
    assert parsed_json["source_type"] == dataset.source._get_source_type()
    assert parsed_json["profile"] == json.dumps(dataset.profile)

    parsed_schema = json.loads(parsed_json["schema"])
    assert TensorDatasetSchema.from_dict(parsed_schema) == dataset.schema


@pytest.mark.parametrize(
    ("features", "targets"),
    [
        (
            {
                "a": np.array([1, 2, 3]),
                "b": np.array([[4, 5]]),
            },
            {
                "c": np.array([1]),
                "d": np.array([[[2]]]),
            },
        ),
        (
            np.array([1, 2, 3]),
            {
                "c": np.array([1]),
                "d": np.array([[[2]]]),
            },
        ),
        (
            {
                "a": np.array([1, 2, 3]),
                "b": np.array([[4, 5]]),
            },
            np.array([1, 2, 3]),
        ),
    ],
)
def test_conversion_to_json_with_multi_tensor_features_and_targets(features, targets):
    source_uri = "test:/my/test/uri"
    source = SampleDatasetSource._resolve(source_uri)
    dataset = NumpyDataset(features=features, targets=targets, source=source)

    dataset_json = dataset.to_json()
    parsed_json = json.loads(dataset_json)
    assert parsed_json.keys() <= {"name", "digest", "source", "source_type", "schema", "profile"}
    assert parsed_json["name"] == dataset.name
    assert parsed_json["digest"] == dataset.digest
    assert parsed_json["source"] == dataset.source.to_json()
    assert parsed_json["source_type"] == dataset.source._get_source_type()
    assert parsed_json["profile"] == json.dumps(dataset.profile)
    parsed_schema = json.loads(parsed_json["schema"])
    assert TensorDatasetSchema.from_dict(parsed_schema) == dataset.schema


@pytest.mark.parametrize(
    ("features", "targets"),
    [
        (
            {
                "a": np.array([1, 2, 3]),
                "b": np.array([[4, 5]]),
            },
            {
                "c": np.array([1]),
                "d": np.array([[[2]]]),
            },
        ),
        (
            np.array([1, 2, 3]),
            {
                "c": np.array([1]),
                "d": np.array([[[2]]]),
            },
        ),
        (
            {
                "a": np.array([1, 2, 3]),
                "b": np.array([[4, 5]]),
            },
            np.array([1, 2, 3]),
        ),
    ],
)
def test_schema_and_profile_with_multi_tensor_features_and_targets(features, targets):
    source_uri = "test:/my/test/uri"
    source = SampleDatasetSource._resolve(source_uri)
    dataset = NumpyDataset(features=features, targets=targets, source=source)

    assert isinstance(dataset.schema, TensorDatasetSchema)
    assert dataset.schema.features == _infer_schema(features)
    assert dataset.schema.targets == _infer_schema(targets)

    if isinstance(features, dict):
        assert {
            "features_shape": {key: array.shape for key, array in features.items()},
            "features_size": {key: array.size for key, array in features.items()},
            "features_nbytes": {key: array.nbytes for key, array in features.items()},
        }.items() <= dataset.profile.items()
    else:
        assert {
            "features_shape": features.shape,
            "features_size": features.size,
            "features_nbytes": features.nbytes,
        }.items() <= dataset.profile.items()

    if isinstance(targets, dict):
        assert {
            "targets_shape": {key: array.shape for key, array in targets.items()},
            "targets_size": {key: array.size for key, array in targets.items()},
            "targets_nbytes": {key: array.nbytes for key, array in targets.items()},
        }.items() <= dataset.profile.items()
    else:
        assert {
            "targets_shape": targets.shape,
            "targets_size": targets.size,
            "targets_nbytes": targets.nbytes,
        }.items() <= dataset.profile.items()


def test_digest_property_has_expected_value():
    source_uri = "test:/my/test/uri"
    source = SampleDatasetSource._resolve(source_uri)
    features = np.array([1, 2, 3])
    targets = np.array([4, 5, 6])
    dataset_with_features = NumpyDataset(features=features, source=source, name="testname")
    assert dataset_with_features.digest == dataset_with_features._compute_digest()
    assert dataset_with_features.digest == "fdf1765f"
    dataset_with_features_and_targets = NumpyDataset(
        features=features, targets=targets, source=source, name="testname"
    )
    assert (
        dataset_with_features_and_targets.digest
        == dataset_with_features_and_targets._compute_digest()
    )
    assert dataset_with_features_and_targets.digest == "1387de76"


def test_features_property():
    source_uri = "test:/my/test/uri"
    source = SampleDatasetSource._resolve(source_uri)
    features = np.array([1, 2, 3])
    dataset = NumpyDataset(features=features, source=source, name="testname")
    assert np.array_equal(dataset.features, features)


def test_targets_property():
    source_uri = "test:/my/test/uri"
    source = SampleDatasetSource._resolve(source_uri)
    features = np.array([1, 2, 3])
    targets = np.array([4, 5, 6])
    dataset_with_targets = NumpyDataset(
        features=features, targets=targets, source=source, name="testname"
    )
    assert np.array_equal(dataset_with_targets.targets, targets)
    dataset_without_targets = NumpyDataset(features=features, source=source, name="testname")
    assert dataset_without_targets.targets is None


def test_to_pyfunc():
    source_uri = "test:/my/test/uri"
    source = SampleDatasetSource._resolve(source_uri)
    features = np.array([1, 2, 3])
    dataset = NumpyDataset(features=features, source=source, name="testname")
    assert isinstance(dataset.to_pyfunc(), PyFuncInputsOutputs)


def test_to_evaluation_dataset():
    source_uri = "test:/my/test/uri"
    source = SampleDatasetSource._resolve(source_uri)
    features = np.array([[1, 2], [3, 4]])
    targets = np.array([0, 1])
    dataset = NumpyDataset(features=features, targets=targets, source=source, name="testname")
    evaluation_dataset = dataset.to_evaluation_dataset()
    assert isinstance(evaluation_dataset, EvaluationDataset)
    assert np.array_equal(evaluation_dataset.features_data, features)
    assert np.array_equal(evaluation_dataset.labels_data, targets)


def test_from_numpy_features_only(tmp_path):
    features = np.array([1, 2, 3])
    path = tmp_path / "temp.csv"
    pd.DataFrame(features).to_csv(path)
    mlflow_features = mlflow.data.from_numpy(features, source=path)

    assert isinstance(mlflow_features, NumpyDataset)
    assert np.array_equal(mlflow_features.features, features)
    assert mlflow_features.schema == TensorDatasetSchema(features=_infer_schema(features))
    assert mlflow_features.profile == {
        "features_shape": features.shape,
        "features_size": features.size,
        "features_nbytes": features.nbytes,
    }

    assert isinstance(mlflow_features.source, FileSystemDatasetSource)


def test_from_numpy_features_and_targets(tmp_path):
    features = np.array([[1, 2, 3], [3, 2, 1], [2, 3, 1]])
    targets = np.array([4, 5, 6])
    path = tmp_path / "temp.csv"
    pd.DataFrame(features).to_csv(path)
    mlflow_ds = mlflow.data.from_numpy(features, targets=targets, source=path)

    assert isinstance(mlflow_ds, NumpyDataset)
    assert np.array_equal(mlflow_ds.features, features)
    assert np.array_equal(mlflow_ds.targets, targets)
    assert mlflow_ds.schema == TensorDatasetSchema(
        features=_infer_schema(features), targets=_infer_schema(targets)
    )
    assert mlflow_ds.profile == {
        "features_shape": features.shape,
        "features_size": features.size,
        "features_nbytes": features.nbytes,
        "targets_shape": targets.shape,
        "targets_size": targets.size,
        "targets_nbytes": targets.nbytes,
    }

    assert isinstance(mlflow_ds.source, FileSystemDatasetSource)


def test_from_numpy_no_source_specified():
    features = np.array([1, 2, 3])
    mlflow_features = mlflow.data.from_numpy(features)

    assert isinstance(mlflow_features, NumpyDataset)

    assert isinstance(mlflow_features.source, CodeDatasetSource)
    assert "mlflow.source.name" in mlflow_features.source.to_json()
