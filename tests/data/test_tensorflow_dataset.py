import json

import numpy as np
import pytest
import tensorflow as tf

import mlflow.data
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.evaluation_dataset import EvaluationDataset
from mlflow.data.pyfunc_dataset_mixin import PyFuncInputsOutputs
from mlflow.data.schema import TensorDatasetSchema
from mlflow.data.tensorflow_dataset import TensorFlowDataset
from mlflow.exceptions import MlflowException
from mlflow.types.utils import _infer_schema

from tests.resources.data.dataset_source import SampleDatasetSource


def test_dataset_construction_validates_features_and_targets():
    x = np.random.sample((100, 2))
    tf_dataset = tf.data.Dataset.from_tensors(x)
    tf_tensor = tf.convert_to_tensor(x)

    with pytest.raises(
        MlflowException,
        match="features' must be an instance of tf.data.Dataset or a TensorFlow Tensor.*NoneType",
    ):
        mlflow.data.from_tensorflow(features=None)
    with pytest.raises(
        MlflowException,
        match="features' must be an instance of tf.data.Dataset or a TensorFlow Tensor.*str",
    ):
        mlflow.data.from_tensorflow(features="foo")
    with pytest.raises(
        MlflowException,
        match="features' must be an instance of tf.data.Dataset or a TensorFlow Tensor.*str",
    ):
        mlflow.data.from_tensorflow(features="foo", targets=tf_tensor)

    mlflow.data.from_tensorflow(features=tf_tensor, targets=tf_tensor)
    mlflow.data.from_tensorflow(features=tf_tensor, targets=None)
    with pytest.raises(
        MlflowException,
        match=(
            "If 'features' is a TensorFlow Tensor, then 'targets' must also be a TensorFlow"
            " Tensor.*str"
        ),
    ):
        mlflow.data.from_tensorflow(features=tf_tensor, targets="foo")
    with pytest.raises(
        MlflowException,
        match=(
            "If 'features' is a TensorFlow Tensor, then 'targets' must also be a TensorFlow"
            " Tensor.*Dataset"
        ),
    ):
        mlflow.data.from_tensorflow(features=tf_tensor, targets=tf_dataset)

    mlflow.data.from_tensorflow(features=tf_dataset, targets=tf_dataset)
    mlflow.data.from_tensorflow(features=tf_dataset, targets=None)
    with pytest.raises(
        MlflowException,
        match=(
            "If 'features' is an instance of tf.data.Dataset, then 'targets' must also be an"
            " instance of tf.data.Dataset.*str"
        ),
    ):
        mlflow.data.from_tensorflow(features=tf_dataset, targets="foo")
    with pytest.raises(
        MlflowException,
        match=(
            "If 'features' is an instance of tf.data.Dataset, then 'targets' must also be an"
            " instance of tf.data.Dataset.*Tensor"
        ),
    ):
        mlflow.data.from_tensorflow(features=tf_dataset, targets=tf_tensor)


def test_conversion_to_json():
    source_uri = "test:/my/test/uri"
    x = np.random.sample((100, 2))
    tf_dataset = tf.data.Dataset.from_tensors(x)
    source = SampleDatasetSource._resolve(source_uri)
    dataset = TensorFlowDataset(features=tf_dataset, source=source, name="testname")

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
            tf.data.Dataset.from_tensors(
                {"a": np.random.sample((100, 2)), "b": np.random.sample((100, 4))}
            ),
            tf.data.Dataset.from_tensors(
                {"c": np.random.sample((100, 1)), "d": np.random.sample((100,))}
            ),
        ),
        (
            tf.data.Dataset.from_tensors(
                (
                    np.random.sample((100, 2)),
                    np.random.sample((100, 4)),
                )
            ),
            tf.data.Dataset.from_tensors(
                (
                    np.random.sample((100, 1)),
                    np.random.sample((100,)),
                )
            ),
        ),
        (
            tf.data.Dataset.from_tensors(
                (
                    np.random.sample((100, 2)),
                    np.random.sample((100, 4)),
                )
            ),
            tf.data.Dataset.from_tensors(
                {"c": np.random.sample((100, 1)), "d": np.random.sample((100,))}
            ),
        ),
        (
            tf.data.Dataset.from_tensors(
                (
                    np.random.sample((100, 2)),
                    np.random.sample((100, 4)),
                )
            ),
            None,
        ),
    ],
)
def test_conversion_to_json_with_multi_tensor_datasets(features, targets):
    source_uri = "test:/my/test/uri"
    source = SampleDatasetSource._resolve(source_uri)
    dataset = TensorFlowDataset(features=features, targets=targets, source=source, name="testname")

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


def test_schema_and_profile_with_multi_tensor_tuple_datasets():
    features_dataset = tf.data.Dataset.from_tensors(
        (
            np.random.sample((100, 2)),
            np.random.sample((100, 4)),
        )
    )
    targets_dataset = tf.data.Dataset.from_tensors(
        (
            np.random.sample((100, 1)),
            np.random.sample((100,)),
        )
    )
    source_uri = "test:/my/test/uri"
    source = SampleDatasetSource._resolve(source_uri)
    dataset = TensorFlowDataset(
        features=features_dataset, targets=targets_dataset, source=source, name="testname"
    )
    assert dataset.schema.features == _infer_schema(
        {
            "0": np.random.sample((100, 2)),
            "1": np.random.sample((100, 4)),
        }
    )
    assert dataset.schema.targets == _infer_schema(
        {
            "0": np.random.sample((100, 1)),
            "1": np.random.sample((100,)),
        }
    )
    assert dataset.profile == {
        "features_cardinality": 1,
        "targets_cardinality": 1,
    }
    assert dataset.profile == {
        "features_cardinality": features_dataset.cardinality().numpy(),
        "targets_cardinality": targets_dataset.cardinality().numpy(),
    }


def test_schema_and_profile_with_multi_tensor_dict_datasets():
    features_dataset = tf.data.Dataset.from_tensors(
        {"a": np.random.sample((100, 2)), "b": np.random.sample((100, 4))}
    )
    targets_dataset = tf.data.Dataset.from_tensors(
        {"c": np.random.sample((100, 1)), "d": np.random.sample((100,))}
    )
    source_uri = "test:/my/test/uri"
    source = SampleDatasetSource._resolve(source_uri)
    dataset = TensorFlowDataset(
        features=features_dataset, targets=targets_dataset, source=source, name="testname"
    )
    assert dataset.schema.features == _infer_schema(
        {
            "a": np.random.sample((100, 2)),
            "b": np.random.sample((100, 4)),
        }
    )
    assert dataset.schema.targets == _infer_schema(
        {
            "c": np.random.sample((100, 1)),
            "d": np.random.sample((100,)),
        }
    )
    assert dataset.profile == {
        "features_cardinality": 1,
        "targets_cardinality": 1,
    }
    assert dataset.profile == {
        "features_cardinality": features_dataset.cardinality().numpy(),
        "targets_cardinality": targets_dataset.cardinality().numpy(),
    }


def test_digest_property_has_expected_value():
    source_uri = "test:/my/test/uri"
    x = [[1, 2, 3], [4, 5, 6]]
    tf_dataset = tf.data.Dataset.from_tensors(x)
    source = SampleDatasetSource._resolve(source_uri)
    dataset = TensorFlowDataset(features=tf_dataset, source=source, name="testname")
    assert dataset.digest == dataset._compute_digest()
    assert dataset.digest == "666a9820"


def test_data_property_has_expected_value():
    source_uri = "test:/my/test/uri"
    x = [[1, 2, 3], [4, 5, 6]]
    tf_dataset = tf.data.Dataset.from_tensors(x)
    source = SampleDatasetSource._resolve(source_uri)
    dataset = TensorFlowDataset(features=tf_dataset, source=source, name="testname")
    assert dataset.data == tf_dataset


def test_source_property_has_expected_value():
    source_uri = "test:/my/test/uri"
    x = [[1, 2, 3], [4, 5, 6]]
    tf_dataset = tf.data.Dataset.from_tensors(x)
    source = SampleDatasetSource._resolve(source_uri)
    dataset = TensorFlowDataset(features=tf_dataset, source=source, name="testname")
    assert dataset.source == source


def test_profile_property_has_expected_value_dataset():
    source_uri = "test:/my/test/uri"
    x = [[1, 2, 3], [4, 5, 6]]
    tf_dataset = tf.data.Dataset.from_tensors(x)
    source = SampleDatasetSource._resolve(source_uri)
    dataset = TensorFlowDataset(features=tf_dataset, source=source, name="testname")
    assert dataset.profile == {
        "features_cardinality": tf_dataset.cardinality().numpy(),
    }


def test_profile_property_has_expected_value_tensors():
    source_uri = "test:/my/test/uri"
    x = [[1, 2, 3], [4, 5, 6]]
    tf_tensor = tf.convert_to_tensor(x)
    source = SampleDatasetSource._resolve(source_uri)
    dataset = TensorFlowDataset(features=tf_tensor, source=source, name="testname")
    assert dataset.profile == {
        "features_cardinality": tf.size(tf_tensor).numpy(),
    }


def test_to_pyfunc():
    source_uri = "test:/my/test/uri"
    x = np.random.sample((100, 2))
    tf_dataset = tf.data.Dataset.from_tensors(x)
    source = SampleDatasetSource._resolve(source_uri)
    dataset = TensorFlowDataset(features=tf_dataset, source=source, name="testname")
    assert isinstance(dataset.to_pyfunc(), PyFuncInputsOutputs)


def test_to_evaluation_dataset():
    source_uri = "test:/my/test/uri"
    x = np.random.sample((2, 2))
    y = np.random.sample((2, 1))
    x_tensors = tf.convert_to_tensor(x)
    y_tensors = tf.convert_to_tensor(y)
    source = SampleDatasetSource._resolve(source_uri)
    dataset = TensorFlowDataset(
        features=x_tensors, source=source, targets=y_tensors, name="testname"
    )
    evaluation_dataset = dataset.to_evaluation_dataset()
    assert isinstance(evaluation_dataset, EvaluationDataset)
    assert np.array_equal(evaluation_dataset.features_data, dataset.data.numpy())
    assert np.array_equal(evaluation_dataset.labels_data, dataset.targets.numpy())


def test_to_evaluation_dataset_with_tensorflow_dataset_data():
    source_uri = "test:/my/test/uri"
    x = np.random.sample((2, 2))
    y = np.random.sample((2, 1))
    x_tf_data = tf.data.Dataset.from_tensors(x)
    y_tf_data = tf.data.Dataset.from_tensors(y)
    source = SampleDatasetSource._resolve(source_uri)
    dataset = TensorFlowDataset(
        features=x_tf_data, source=source, targets=y_tf_data, name="testname"
    )
    with pytest.raises(
        MlflowException, match="Data must be a Tensor to convert to an EvaluationDataset"
    ):
        dataset.to_evaluation_dataset()


def test_from_tensorflow_dataset_constructs_expected_dataset():
    x = np.random.sample((100, 2))
    tf_dataset = tf.data.Dataset.from_tensors(x)
    mlflow_ds = mlflow.data.from_tensorflow(tf_dataset, source="my_source")
    assert isinstance(mlflow_ds, TensorFlowDataset)
    assert mlflow_ds.data == tf_dataset
    assert mlflow_ds.schema == TensorDatasetSchema(
        features=_infer_schema(next(tf_dataset.as_numpy_iterator()))
    )
    assert mlflow_ds.profile == {
        "features_cardinality": tf_dataset.cardinality().numpy(),
    }


def test_from_tensorflow_dataset_with_targets_constructs_expected_dataset():
    x = np.random.sample((100, 2))
    y = np.random.sample((100, 1))
    tf_dataset_x = tf.data.Dataset.from_tensors(x)
    tf_dataset_y = tf.data.Dataset.from_tensors(y)
    mlflow_ds = mlflow.data.from_tensorflow(tf_dataset_x, source="my_source", targets=tf_dataset_y)
    assert isinstance(mlflow_ds, TensorFlowDataset)
    assert mlflow_ds.data == tf_dataset_x
    assert mlflow_ds.targets == tf_dataset_y
    assert mlflow_ds.schema == TensorDatasetSchema(
        features=_infer_schema(next(tf_dataset_x.as_numpy_iterator())),
        targets=_infer_schema(next(tf_dataset_y.as_numpy_iterator())),
    )
    assert mlflow_ds.profile == {
        "features_cardinality": tf_dataset_x.cardinality().numpy(),
        "targets_cardinality": tf_dataset_y.cardinality().numpy(),
    }


def test_from_tensorflow_tensor_constructs_expected_dataset():
    x = np.random.sample((100, 2))
    tf_tensor = tf.convert_to_tensor(x)
    mlflow_ds = mlflow.data.from_tensorflow(tf_tensor, source="my_source")
    assert isinstance(mlflow_ds, TensorFlowDataset)
    # compare if two tensors are equal using tensorflow utils
    assert tf.reduce_all(tf.math.equal(mlflow_ds.data, tf_tensor))
    assert mlflow_ds.schema == TensorDatasetSchema(features=_infer_schema(tf_tensor.numpy()))
    assert mlflow_ds.profile == {
        "features_cardinality": tf.size(tf_tensor).numpy(),
    }


def test_from_tensorflow_tensor_with_targets_constructs_expected_dataset():
    x = np.random.sample((100, 2))
    y = np.random.sample((100, 1))
    tf_tensor_x = tf.convert_to_tensor(x)
    tf_tensor_y = tf.convert_to_tensor(y)
    mlflow_ds = mlflow.data.from_tensorflow(tf_tensor_x, source="my_source", targets=tf_tensor_y)
    assert isinstance(mlflow_ds, TensorFlowDataset)
    assert tf.reduce_all(tf.math.equal(mlflow_ds.data, tf_tensor_x))
    assert tf.reduce_all(tf.math.equal(mlflow_ds.targets, tf_tensor_y))
    assert mlflow_ds.schema == TensorDatasetSchema(
        features=_infer_schema(tf_tensor_x.numpy()),
        targets=_infer_schema(tf_tensor_y.numpy()),
    )
    assert mlflow_ds.profile == {
        "features_cardinality": tf.size(tf_tensor_x).numpy(),
        "targets_cardinality": tf.size(tf_tensor_y).numpy(),
    }


def test_from_tensorflow_no_source_specified():
    x = np.random.sample((100, 2))
    tf_dataset = tf.data.Dataset.from_tensors(x)
    mlflow_ds = mlflow.data.from_tensorflow(tf_dataset)

    assert isinstance(mlflow_ds, TensorFlowDataset)

    assert isinstance(mlflow_ds.source, CodeDatasetSource)
    assert "mlflow.source.name" in mlflow_ds.source.to_json()


def test_digest_computation_succeeds_with_none_element_in_numpy_iterator():
    x = np.array([[0, 1], [1, 2]])
    tf_dataset = tf.data.Dataset.from_tensors(x)
    tf_dataset.as_numpy_iterator = lambda: [None, x]
    mlflow_ds = mlflow.data.from_tensorflow(tf_dataset)
    assert mlflow_ds.digest == "bc8ef018"
