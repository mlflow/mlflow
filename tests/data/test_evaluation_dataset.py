import numpy as np
import pandas as pd
import pytest

import mlflow
from mlflow.data.evaluation_dataset import (
    _hash_array_like_obj_as_bytes,
    convert_data_to_mlflow_dataset,
)
from mlflow.exceptions import MlflowException


def _ragged_array(rows):
    # np.array() can't build an object array of uneven-length arrays directly.
    arr = np.empty(len(rows), dtype=object)
    arr[:] = [np.array(row) for row in rows]
    return arr


@pytest.mark.parametrize("data", [[[1], [2], [3]], [1, 2, 3]])
def test_convert_list_data_with_numpy_targets(data):
    targets = np.array([1, 2, 3])

    dataset = convert_data_to_mlflow_dataset(
        data=data,
        targets=targets,
    )

    assert np.array_equal(dataset.targets, targets)
    assert dataset.features.shape == (3, 1)


def test_convert_list_data_without_targets():
    dataset = convert_data_to_mlflow_dataset(data=[[1], [2], [3]], targets=None)

    assert dataset.targets is None


def test_convert_list_data_with_empty_targets_uses_standard_validation():
    dataset = convert_data_to_mlflow_dataset(data=[[1], [2], [3]], targets=[])

    with pytest.raises(MlflowException, match="same length"):
        dataset.to_evaluation_dataset()


def test_convert_empty_list_uses_standard_evaluation_dataset_validation():
    dataset = convert_data_to_mlflow_dataset(data=[], targets=[1, 2, 3])

    with pytest.raises(MlflowException, match="2-dimensional"):
        dataset.to_evaluation_dataset()


@pytest.mark.parametrize(
    ("rows_a", "rows_b"),
    [
        ([[{"a": 1}, {"a": 2}], [{"a": 3}]], [[{"a": 1}], [{"a": 2}, {"a": 3}]]),
        ([[1, 2], [3]], [[1], [2, 3]]),
    ],
)
def test_hash_array_like_obj_as_bytes_with_ragged_nested_arrays(rows_a, rows_b):
    hash_a = _hash_array_like_obj_as_bytes(_ragged_array(rows_a))

    assert hash_a == _hash_array_like_obj_as_bytes(_ragged_array(rows_a))
    assert hash_a != _hash_array_like_obj_as_bytes(_ragged_array(rows_b))


@pytest.mark.parametrize("missing_idx", [0, 1])
@pytest.mark.parametrize("missing", [None, float("nan")])
def test_hash_array_like_obj_as_bytes_with_ragged_nested_arrays_and_missing_rows(
    missing, missing_idx
):
    data = _ragged_array([[{"a": 1}, {"a": 2}], [{"a": 3}]])
    data[missing_idx] = missing

    assert _hash_array_like_obj_as_bytes(data) == _hash_array_like_obj_as_bytes(data.copy())


def test_from_numpy_with_ragged_nested_targets():
    features = np.array([[1], [2]])
    targets_a = _ragged_array([[{"a": 1}, {"a": 2}], [{"a": 3}]])
    targets_b = _ragged_array([[{"a": 1}], [{"a": 2}, {"a": 3}]])

    eval_dataset_a = mlflow.data.from_numpy(features, targets=targets_a).to_evaluation_dataset()
    eval_dataset_b = mlflow.data.from_numpy(features, targets=targets_b).to_evaluation_dataset()

    assert eval_dataset_a.hash != eval_dataset_b.hash


@pytest.mark.parametrize("column", ["targets", "predictions"])
@pytest.mark.parametrize("values", [[["doc-a"]], [["doc-a", "doc-b"], []]])
def test_list_valued_evaluation_hash_is_repeatable(column, values):
    frame = pd.DataFrame({"query": ["question"] * len(values), "documents": values})
    datasets = [
        mlflow.data.from_pandas(
            frame.copy(deep=True), **{column: "documents"}
        ).to_evaluation_dataset()
        for _ in range(5)
    ]

    assert len({dataset.hash for dataset in datasets}) == 1


@pytest.mark.parametrize("column", ["targets", "predictions"])
def test_list_valued_evaluation_hash_changes_with_document_ids(column):
    datasets = [
        mlflow.data.from_pandas(
            pd.DataFrame({"query": ["question"], "documents": [[document]]}),
            **{column: "documents"},
        ).to_evaluation_dataset()
        for document in ("doc-a", "doc-b", "doc-c")
    ]

    assert len({dataset.hash for dataset in datasets}) == len(datasets)
