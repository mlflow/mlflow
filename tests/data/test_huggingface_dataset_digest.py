from unittest import mock

import datasets
import pytest

import mlflow.data
from mlflow.data.digest_utils import MAX_ROWS, compute_pandas_digest


@pytest.mark.parametrize("num_rows", [MAX_ROWS, MAX_ROWS + 1])
def test_digest_distinguishes_datasets_with_identical_prefix_and_different_lengths(num_rows):
    shorter = datasets.Dataset.from_dict({"value": [1] * num_rows})
    longer = datasets.Dataset.from_dict({"value": [1] * (num_rows + 1)})

    assert (
        mlflow.data.from_huggingface(shorter).digest != mlflow.data.from_huggingface(longer).digest
    )


@pytest.mark.parametrize("num_rows", [1, MAX_ROWS])
def test_digest_preserves_untruncated_dataset_digest(num_rows):
    ds = datasets.Dataset.from_dict({"value": list(range(num_rows))})
    assert mlflow.data.from_huggingface(ds).digest == compute_pandas_digest(ds.to_pandas())


def test_digest_only_converts_first_batch_to_pandas():
    ds = datasets.Dataset.from_dict({"value": list(range(MAX_ROWS + 1))})
    with mock.patch.object(ds, "to_pandas", wraps=ds.to_pandas) as to_pandas:
        digest = mlflow.data.from_huggingface(ds).digest
    to_pandas.assert_called_once_with(batch_size=MAX_ROWS, batched=True)
    assert digest == mlflow.data.from_huggingface(ds).digest


def test_explicit_digest_does_not_convert_dataset_to_pandas():
    ds = datasets.Dataset.from_dict({"value": [1]})
    with mock.patch.object(ds, "to_pandas") as to_pandas:
        assert mlflow.data.from_huggingface(ds, digest="custom").digest == "custom"
    to_pandas.assert_not_called()
