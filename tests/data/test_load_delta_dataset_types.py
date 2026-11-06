import sys
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import pytest

import mlflow.data
from mlflow.data.delta_dataset_source import DeltaDatasetSource
from mlflow.exceptions import MlflowException


def test_load_delta_invalid_dataset_type_raises():
    with pytest.raises(
        MlflowException,
        match="`dataset_type` must be one of 'spark', 'pandas', or 'polars'",
    ):
        mlflow.data.load_delta(path="/tmp/delta", dataset_type="duckdb")


def test_load_delta_non_spark_requires_path():
    with pytest.raises(
        MlflowException,
        match="`dataset_type='pandas'` and `dataset_type='polars'` require `path`",
    ):
        mlflow.data.load_delta(table_name="my_table", dataset_type="pandas")


def test_load_delta_pandas_uses_deltalake(monkeypatch):
    class FakeDeltaTable:
        def __init__(self, path, version=None):
            self.path = path
            self._version = version

        def version(self):
            return 5

        def to_pandas(self):
            return pd.DataFrame({"feature": [1], "target": [0]})

    monkeypatch.setitem(sys.modules, "deltalake", SimpleNamespace(DeltaTable=FakeDeltaTable))

    expected_dataset = object()
    with mock.patch(
        "mlflow.data.pandas_dataset.from_pandas", return_value=expected_dataset
    ) as from_pandas:
        result = mlflow.data.load_delta(
            path="/tmp/delta",
            dataset_type="pandas",
            targets="target",
            name="my-delta-dataset",
            digest="123",
        )

    assert result is expected_dataset
    from_pandas.assert_called_once()
    call_kwargs = from_pandas.call_args.kwargs
    assert isinstance(call_kwargs["source"], DeltaDatasetSource)
    assert call_kwargs["source"].path == "/tmp/delta"
    assert call_kwargs["source"].delta_table_version == 5
    assert call_kwargs["targets"] == "target"
    assert call_kwargs["name"] == "my-delta-dataset"
    assert call_kwargs["digest"] == "123"
