from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.evaluation_dataset import EvaluationDataset
from mlflow.data.filesystem_dataset_source import FileSystemDatasetSource
from mlflow.data.polars_dataset import PolarsDataset, from_polars, infer_schema
from mlflow.data.pyfunc_dataset_mixin import PyFuncInputsOutputs
from mlflow.exceptions import MlflowException
from mlflow.types.schema import Array, ColSpec, DataType, Object, Property, Schema

from tests.resources.data.dataset_source import SampleDatasetSource


@pytest.fixture(name="source", scope="module")
def sample_source() -> SampleDatasetSource:
    source_uri = "test:/my/test/uri"
    return SampleDatasetSource._resolve(source_uri)


def test_infer_schema() -> None:
    data = [
        [
            b"asd",
            True,
            datetime(2024, 1, 1, 12, 34, 56, 789),
            10,
            10,
            10,
            10,
            10,
            10,
            "asd",
            "😆",
            "category",
            "val2",
            date(2024, 1, 1),
            10,
            10,
            10,
            [1, 2, 3],
            [1, 2, 3],
            {"col1": 1},
        ]
    ]
    schema = {
        "Binary": pl.Binary,
        "Boolean": pl.Boolean,
        "Datetime": pl.Datetime,
        "Float32": pl.Float32,
        "Float64": pl.Float64,
        "Int8": pl.Int8,
        "Int16": pl.Int16,
        "Int32": pl.Int32,
        "Int64": pl.Int64,
        "String": pl.String,
        "Utf8": pl.Utf8,
        "Categorical": pl.Categorical,
        "Enum": pl.Enum(["val1", "val2"]),
        "Date": pl.Date,
        "UInt8": pl.UInt8,
        "UInt16": pl.UInt16,
        "UInt32": pl.UInt32,
        "List": pl.List(pl.Int8),
        "Array": pl.Array(pl.Int8, 3),
        "Struct": pl.Struct({"col1": pl.Int8}),
    }
    df = pl.DataFrame(data=data, schema=schema)

    assert infer_schema(df) == Schema(
        [
            ColSpec(name="Binary", type=DataType.binary),
            ColSpec(name="Boolean", type=DataType.boolean),
            ColSpec(name="Datetime", type=DataType.datetime),
            ColSpec(name="Float32", type=DataType.float),
            ColSpec(name="Float64", type=DataType.double),
            ColSpec(name="Int8", type=DataType.integer),
            ColSpec(name="Int16", type=DataType.integer),
            ColSpec(name="Int32", type=DataType.integer),
            ColSpec(name="Int64", type=DataType.long),
            ColSpec(name="String", type=DataType.string),
            ColSpec(name="Utf8", type=DataType.string),
            ColSpec(name="Categorical", type=DataType.string),
            ColSpec(name="Enum", type=DataType.string),
            ColSpec(name="Date", type=DataType.datetime),
            ColSpec(name="UInt8", type=DataType.integer),
            ColSpec(name="UInt16", type=DataType.integer),
            ColSpec(name="UInt32", type=DataType.long),
            ColSpec(name="List", type=Array(DataType.integer)),
            ColSpec(name="Array", type=Array(DataType.integer)),
            ColSpec(name="Struct", type=Object([Property(name="col1", dtype=DataType.integer)])),
        ]
    )


def test_conversion_to_json(source: SampleDatasetSource) -> None:
    dataset = PolarsDataset(
        df=pl.DataFrame([1, 2, 3], schema=["Numbers"]), source=source, name="testname"
    )

    dataset_json = dataset.to_json()
    parsed_json = json.loads(dataset_json)

    assert parsed_json.keys() <= {"name", "digest", "source", "source_type", "schema", "profile"}
    assert parsed_json["name"] == dataset.name
    assert parsed_json["digest"] == dataset.digest
    assert parsed_json["source"] == dataset.source.to_json()
    assert parsed_json["source_type"] == dataset.source._get_source_type()
    assert parsed_json["profile"] == json.dumps(dataset.profile)

    schema_json = json.dumps(json.loads(parsed_json["schema"])["mlflow_colspec"])
    assert Schema.from_json(schema_json) == dataset.schema


def test_digest_property_has_expected_value(source: SampleDatasetSource) -> None:
    dataset = PolarsDataset(df=pl.DataFrame([1, 2, 3], schema=["Numbers"]), source=source)
    assert dataset.digest == dataset._compute_digest()
    assert dataset.digest == "2485371048825281677"


def test_digest_consistent(source: SampleDatasetSource) -> None:
    """Row order does not affect digest."""
    dataset1 = PolarsDataset(
        df=pl.DataFrame({"numbers": [1, 2, 3], "strs": ["a", "b", "c"]}), source=source
    )

    dataset2 = PolarsDataset(
        df=pl.DataFrame({"numbers": [2, 3, 1], "strs": ["b", "c", "a"]}), source=source
    )
    assert dataset1.digest == dataset2.digest


def test_digest_change(source: SampleDatasetSource) -> None:
    """Different rows produce different digests."""
    dataset1 = PolarsDataset(
        df=pl.DataFrame({"numbers": [1, 2, 3], "strs": ["a", "b", "c"]}), source=source
    )

    dataset2 = PolarsDataset(
        df=pl.DataFrame({"numbers": [10, 20, 30], "strs": ["aa", "bb", "cc"]}), source=source
    )
    assert dataset1.digest != dataset2.digest


def test_df_property(source: SampleDatasetSource) -> None:
    df = pl.DataFrame({"numbers": [1, 2, 3]})
    dataset = PolarsDataset(df=df, source=source)
    assert dataset.df.equals(df)


def test_targets_none(source: SampleDatasetSource) -> None:
    df_no_targets = pl.DataFrame({"numbers": [1, 2, 3]})
    dataset_no_targets = PolarsDataset(df=df_no_targets, source=source)
    assert dataset_no_targets._targets is None


def test_targets_not_none(source: SampleDatasetSource) -> None:
    df_with_targets = pl.DataFrame({"a": [1, 1], "b": [2, 2], "c": [3, 3]})
    dataset_with_targets = PolarsDataset(df=df_with_targets, source=source, targets="c")
    assert dataset_with_targets._targets == "c"


def test_targets_invalid(source: SampleDatasetSource) -> None:
    df = pl.DataFrame({"a": [1, 1], "b": [2, 2], "c": [3, 3]})
    with pytest.raises(
        MlflowException,
        match="DataFrame does not contain specified targets column: 'd'",
    ):
        PolarsDataset(df=df, source=source, targets="d")


def test_to_pyfunc_wo_outputs(source: SampleDatasetSource) -> None:
    df = pl.DataFrame({"numbers": [1, 2, 3]})
    dataset = PolarsDataset(df=df, source=source)

    input_outputs = dataset.to_pyfunc()

    assert isinstance(input_outputs, PyFuncInputsOutputs)
    assert len(input_outputs.inputs) == 1
    assert isinstance(input_outputs.inputs[0], pd.DataFrame)
    assert input_outputs.inputs[0].equals(pd.DataFrame({"numbers": [1, 2, 3]}))


def test_to_pyfunc_with_outputs(source: SampleDatasetSource) -> None:
    df = pl.DataFrame({"a": [1, 1], "b": [2, 2], "c": [3, 3]})
    dataset = PolarsDataset(df=df, source=source, targets="c")

    input_outputs = dataset.to_pyfunc()

    assert isinstance(input_outputs, PyFuncInputsOutputs)
    assert len(input_outputs.inputs) == 1
    assert isinstance(input_outputs.inputs[0], pd.DataFrame)
    assert input_outputs.inputs[0].equals(pd.DataFrame({"a": [1, 1], "b": [2, 2]}))
    assert len(input_outputs.outputs) == 1
    assert isinstance(input_outputs.outputs[0], pd.Series)
    assert input_outputs.outputs[0].equals(pd.Series([3, 3], name="c"))


def test_from_polars_with_targets(tmp_path: Path) -> None:
    df = pl.DataFrame({"a": [1, 1], "b": [2, 2], "c": [3, 3]})
    path = tmp_path / "temp.csv"
    df.write_csv(path)

    dataset = from_polars(df, targets="c", source=str(path))
    input_outputs = dataset.to_pyfunc()

    assert isinstance(input_outputs, PyFuncInputsOutputs)
    assert len(input_outputs.inputs) == 1
    assert isinstance(input_outputs.inputs[0], pd.DataFrame)
    assert input_outputs.inputs[0].equals(pd.DataFrame({"a": [1, 1], "b": [2, 2]}))
    assert len(input_outputs.outputs) == 1
    assert isinstance(input_outputs.outputs[0], pd.Series)
    assert input_outputs.outputs[0].equals(pd.Series([3, 3], name="c"))


def test_from_polars_file_system_datasource(tmp_path: Path) -> None:
    df = pl.DataFrame({"a": [1, 1], "b": [2, 2], "c": [3, 3]})
    path = tmp_path / "temp.csv"
    df.write_csv(path)

    mlflow_df = from_polars(df, source=str(path))

    assert isinstance(mlflow_df, PolarsDataset)
    assert mlflow_df.df.equals(df)
    assert mlflow_df.schema == infer_schema(df)
    assert mlflow_df.profile == {"num_rows": 2, "num_elements": 6}
    assert isinstance(mlflow_df.source, FileSystemDatasetSource)


def test_from_polars_no_source_specified() -> None:
    df = pl.DataFrame({"a": [1, 1], "b": [2, 2], "c": [3, 3]})

    mlflow_df = from_polars(df)

    assert isinstance(mlflow_df, PolarsDataset)
    assert isinstance(mlflow_df.source, CodeDatasetSource)
    assert "mlflow.source.name" in mlflow_df.source.to_json()


def test_to_evaluation_dataset(source: SampleDatasetSource) -> None:
    import numpy as np

    df = pl.DataFrame({"a": [1, 1], "b": [2, 2], "c": [3, 3]})
    dataset = PolarsDataset(df=df, source=source, targets="c", name="testname")
    evaluation_dataset = dataset.to_evaluation_dataset()

    assert evaluation_dataset.name is not None
    assert evaluation_dataset.digest is not None
    assert isinstance(evaluation_dataset, EvaluationDataset)
    assert isinstance(evaluation_dataset.features_data, pd.DataFrame)
    assert evaluation_dataset.features_data.equals(df.drop("c").to_pandas())
    assert isinstance(evaluation_dataset.labels_data, np.ndarray)
    assert np.array_equal(evaluation_dataset.labels_data, df["c"].to_numpy())
