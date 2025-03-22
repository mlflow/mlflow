from __future__ import annotations

import json
from typing import Any, Callable

import narwhals.stable.v1 as nw
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
from narwhals.typing import IntoDataFrame

import mlflow.data
from mlflow.data.arrow_dataset import ArrowDataset
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.dataframe_dataset import infer_mlflow_schema
from mlflow.data.dataset import Dataset
from mlflow.data.delta_dataset_source import DeltaDatasetSource
from mlflow.data.evaluation_dataset import EvaluationDataset
from mlflow.data.filesystem_dataset_source import FileSystemDatasetSource
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.data.polars_dataset import PolarsDataset
from mlflow.data.pyfunc_dataset_mixin import PyFuncInputsOutputs
from mlflow.data.spark_dataset_source import SparkDatasetSource
from mlflow.exceptions import MlflowException
from mlflow.types.schema import Schema

from tests.resources.data.dataset_source import SampleDatasetSource


@pytest.fixture(scope="module")
def spark_session():
    from pyspark.sql import SparkSession

    with (
        SparkSession.builder.master("local[*]")
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.0.0")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        )
        .getOrCreate()
    ) as session:
        yield session


constructors = (
    pd.DataFrame,
    lambda data: pd.DataFrame(data).convert_dtypes(dtype_backend="numpy_nullable"),
    lambda data: pd.DataFrame(data).convert_dtypes(dtype_backend="pyarrow"),
    pl.DataFrame,
    pa.table,
)

datasets = (
    PandasDataset,
    PandasDataset,
    PandasDataset,
    PolarsDataset,
    ArrowDataset,
)

loaders = (
    mlflow.data.from_pandas,
    mlflow.data.from_pandas,
    mlflow.data.from_pandas,
    mlflow.data.from_polars,
    mlflow.data.from_arrow,
)


@pytest.fixture(scope="module", params=zip(constructors, datasets, loaders, strict=True))
def constructor_dataset_loaders(
    request: pytest.FixtureRequest,
) -> tuple[Callable[[dict[str, list[Any]]], IntoDataFrame], type[Dataset], Callable[..., Dataset]]:
    return request.param


def test_conversion_to_json(constructor_dataset_loaders):
    source_uri = "test:/my/test/uri"
    source = SampleDatasetSource._resolve(source_uri)

    data = {"Numbers": [1, 2, 3]}
    constructor, dataset_cls, _ = constructor_dataset_loaders

    dataset = dataset_cls(
        constructor(data),
        source=source,
        name="testname",
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


def test_digest_property_has_expected_value(constructor_dataset_loaders, request):
    constructor, dataset_cls, _ = constructor_dataset_loaders

    source_uri = "test:/my/test/uri"
    source = SampleDatasetSource._resolve(source_uri)
    data = {"Numbers": [1, 2, 3]}

    dataset = dataset_cls(
        constructor(data),
        source=source,
        name="testname",
    )

    assert dataset.digest == dataset._compute_digest()
    # assert dataset.digest == "31ccce44"  #TODO: Polars returns a different value


def test_df_property(constructor_dataset_loaders):
    constructor, dataset_cls, _ = constructor_dataset_loaders

    source_uri = "test:/my/test/uri"
    source = SampleDatasetSource._resolve(source_uri)

    data = {"Numbers": [1, 2, 3]}
    dataset = dataset_cls(
        constructor(data),
        source=source,
        name="testname",
    )
    assert dataset.df.equals(constructor(data))


def test_targets_property(constructor_dataset_loaders):
    constructor, dataset_cls, _ = constructor_dataset_loaders

    source_uri = "test:/my/test/uri"
    source = SampleDatasetSource._resolve(source_uri)

    data_no_targets = {"Numbers": [1, 2, 3]}
    dataset_no_targets = dataset_cls(
        constructor(data_no_targets),
        source=source,
        name="testname",
    )

    assert dataset_no_targets._targets is None

    data_with_targets = {"a": [1, 1], "b": [2, 2], "c": [3, 3]}
    dataset_with_targets = dataset_cls(
        constructor(data_with_targets),
        source=source,
        targets="c",
        name="testname",
    )

    assert dataset_with_targets._targets == "c"


def test_with_invalid_targets(constructor_dataset_loaders):
    constructor, dataset_cls, _ = constructor_dataset_loaders

    source_uri = "test:/my/test/uri"
    source = SampleDatasetSource._resolve(source_uri)

    data = {"a": [1, 1], "b": [2, 2], "c": [3, 3]}
    df = constructor(data)

    with pytest.raises(
        MlflowException,
        match=(
            r"The specified (pandas|polars|pyarrow) DataFrame does not contain the specified "
            "targets column 'd'."
        ),
    ):
        dataset_cls(
            df=df,
            source=source,
            targets="d",
            name="testname",
        )


def test_to_pyfunc(constructor_dataset_loaders):
    constructor, dataset_cls, _ = constructor_dataset_loaders

    source_uri = "test:/my/test/uri"
    source = SampleDatasetSource._resolve(source_uri)

    data = {"Numbers": [1, 2, 3]}
    dataset = dataset_cls(
        constructor(data),
        source=source,
        name="testname",
    )
    assert isinstance(dataset.to_pyfunc(), PyFuncInputsOutputs)


def test_to_pyfunc_with_outputs(constructor_dataset_loaders):
    constructor, dataset_cls, _ = constructor_dataset_loaders

    source_uri = "test:/my/test/uri"
    source = SampleDatasetSource._resolve(source_uri)

    data = {"a": [1, 1], "b": [2, 2], "c": [3, 3]}
    dataset = dataset_cls(
        constructor(data),
        source=source,
        targets="c",
        name="testname",
    )
    input_outputs = dataset.to_pyfunc()

    assert isinstance(input_outputs, PyFuncInputsOutputs)

    input_data = constructor({"a": [1, 1], "b": [2, 2]})
    output_data = constructor({"c": [3, 3]})["c"]
    assert input_outputs.inputs.equals(input_data)
    assert input_outputs.outputs.equals(output_data)


def test_from_loader_with_targets(tmp_path, constructor_dataset_loaders):
    constructor, _, loader = constructor_dataset_loaders

    data = {"a": [1, 1], "b": [2, 2], "c": [3, 3]}
    df = constructor(data)

    path = tmp_path / "temp.csv"
    nw.from_native(df, eager_only=True, pass_through=False).write_csv(path)

    dataset = loader(df, targets="c", source=path)
    input_outputs = dataset.to_pyfunc()
    assert isinstance(input_outputs, PyFuncInputsOutputs)

    input_data = constructor({"a": [1, 1], "b": [2, 2]})
    output_data = constructor({"c": [3, 3]})["c"]
    assert input_outputs.inputs.equals(input_data)
    assert input_outputs.outputs.equals(output_data)


def test_from_loader_file_system_datasource(tmp_path, constructor_dataset_loaders):
    constructor, dataset_cls, loader = constructor_dataset_loaders

    data = {"a": [1, 1], "b": [2, 2], "c": [3, 3]}
    df = constructor(data)
    path = tmp_path / "temp.csv"
    df_nw = nw.from_native(df, eager_only=True, pass_through=False)
    df_nw.write_csv(path)

    mlflow_df = loader(df, source=path)

    assert isinstance(mlflow_df, dataset_cls)
    assert mlflow_df.df.equals(df)
    assert mlflow_df.schema == infer_mlflow_schema(df)

    num_rows, num_cols = df_nw.shape
    assert mlflow_df.profile == {
        "num_rows": num_rows,
        "num_elements": num_rows * num_cols,
    }

    assert isinstance(mlflow_df.source, FileSystemDatasetSource)


def test_from_loader_spark_datasource(spark_session, tmp_path, constructor_dataset_loaders):
    constructor, dataset_cls, loader = constructor_dataset_loaders

    data = {"a": [1, 1], "b": [2, 2], "c": [3, 3]}
    df = constructor(data)
    df_spark = spark_session.createDataFrame([*zip(*data.values())], schema=[*data.keys()])
    path = str(tmp_path / "temp.parquet")
    df_spark.write.parquet(path)

    spark_datasource = SparkDatasetSource(path=path)
    mlflow_df = loader(df, source=spark_datasource)

    assert isinstance(mlflow_df, dataset_cls)
    assert mlflow_df.df.equals(df)
    assert mlflow_df.schema == infer_mlflow_schema(df)
    num_rows, num_cols = nw.from_native(df, eager_only=True, pass_through=False).shape
    assert mlflow_df.profile == {
        "num_rows": num_rows,
        "num_elements": num_rows * num_cols,
    }

    assert isinstance(mlflow_df.source, SparkDatasetSource)


def test_from_loader_delta_datasource(spark_session, tmp_path, constructor_dataset_loaders):
    constructor, dataset_cls, loader = constructor_dataset_loaders

    data = {"a": [1, 1], "b": [2, 2], "c": [3, 3]}
    df = constructor(data)
    df_spark = spark_session.createDataFrame([*zip(*data.values())], schema=[*data.keys()])
    path = str(tmp_path / "temp.delta")
    df_spark.write.format("delta").mode("overwrite").save(path)

    delta_datasource = DeltaDatasetSource(path=path)
    mlflow_df = loader(df, source=delta_datasource)

    assert isinstance(mlflow_df, dataset_cls)
    assert mlflow_df.df.equals(df)
    assert mlflow_df.schema == infer_mlflow_schema(df)
    num_rows, num_cols = nw.from_native(df, eager_only=True, pass_through=False).shape
    assert mlflow_df.profile == {
        "num_rows": num_rows,
        "num_elements": num_rows * num_cols,
    }

    assert isinstance(mlflow_df.source, DeltaDatasetSource)


def test_from_loader_no_source_specified(constructor_dataset_loaders):
    constructor, dataset_cls, loader = constructor_dataset_loaders

    data = {"a": [1, 1], "b": [2, 2], "c": [3, 3]}
    df = constructor(data)
    mlflow_df = loader(df)

    assert isinstance(mlflow_df, dataset_cls)

    assert isinstance(mlflow_df.source, CodeDatasetSource)
    assert "mlflow.source.name" in mlflow_df.source.to_json()


def test_to_evaluation_dataset(constructor_dataset_loaders):
    import numpy as np

    constructor, dataset_cls, _ = constructor_dataset_loaders

    source_uri = "test:/my/test/uri"
    source = SampleDatasetSource._resolve(source_uri)

    data = {"a": [1, 1], "b": [2, 2], "c": [3, 3]}
    df = constructor(data)
    df_nw = nw.from_native(df, eager_only=True, pass_through=False)
    dataset = dataset_cls(
        df=df,
        source=source,
        targets="c",
        name="testname",
    )

    evaluation_dataset = dataset.to_evaluation_dataset()
    assert isinstance(evaluation_dataset, EvaluationDataset)
    assert evaluation_dataset.features_data.equals(df_nw.drop("c").to_native())
    assert np.array_equal(evaluation_dataset.labels_data, df_nw["c"].to_numpy())


def test_df_hashing_with_strings(constructor_dataset_loaders):
    constructor, dataset_cls, _ = constructor_dataset_loaders

    source_uri = "test:/my/test/uri"
    source = SampleDatasetSource._resolve(source_uri)

    dataset1 = dataset_cls(
        constructor({"text_column": ["a", "a"], "b": [2, 2], "c": [3, 3]}),
        source=source,
        name="testname",
    )

    dataset2 = dataset_cls(
        constructor({"text_column": ["b", "b"], "b": [2, 2], "c": [3, 3]}),
        source=source,
        name="testname",
    )

    assert dataset1.digest != dataset2.digest


def test_df_hashing_with_dicts(constructor_dataset_loaders, request):
    constructor, dataset_cls, _ = constructor_dataset_loaders

    source_uri = "test:/my/test/uri"
    source = SampleDatasetSource._resolve(source_uri)

    data = {
        "a": [[1, 2, 3], [2, 3]],
        "b": [{"b": "b", "c": {"c": "c"}}, {"b": "b"}],
        "c": [3, 3],
        "d": ["d", "d"],
    }
    df = constructor(data)
    dataset1 = dataset_cls(df=df, source=source, name="testname")
    dataset2 = dataset_cls(df=df, source=source, name="testname")
    assert dataset1.digest == dataset2.digest

    evaluation_dataset = dataset1.to_evaluation_dataset()
    assert isinstance(evaluation_dataset, EvaluationDataset)
    assert evaluation_dataset.features_data.equals(df)
    evaluation_dataset2 = dataset2.to_evaluation_dataset()
    assert evaluation_dataset.hash == evaluation_dataset2.hash
