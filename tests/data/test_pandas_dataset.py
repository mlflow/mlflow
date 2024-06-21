import json

import pandas as pd
import pytest

import mlflow.data
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.delta_dataset_source import DeltaDatasetSource
from mlflow.data.evaluation_dataset import EvaluationDataset
from mlflow.data.filesystem_dataset_source import FileSystemDatasetSource
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.data.pyfunc_dataset_mixin import PyFuncInputsOutputs
from mlflow.data.spark_dataset_source import SparkDatasetSource
from mlflow.exceptions import MlflowException
from mlflow.types.schema import Schema
from mlflow.types.utils import _infer_schema

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


def test_conversion_to_json():
    source_uri = "test:/my/test/uri"
    source = SampleDatasetSource._resolve(source_uri)

    dataset = PandasDataset(
        df=pd.DataFrame([1, 2, 3], columns=["Numbers"]),
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


def test_digest_property_has_expected_value():
    source_uri = "test:/my/test/uri"
    source = SampleDatasetSource._resolve(source_uri)
    dataset = PandasDataset(
        df=pd.DataFrame([1, 2, 3], columns=["Numbers"]),
        source=source,
        name="testname",
    )
    assert dataset.digest == dataset._compute_digest()
    assert dataset.digest == "31ccce44"


def test_df_property():
    source_uri = "test:/my/test/uri"
    source = SampleDatasetSource._resolve(source_uri)
    df = pd.DataFrame([1, 2, 3], columns=["Numbers"])
    dataset = PandasDataset(
        df=df,
        source=source,
        name="testname",
    )
    assert dataset.df.equals(df)


def test_targets_property():
    source_uri = "test:/my/test/uri"
    source = SampleDatasetSource._resolve(source_uri)
    df_no_targets = pd.DataFrame([1, 2, 3], columns=["Numbers"])
    dataset_no_targets = PandasDataset(
        df=df_no_targets,
        source=source,
        name="testname",
    )
    assert dataset_no_targets._targets is None
    df_with_targets = pd.DataFrame([[1, 2, 3], [1, 2, 3]], columns=["a", "b", "c"])
    dataset_with_targets = PandasDataset(
        df=df_with_targets,
        source=source,
        targets="c",
        name="testname",
    )
    assert dataset_with_targets._targets == "c"


def test_with_invalid_targets():
    source_uri = "test:/my/test/uri"
    source = SampleDatasetSource._resolve(source_uri)
    df = pd.DataFrame([[1, 2, 3], [1, 2, 3]], columns=["a", "b", "c"])
    with pytest.raises(
        MlflowException,
        match="The specified pandas DataFrame does not contain the specified targets column 'd'.",
    ):
        PandasDataset(
            df=df,
            source=source,
            targets="d",
            name="testname",
        )


def test_to_pyfunc():
    source_uri = "test:/my/test/uri"
    source = SampleDatasetSource._resolve(source_uri)
    df = pd.DataFrame([1, 2, 3], columns=["Numbers"])
    dataset = PandasDataset(
        df=df,
        source=source,
        name="testname",
    )
    assert isinstance(dataset.to_pyfunc(), PyFuncInputsOutputs)


def test_to_pyfunc_with_outputs():
    source_uri = "test:/my/test/uri"
    source = SampleDatasetSource._resolve(source_uri)
    df = pd.DataFrame([[1, 2, 3], [1, 2, 3]], columns=["a", "b", "c"])
    dataset = PandasDataset(
        df=df,
        source=source,
        targets="c",
        name="testname",
    )
    input_outputs = dataset.to_pyfunc()
    assert isinstance(input_outputs, PyFuncInputsOutputs)
    assert input_outputs.inputs.equals(pd.DataFrame([[1, 2], [1, 2]], columns=["a", "b"]))
    assert input_outputs.outputs.equals(pd.Series([3, 3], name="c"))


def test_from_pandas_with_targets(tmp_path):
    df = pd.DataFrame([[1, 2, 3], [1, 2, 3]], columns=["a", "b", "c"])
    path = tmp_path / "temp.csv"
    df.to_csv(path)
    dataset = mlflow.data.from_pandas(df, targets="c", source=path)
    input_outputs = dataset.to_pyfunc()
    assert isinstance(input_outputs, PyFuncInputsOutputs)
    assert input_outputs.inputs.equals(pd.DataFrame([[1, 2], [1, 2]], columns=["a", "b"]))
    assert input_outputs.outputs.equals(pd.Series([3, 3], name="c"))


def test_from_pandas_file_system_datasource(tmp_path):
    df = pd.DataFrame([[1, 2, 3], [1, 2, 3]], columns=["a", "b", "c"])
    path = tmp_path / "temp.csv"
    df.to_csv(path)
    mlflow_df = mlflow.data.from_pandas(df, source=path)

    assert isinstance(mlflow_df, PandasDataset)
    assert mlflow_df.df.equals(df)
    assert mlflow_df.schema == _infer_schema(df)
    assert mlflow_df.profile == {
        "num_rows": len(df),
        "num_elements": df.size,
    }

    assert isinstance(mlflow_df.source, FileSystemDatasetSource)


def test_from_pandas_spark_datasource(spark_session, tmp_path):
    df = pd.DataFrame([[1, 2, 3], [1, 2, 3]], columns=["a", "b", "c"])
    df_spark = spark_session.createDataFrame(df)
    path = str(tmp_path / "temp.parquet")
    df_spark.write.parquet(path)

    spark_datasource = SparkDatasetSource(path=path)
    mlflow_df = mlflow.data.from_pandas(df, source=spark_datasource)

    assert isinstance(mlflow_df, PandasDataset)
    assert mlflow_df.df.equals(df)
    assert mlflow_df.schema == _infer_schema(df)
    assert mlflow_df.profile == {
        "num_rows": len(df),
        "num_elements": df.size,
    }

    assert isinstance(mlflow_df.source, SparkDatasetSource)


def test_from_pandas_delta_datasource(spark_session, tmp_path):
    df = pd.DataFrame([[1, 2, 3], [1, 2, 3]], columns=["a", "b", "c"])
    df_spark = spark_session.createDataFrame(df)
    path = str(tmp_path / "temp.delta")
    df_spark.write.format("delta").mode("overwrite").save(path)

    delta_datasource = DeltaDatasetSource(path=path)
    mlflow_df = mlflow.data.from_pandas(df, source=delta_datasource)

    assert isinstance(mlflow_df, PandasDataset)
    assert mlflow_df.df.equals(df)
    assert mlflow_df.schema == _infer_schema(df)
    assert mlflow_df.profile == {
        "num_rows": len(df),
        "num_elements": df.size,
    }

    assert isinstance(mlflow_df.source, DeltaDatasetSource)


def test_from_pandas_no_source_specified():
    df = pd.DataFrame([[1, 2, 3], [1, 2, 3]], columns=["a", "b", "c"])
    mlflow_df = mlflow.data.from_pandas(df)

    assert isinstance(mlflow_df, PandasDataset)

    assert isinstance(mlflow_df.source, CodeDatasetSource)
    assert "mlflow.source.name" in mlflow_df.source.to_json()


def test_to_evaluation_dataset():
    import numpy as np

    source_uri = "test:/my/test/uri"
    source = SampleDatasetSource._resolve(source_uri)
    df = pd.DataFrame([[1, 2, 3], [1, 2, 3]], columns=["a", "b", "c"])
    dataset = PandasDataset(
        df=df,
        source=source,
        targets="c",
        name="testname",
    )
    evaluation_dataset = dataset.to_evaluation_dataset()
    assert isinstance(evaluation_dataset, EvaluationDataset)
    assert evaluation_dataset.features_data.equals(df.drop("c", axis=1))
    assert np.array_equal(evaluation_dataset.labels_data, df["c"].to_numpy())


def test_df_hashing_with_strings():
    source_uri = "test:/my/test/uri"
    source = SampleDatasetSource._resolve(source_uri)

    dataset1 = PandasDataset(
        df=pd.DataFrame([["a", 2, 3], ["a", 2, 3]], columns=["text_column", "b", "c"]),
        source=source,
        name="testname",
    )

    dataset2 = PandasDataset(
        df=pd.DataFrame([["b", 2, 3], ["b", 2, 3]], columns=["text_column", "b", "c"]),
        source=source,
        name="testname",
    )

    assert dataset1.digest != dataset2.digest


def test_df_hashing_with_dicts():
    source_uri = "test:/my/test/uri"
    source = SampleDatasetSource._resolve(source_uri)

    df = pd.DataFrame(
        [
            {"a": [1, 2, 3], "b": {"b": "b", "c": {"c": "c"}}, "c": 3, "d": "d"},
            {"a": [2, 3], "b": {"b": "b"}, "c": 3, "d": "d"},
        ]
    )
    dataset1 = PandasDataset(df=df, source=source, name="testname")
    dataset2 = PandasDataset(df=df, source=source, name="testname")
    assert dataset1.digest == dataset2.digest

    evaluation_dataset = dataset1.to_evaluation_dataset()
    assert isinstance(evaluation_dataset, EvaluationDataset)
    assert evaluation_dataset.features_data.equals(df)
    evaluation_dataset2 = dataset2.to_evaluation_dataset()
    assert evaluation_dataset.hash == evaluation_dataset2.hash
