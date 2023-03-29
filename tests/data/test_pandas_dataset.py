import json

from mlflow.types.schema import Schema

import pandas as pd

from tests.resources.data.dataset_source import TestDatasetSource
from mlflow.data.pandas_dataset import PandasDataset
from mlflow.data.pyfunc_dataset_mixin import PyFuncInputsOutputs
from mlflow.data.filesystem_dataset_source import FileSystemDatasetSource
from mlflow.data.spark_dataset_source import SparkDatasetSource
# from mlflow.data.delta_dataset_source import DeltaDatasetSource

import mlflow.data
from mlflow.types.utils import _infer_schema


def test_conversion_to_json():
    source_uri = "test:/my/test/uri"
    source = TestDatasetSource._resolve(source_uri)

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
    source = TestDatasetSource._resolve(source_uri)
    dataset = PandasDataset(
        df=pd.DataFrame([1, 2, 3], columns=["Numbers"]),
        source=source,
        name="testname",
    )
    assert dataset.digest == dataset._compute_digest()
    assert dataset.digest == "31ccce44"


def test_df_property():
    source_uri = "test:/my/test/uri"
    source = TestDatasetSource._resolve(source_uri)
    df = pd.DataFrame([1, 2, 3], columns=["Numbers"])
    dataset = PandasDataset(
        df=df,
        source=source,
        name="testname",
    )
    assert dataset.df.equals(df)


def test_to_pyfunc():
    source_uri = "test:/my/test/uri"
    source = TestDatasetSource._resolve(source_uri)
    df = pd.DataFrame([1, 2, 3], columns=["Numbers"])
    dataset = PandasDataset(
        df=df,
        source=source,
        name="testname",
    )
    assert isinstance(dataset.to_pyfunc(), PyFuncInputsOutputs)


def test_to_pyfunc_with_outputs():
    source_uri = "test:/my/test/uri"
    source = TestDatasetSource._resolve(source_uri)
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


def test_from_pandas_spark_datasource(tmp_path):
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()

    df = pd.DataFrame([[1, 2, 3], [1, 2, 3]], columns=["a", "b", "c"])
    df_spark = spark.createDataFrame(df)

    path = str(tmp_path / "temp.parquet")
    df_spark.write.parquet(path)
    mlflow_df = mlflow.data.from_pandas(df, source=path)

    assert isinstance(mlflow_df, PandasDataset)
    assert mlflow_df.df.equals(df)
    assert mlflow_df.schema == _infer_schema(df)
    assert mlflow_df.profile == {
        "num_rows": len(df),
        "num_elements": df.size,
    }

    assert isinstance(mlflow_df.source, SparkDatasetSource)


# def test_from_pandas_delta_datasource(tmp_path):
#     from pyspark.sql import SparkSession

#     spark = SparkSession.builder.getOrCreate()


#     df = pd.DataFrame([[1, 2, 3], [1, 2, 3]], columns=["a", "b", "c"])
#     df_spark = spark.createDataFrame(df)

#     path = tmp_path / "temp"
#     df_spark.to_delta(path)
#     mlflow_df = mlflow.data.from_pandas(df, source=path)

#     assert isinstance(mlflow_df, PandasDataset)
#     assert mlflow_df.df.equals(df)
#     assert mlflow_df.schema == _infer_schema(df)
#     assert mlflow_df.profile == {
#         "num_rows": len(df),
#         "num_elements": df.size,
#     }

#     assert isinstance(mlflow_df.source, DeltaDatasetSource)
