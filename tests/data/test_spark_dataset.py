import json
import os
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

import mlflow.data
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.delta_dataset_source import DeltaDatasetSource
from mlflow.data.evaluation_dataset import EvaluationDataset
from mlflow.data.spark_dataset import SparkDataset
from mlflow.data.spark_dataset_source import SparkDatasetSource
from mlflow.exceptions import MlflowException
from mlflow.types.schema import Schema
from mlflow.types.utils import _infer_schema

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


@pytest.fixture(scope="module")
def spark_session(tmp_path_factory: pytest.TempPathFactory):
    import pyspark
    from packaging.version import Version
    from pyspark.sql import SparkSession

    pyspark_version = Version(pyspark.__version__)
    if pyspark_version.major >= 4:
        delta_package = "io.delta:delta-spark_2.13:4.0.0"
    else:
        delta_package = "io.delta:delta-spark_2.12:3.0.0"

    tmp_dir = tmp_path_factory.mktemp("spark_tmp")
    with (
        SparkSession.builder.master("local[*]")
        .config("spark.jars.packages", delta_package)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        )
        .config("spark.sql.warehouse.dir", str(tmp_dir))
        .getOrCreate()
    ) as session:
        yield session


@pytest.fixture(autouse=True)
def drop_tables(spark_session: "SparkSession"):
    yield
    for row in spark_session.sql("SHOW TABLES").collect():
        spark_session.sql(f"DROP TABLE IF EXISTS {row.tableName}")


@pytest.fixture
def df():
    return pd.DataFrame([[1, 2, 3], [1, 2, 3]], columns=["a", "b", "c"])


def _assert_dataframes_equal(df1, df2):
    if df1.schema == df2.schema:
        diff = df1.exceptAll(df2)
        assert diff.rdd.isEmpty()
    else:
        assert False


def _validate_profile_approx_count(parsed_json: dict[str, Any]) -> None:
    """Validate approx_count in profile data, handling platform/version differences."""
    # On Windows with certain PySpark versions, Spark datasets may return "unknown" for approx_count
    # instead of the actual count. We should check that the profile is valid JSON and contains
    # the expected key, but not assert on the exact value.
    profile_data = json.loads(parsed_json["profile"])
    assert "approx_count" in profile_data
    assert profile_data["approx_count"] in [1, 2, "unknown"]


def _check_spark_dataset(dataset, original_df, df_spark, expected_source_type, expected_name=None):
    assert isinstance(dataset, SparkDataset)
    _assert_dataframes_equal(dataset.df, df_spark)
    assert dataset.schema == _infer_schema(original_df)
    assert isinstance(dataset.profile, dict)
    approx_count = dataset.profile.get("approx_count")
    assert isinstance(approx_count, int) or approx_count == "unknown"
    assert isinstance(dataset.source, expected_source_type)
    # NB: In real-world scenarios, Spark dataset sources may not match Spark DataFrames precisely.
    # For example, users may transform Spark DataFrames after loading contents from source files.
    # To ensure that source loading works properly for the purpose of the test cases in this suite,
    # we require the source to match the DataFrame and make the following equality assertion
    _assert_dataframes_equal(dataset.source.load(), df_spark)
    if expected_name is not None:
        assert dataset.name == expected_name


def test_conversion_to_json_spark_dataset_source(spark_session, tmp_path, df):
    df_spark = spark_session.createDataFrame(df)
    path = str(tmp_path / "temp.parquet")
    df_spark.write.parquet(path)

    source = SparkDatasetSource(path=path)

    dataset = SparkDataset(
        df=df_spark,
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
    _validate_profile_approx_count(parsed_json)

    schema_json = json.dumps(json.loads(parsed_json["schema"])["mlflow_colspec"])
    assert Schema.from_json(schema_json) == dataset.schema


def test_conversion_to_json_delta_dataset_source(spark_session, tmp_path, df):
    df_spark = spark_session.createDataFrame(df)
    path = str(tmp_path / "temp.parquet")
    df_spark.write.format("delta").save(path)

    source = DeltaDatasetSource(path=path)

    dataset = SparkDataset(
        df=df_spark,
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
    _validate_profile_approx_count(parsed_json)

    schema_json = json.dumps(json.loads(parsed_json["schema"])["mlflow_colspec"])
    assert Schema.from_json(schema_json) == dataset.schema


def test_digest_property_has_expected_value(spark_session, tmp_path, df):
    df_spark = spark_session.createDataFrame(df)
    path = str(tmp_path / "temp.parquet")
    df_spark.write.parquet(path)

    source = SparkDatasetSource(path=path)

    dataset = SparkDataset(
        df=df_spark,
        source=source,
        name="testname",
    )
    assert dataset.digest == dataset._compute_digest()
    # Note that digests are stable within a session, but may not be stable across sessions
    # Hence we are not checking the digest value here


def test_df_property_has_expected_value(spark_session, tmp_path, df):
    df_spark = spark_session.createDataFrame(df)
    path = str(tmp_path / "temp.parquet")
    df_spark.write.parquet(path)

    source = SparkDatasetSource(path=path)

    dataset = SparkDataset(
        df=df_spark,
        source=source,
        name="testname",
    )
    assert dataset.df == df_spark


def test_targets_property(spark_session, tmp_path, df):
    df_spark = spark_session.createDataFrame(df)
    path = str(tmp_path / "temp.parquet")
    df_spark.write.parquet(path)

    source = SparkDatasetSource(path=path)
    dataset_no_targets = SparkDataset(
        df=df_spark,
        source=source,
        name="testname",
    )
    assert dataset_no_targets.targets is None
    dataset_with_targets = SparkDataset(
        df=df_spark,
        source=source,
        targets="c",
        name="testname",
    )
    assert dataset_with_targets.targets == "c"

    with pytest.raises(
        MlflowException,
        match="The specified Spark dataset does not contain the specified targets column",
    ):
        SparkDataset(
            df=df_spark,
            source=source,
            targets="nonexistent",
            name="testname",
        )


def test_predictions_property(spark_session, tmp_path, df):
    df_spark = spark_session.createDataFrame(df)
    path = str(tmp_path / "temp.parquet")
    df_spark.write.parquet(path)

    source = SparkDatasetSource(path=path)
    dataset_no_predictions = SparkDataset(
        df=df_spark,
        source=source,
        name="testname",
    )
    assert dataset_no_predictions.predictions is None
    dataset_with_predictions = SparkDataset(
        df=df_spark,
        source=source,
        predictions="b",
        name="testname",
    )
    assert dataset_with_predictions.predictions == "b"

    with pytest.raises(
        MlflowException,
        match="The specified Spark dataset does not contain the specified predictions column",
    ):
        SparkDataset(
            df=df_spark,
            source=source,
            predictions="nonexistent",
            name="testname",
        )


def test_from_spark_no_source_specified(spark_session, df):
    df_spark = spark_session.createDataFrame(df)
    mlflow_df = mlflow.data.from_spark(df_spark)

    assert isinstance(mlflow_df, SparkDataset)

    assert isinstance(mlflow_df.source, CodeDatasetSource)
    assert "mlflow.source.name" in mlflow_df.source.to_json()


def test_from_spark_with_sql_and_version(spark_session, tmp_path, df):
    df_spark = spark_session.createDataFrame(df)
    path = str(tmp_path / "temp.parquet")
    df_spark.write.parquet(path)
    with pytest.raises(
        MlflowException,
        match="`version` may not be specified when `sql` is specified. `version` may only be"
        " specified when `table_name` or `path` is specified.",
    ):
        mlflow.data.from_spark(df_spark, sql="SELECT * FROM table", version=1)


def test_from_spark_path(spark_session, tmp_path, df):
    df_spark = spark_session.createDataFrame(df)
    dir_path = str(tmp_path / "df_dir")
    df_spark.write.parquet(dir_path)
    assert os.path.isdir(dir_path)

    mlflow_df_from_dir = mlflow.data.from_spark(df_spark, path=dir_path)
    _check_spark_dataset(mlflow_df_from_dir, df, df_spark, SparkDatasetSource)

    file_path = str(tmp_path / "df.parquet")
    df_spark.toPandas().to_parquet(file_path)
    assert not os.path.isdir(file_path)

    mlflow_df_from_file = mlflow.data.from_spark(df_spark, path=file_path)
    _check_spark_dataset(mlflow_df_from_file, df, df_spark, SparkDatasetSource)


def test_from_spark_delta_path(spark_session, tmp_path, df):
    df_spark = spark_session.createDataFrame(df)
    path = str(tmp_path / "temp.delta")
    df_spark.write.format("delta").save(path)

    mlflow_df = mlflow.data.from_spark(df_spark, path=path)

    _check_spark_dataset(mlflow_df, df, df_spark, DeltaDatasetSource)


def test_from_spark_sql(spark_session, df):
    df_spark = spark_session.createDataFrame(df)
    df_spark.createOrReplaceTempView("table")

    mlflow_df = mlflow.data.from_spark(df_spark, sql="SELECT * FROM table")

    _check_spark_dataset(mlflow_df, df, df_spark, SparkDatasetSource)


def test_from_spark_table_name(spark_session, df):
    df_spark = spark_session.createDataFrame(df)
    df_spark.createOrReplaceTempView("my_spark_table")

    mlflow_df = mlflow.data.from_spark(df_spark, table_name="my_spark_table")

    _check_spark_dataset(mlflow_df, df, df_spark, SparkDatasetSource)


def test_from_spark_table_name_with_version(spark_session, df):
    df_spark = spark_session.createDataFrame(df)
    df_spark.createOrReplaceTempView("my_spark_table")

    with pytest.raises(
        MlflowException,
        match="Version '1' was specified, but could not find a Delta table "
        "with name 'my_spark_table'",
    ):
        mlflow.data.from_spark(df_spark, table_name="my_spark_table", version=1)


def test_from_spark_delta_table_name(spark_session, df):
    df_spark = spark_session.createDataFrame(df)
    # write to delta table
    df_spark.write.format("delta").mode("overwrite").saveAsTable("my_delta_table")

    mlflow_df = mlflow.data.from_spark(df_spark, table_name="my_delta_table")

    _check_spark_dataset(mlflow_df, df, df_spark, DeltaDatasetSource)


def test_from_spark_delta_table_name_and_version(spark_session, df):
    df_spark = spark_session.createDataFrame(df)
    # write to delta table
    df_spark.write.format("delta").mode("overwrite").saveAsTable("my_delta_table")

    mlflow_df = mlflow.data.from_spark(df_spark, table_name="my_delta_table", version=0)

    _check_spark_dataset(mlflow_df, df, df_spark, DeltaDatasetSource)


def test_load_delta_with_no_source_info():
    with pytest.raises(
        MlflowException,
        match="Must specify exactly one of `table_name` or `path`.",
    ):
        mlflow.data.load_delta()


def test_load_delta_with_both_table_name_and_path():
    with pytest.raises(
        MlflowException,
        match="Must specify exactly one of `table_name` or `path`.",
    ):
        mlflow.data.load_delta(table_name="my_table", path="my_path")


def test_load_delta_path(spark_session, tmp_path, df):
    df_spark = spark_session.createDataFrame(df)
    path = str(tmp_path / "temp.delta")
    df_spark.write.format("delta").mode("overwrite").save(path)

    mlflow_df = mlflow.data.load_delta(path=path)

    _check_spark_dataset(mlflow_df, df, df_spark, DeltaDatasetSource)


def test_load_delta_path_with_version(spark_session, tmp_path, df):
    path = str(tmp_path / "temp.delta")

    df_v0 = pd.DataFrame([[4, 5, 6], [4, 5, 6]], columns=["a", "b", "c"])
    assert not df_v0.equals(df)
    df_v0_spark = spark_session.createDataFrame(df_v0)
    df_v0_spark.write.format("delta").mode("overwrite").save(path)

    # write again to create a new version
    df_v1_spark = spark_session.createDataFrame(df)
    df_v1_spark.write.format("delta").mode("overwrite").save(path)

    mlflow_df = mlflow.data.load_delta(path=path, version=1)
    _check_spark_dataset(mlflow_df, df, df_v1_spark, DeltaDatasetSource)


def test_load_delta_table_name(spark_session, df):
    df_spark = spark_session.createDataFrame(df)
    # write to delta table
    df_spark.write.format("delta").mode("overwrite").saveAsTable("my_delta_table")

    mlflow_df = mlflow.data.load_delta(table_name="my_delta_table")

    _check_spark_dataset(mlflow_df, df, df_spark, DeltaDatasetSource, "my_delta_table@v0")


def test_load_delta_table_name_with_version(spark_session, df):
    df_spark = spark_session.createDataFrame(df)
    df_spark.write.format("delta").mode("overwrite").saveAsTable("my_delta_table_versioned")

    df2 = pd.DataFrame([[4, 5, 6], [4, 5, 6]], columns=["a", "b", "c"])
    assert not df2.equals(df)
    df2_spark = spark_session.createDataFrame(df2)
    df2_spark.write.format("delta").mode("overwrite").saveAsTable("my_delta_table_versioned")

    mlflow_df = mlflow.data.load_delta(table_name="my_delta_table_versioned", version=1)

    _check_spark_dataset(
        mlflow_df, df2, df2_spark, DeltaDatasetSource, "my_delta_table_versioned@v1"
    )
    pd.testing.assert_frame_equal(mlflow_df.df.toPandas(), df2)


def test_to_evaluation_dataset(spark_session, tmp_path, df):
    import numpy as np

    df_spark = spark_session.createDataFrame(df)
    path = str(tmp_path / "temp.parquet")
    df_spark.write.parquet(path)

    source = SparkDatasetSource(path=path)

    dataset = SparkDataset(
        df=df_spark,
        source=source,
        targets="c",
        name="testname",
        predictions="b",
    )
    evaluation_dataset = dataset.to_evaluation_dataset()
    assert isinstance(evaluation_dataset, EvaluationDataset)
    assert evaluation_dataset.features_data.equals(df_spark.toPandas()[["a"]])
    assert np.array_equal(evaluation_dataset.labels_data, df_spark.toPandas()["c"].values)
    assert np.array_equal(evaluation_dataset.predictions_data, df_spark.toPandas()["b"].values)
