import json
import pytest
import pandas as pd
import mlflow.data
from mlflow.data.spark_dataset import SparkDataset
from mlflow.data.spark_dataset_source import SparkDatasetSource
from mlflow.data.delta_dataset_source import DeltaDatasetSource
from mlflow.exceptions import MlflowException
from mlflow.models.evaluation.base import EvaluationDataset
from mlflow.types.schema import Schema
from mlflow.types.utils import _infer_schema


@pytest.fixture(scope="class", autouse=True)
def spark_session():
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[*]")
        .config("spark.jars.packages", "io.delta:delta-core_2.12:2.2.0")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        )
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture
def df():
    return pd.DataFrame([[1, 2, 3], [1, 2, 3]], columns=["a", "b", "c"])


def _assert_dataframes_equal(df1, df2):
    if df1.schema == df2.schema:
        diff = df1.exceptAll(df2)
        assert diff.rdd.isEmpty()
    else:
        assert False


def _check_spark_dataset(dataset, original_df, df_spark, expected_source_type):
    assert isinstance(dataset, SparkDataset)
    _assert_dataframes_equal(dataset.df, df_spark)
    assert dataset.schema == _infer_schema(original_df)
    assert dataset.profile == {"approx_count": 2}
    assert isinstance(dataset.source, expected_source_type)


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
    assert parsed_json["profile"] == json.dumps(dataset.profile)

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
    assert parsed_json["profile"] == json.dumps(dataset.profile)

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


def test_from_spark_with_no_source_info(spark_session, tmp_path, df):
    df_spark = spark_session.createDataFrame(df)
    path = str(tmp_path / "temp.parquet")
    df_spark.write.parquet(path)
    with pytest.raises(
        MlflowException,
        match="Must specify exactly one of `path`, `table_name`, or `sql`.",
    ):
        # pylint: disable=unused-variable
        mlflow_df = mlflow.data.from_spark(df_spark)


def test_from_spark_with_sql_and_version(spark_session, tmp_path, df):
    df_spark = spark_session.createDataFrame(df)
    path = str(tmp_path / "temp.parquet")
    df_spark.write.parquet(path)
    with pytest.raises(
        MlflowException,
        match="`version` may not be specified when `sql` is specified. `version` may only be"
        " specified when `table_name` or `path` is specified.",
    ):
        # pylint: disable=unused-variable
        mlflow_df = mlflow.data.from_spark(df_spark, sql="SELECT * FROM table", version=1)


def test_from_spark_path(spark_session, tmp_path, df):
    df_spark = spark_session.createDataFrame(df)
    path = str(tmp_path / "temp.parquet")
    df_spark.write.parquet(path)

    mlflow_df = mlflow.data.from_spark(df_spark, path=path)

    _check_spark_dataset(mlflow_df, df, df_spark, SparkDatasetSource)


def test_from_spark_delta_path(spark_session, tmp_path, df):
    df_spark = spark_session.createDataFrame(df)
    path = str(tmp_path / "temp.delta")
    df_spark.write.format("delta").save(path)

    mlflow_df = mlflow.data.from_spark(df_spark, path=path)

    _check_spark_dataset(mlflow_df, df, df_spark, DeltaDatasetSource)


def test_from_spark_sql(spark_session, tmp_path, df):
    df_spark = spark_session.createDataFrame(df)
    df_spark.createOrReplaceTempView("table")

    mlflow_df = mlflow.data.from_spark(df_spark, sql="SELECT * FROM table")

    _check_spark_dataset(mlflow_df, df, df_spark, SparkDatasetSource)


def test_from_spark_table_name(spark_session, tmp_path, df):
    df_spark = spark_session.createDataFrame(df)
    df_spark.createOrReplaceTempView("my_spark_table")

    mlflow_df = mlflow.data.from_spark(df_spark, table_name="my_spark_table")

    _check_spark_dataset(mlflow_df, df, df_spark, SparkDatasetSource)


def test_from_spark_table_name_with_version(spark_session, tmp_path, df):
    df_spark = spark_session.createDataFrame(df)
    df_spark.createOrReplaceTempView("my_spark_table")

    with pytest.raises(
        MlflowException,
        match="Version '1' was specified, but could not find a Delta table "
        "with name 'my_spark_table'",
    ):
        # pylint: disable=unused-variable
        mlflow_df = mlflow.data.from_spark(df_spark, table_name="my_spark_table", version=1)


def test_from_spark_delta_table_name(spark_session, tmp_path, df):
    df_spark = spark_session.createDataFrame(df)
    # write to delta table
    df_spark.write.format("delta").mode("overwrite").saveAsTable("my_delta_table")

    mlflow_df = mlflow.data.from_spark(df_spark, table_name="my_delta_table")

    _check_spark_dataset(mlflow_df, df, df_spark, DeltaDatasetSource)


def test_from_spark_delta_table_name_and_version(spark_session, tmp_path, df):
    df_spark = spark_session.createDataFrame(df)
    # write to delta table
    df_spark.write.format("delta").mode("overwrite").saveAsTable("my_delta_table")

    mlflow_df = mlflow.data.from_spark(df_spark, table_name="my_delta_table", version=1)

    _check_spark_dataset(mlflow_df, df, df_spark, DeltaDatasetSource)


def test_load_delta_with_no_source_info(spark_session, tmp_path):
    with pytest.raises(
        MlflowException,
        match="Must specify exactly one of `table_name` or `path`.",
    ):
        # pylint: disable=unused-variable
        mlflow_df = mlflow.data.load_delta()


def test_load_delta_with_both_table_name_and_path(spark_session, tmp_path):
    with pytest.raises(
        MlflowException,
        match="Must specify exactly one of `table_name` or `path`.",
    ):
        # pylint: disable=unused-variable
        mlflow_df = mlflow.data.load_delta(table_name="my_table", path="my_path")


def test_load_delta_path(spark_session, tmp_path, df):
    df_spark = spark_session.createDataFrame(df)
    path = str(tmp_path / "temp.delta")
    df_spark.write.format("delta").mode("overwrite").save(path)

    mlflow_df = mlflow.data.load_delta(path=path)

    _check_spark_dataset(mlflow_df, df, df_spark, DeltaDatasetSource)


def test_load_delta_path_with_version(spark_session, tmp_path, df):
    df_spark = spark_session.createDataFrame(df)
    path = str(tmp_path / "temp.delta")
    df_spark.write.format("delta").mode("overwrite").save(path)
    # write again to create a new version
    df_spark.write.format("delta").mode("overwrite").save(path)

    mlflow_df = mlflow.data.load_delta(path=path, version=1)

    _check_spark_dataset(mlflow_df, df, df_spark, DeltaDatasetSource)


def test_load_delta_table_name(spark_session, tmp_path, df):
    df_spark = spark_session.createDataFrame(df)
    # write to delta table
    df_spark.write.format("delta").mode("overwrite").saveAsTable("my_delta_table")

    mlflow_df = mlflow.data.load_delta(table_name="my_delta_table")

    _check_spark_dataset(mlflow_df, df, df_spark, DeltaDatasetSource)


def test_load_delta_table_name_with_version(spark_session, tmp_path, df):
    df_spark = spark_session.createDataFrame(df)
    # write to delta table
    df_spark.write.format("delta").mode("overwrite").saveAsTable("my_delta_table")

    mlflow_df = mlflow.data.load_delta(table_name="my_delta_table", version=1)

    _check_spark_dataset(mlflow_df, df, df_spark, DeltaDatasetSource)


def test_to_evaluation_dataset(spark_session, tmp_path, df):
    df_spark = spark_session.createDataFrame(df)
    path = str(tmp_path / "temp.parquet")
    df_spark.write.parquet(path)

    source = SparkDatasetSource(path=path)

    dataset = SparkDataset(
        df=df_spark,
        source=source,
        targets="c",
        name="testname",
    )
    evaluation_dataset = dataset.to_evaluation_dataset()
    assert isinstance(evaluation_dataset, EvaluationDataset)
