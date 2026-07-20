import json

import pandas as pd
import pytest

from mlflow.data.dataset_source_registry import get_dataset_source_from_json
from mlflow.data.spark_dataset_source import SparkDatasetSource
from mlflow.exceptions import MlflowException


@pytest.fixture(scope="module")
def spark_session(tmp_path_factory: pytest.TempPathFactory):
    from pyspark.sql import SparkSession

    # Isolate the warehouse and embedded Derby metastore under a per-module tmp dir. Both
    # default to the process cwd (./spark-warehouse, ./metastore_db), which every xdist
    # worker shares, so concurrent workers collide on the Derby lock. Isolating them lets
    # this run in the parallel pass. See _XDIST_SERIAL_PATHS in tests/conftest.py.
    tmp_dir = tmp_path_factory.mktemp("spark_tmp")
    with (
        SparkSession.builder
        .master("local[*]")
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.0.0")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        )
        .config("spark.sql.warehouse.dir", str(tmp_dir))
        .config(
            "javax.jdo.option.ConnectionURL",
            f"jdbc:derby:;databaseName={tmp_dir}/metastore_db;create=true",
        )
        .getOrCreate()
    ) as session:
        yield session


def test_spark_dataset_source_from_path(spark_session, tmp_path):
    df = pd.DataFrame([[1, 2, 3], [1, 2, 3]], columns=["a", "b", "c"])
    df_spark = spark_session.createDataFrame(df)
    path = str(tmp_path / "temp.parquet")
    df_spark.write.parquet(path)

    spark_datasource = SparkDatasetSource(path=path)
    assert spark_datasource.to_json() == json.dumps({"path": path})
    loaded_df_spark = spark_datasource.load()
    assert loaded_df_spark.count() == df_spark.count()

    reloaded_source = get_dataset_source_from_json(
        spark_datasource.to_json(), source_type=spark_datasource._get_source_type()
    )
    assert isinstance(reloaded_source, SparkDatasetSource)
    assert type(spark_datasource) == type(reloaded_source)
    assert reloaded_source.to_json() == spark_datasource.to_json()


def test_spark_dataset_source_from_table(spark_session, tmp_path):
    df = pd.DataFrame([[1, 2, 3], [1, 2, 3]], columns=["a", "b", "c"])
    df_spark = spark_session.createDataFrame(df)
    df_spark.write.mode("overwrite").saveAsTable("temp", path=tmp_path)

    spark_datasource = SparkDatasetSource(table_name="temp")
    assert spark_datasource.to_json() == json.dumps({"table_name": "temp"})
    loaded_df_spark = spark_datasource.load()
    assert loaded_df_spark.count() == df_spark.count()

    reloaded_source = get_dataset_source_from_json(
        spark_datasource.to_json(), source_type=spark_datasource._get_source_type()
    )
    assert isinstance(reloaded_source, SparkDatasetSource)
    assert type(spark_datasource) == type(reloaded_source)
    assert reloaded_source.to_json() == spark_datasource.to_json()


def test_spark_dataset_source_from_sql(spark_session, tmp_path):
    df = pd.DataFrame([[1, 2, 3], [1, 2, 3]], columns=["a", "b", "c"])
    df_spark = spark_session.createDataFrame(df)
    df_spark.write.mode("overwrite").saveAsTable("temp_sql", path=tmp_path)

    spark_datasource = SparkDatasetSource(sql="SELECT * FROM temp_sql")
    assert spark_datasource.to_json() == json.dumps({"sql": "SELECT * FROM temp_sql"})
    loaded_df_spark = spark_datasource.load()
    assert loaded_df_spark.count() == df_spark.count()

    reloaded_source = get_dataset_source_from_json(
        spark_datasource.to_json(), source_type=spark_datasource._get_source_type()
    )
    assert isinstance(reloaded_source, SparkDatasetSource)
    assert type(spark_datasource) == type(reloaded_source)
    assert reloaded_source.to_json() == spark_datasource.to_json()


def test_spark_dataset_source_too_many_inputs(spark_session, tmp_path):
    df = pd.DataFrame([[1, 2, 3], [1, 2, 3]], columns=["a", "b", "c"])
    df_spark = spark_session.createDataFrame(df)
    df_spark.write.mode("overwrite").saveAsTable("temp", path=tmp_path)

    with pytest.raises(
        MlflowException, match='Must specify exactly one of "path", "table_name", or "sql"'
    ):
        SparkDatasetSource(path=tmp_path, table_name="temp")
