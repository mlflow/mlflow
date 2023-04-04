import pytest
import pandas as pd

from mlflow.data.spark_dataset_source import SparkDatasetSource
from mlflow.exceptions import MlflowException


@pytest.fixture(scope="module", autouse=True)
def spark_session():
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[*]")
        .config("spark.jars.packages", "io.delta:delta-core_2.12:1.2.1")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        )
        .getOrCreate()
    )
    yield session
    session.stop()


def test_spark_dataset_source_from_path(spark_session, tmp_path):
    df = pd.DataFrame([[1, 2, 3], [1, 2, 3]], columns=["a", "b", "c"])
    df_spark = spark_session.createDataFrame(df)
    path = str(tmp_path / "temp.parquet")
    df_spark.write.parquet(path)

    spark_datasource = SparkDatasetSource(path=path)
    assert spark_datasource._to_dict() == {"path": path}
    loaded_df_spark = spark_datasource.load()
    assert loaded_df_spark.count() == df_spark.count()


def test_spark_dataset_source_from_table(spark_session, tmp_path):
    df = pd.DataFrame([[1, 2, 3], [1, 2, 3]], columns=["a", "b", "c"])
    df_spark = spark_session.createDataFrame(df)
    df_spark.write.mode("overwrite").saveAsTable("temp", path=tmp_path)

    spark_datasource = SparkDatasetSource(table_name="temp")
    assert spark_datasource._to_dict() == {"table_name": "temp"}
    loaded_df_spark = spark_datasource.load()
    assert loaded_df_spark.count() == df_spark.count()


def test_spark_dataset_source_from_sql(spark_session, tmp_path):
    df = pd.DataFrame([[1, 2, 3], [1, 2, 3]], columns=["a", "b", "c"])
    df_spark = spark_session.createDataFrame(df)
    df_spark.write.mode("overwrite").saveAsTable("temp_sql", path=tmp_path)

    spark_datasource = SparkDatasetSource(sql="SELECT * FROM temp_sql")
    assert spark_datasource._to_dict() == {"sql": "SELECT * FROM temp_sql"}
    loaded_df_spark = spark_datasource.load()
    assert loaded_df_spark.count() == df_spark.count()


def test_spark_dataset_source_too_many_inputs(spark_session, tmp_path):
    df = pd.DataFrame([[1, 2, 3], [1, 2, 3]], columns=["a", "b", "c"])
    df_spark = spark_session.createDataFrame(df)
    df_spark.write.mode("overwrite").saveAsTable("temp", path=tmp_path)

    with pytest.raises(
        MlflowException, match='Must specify exactly one of "path", "table_name", or "sql"'
    ):
        SparkDatasetSource(path=tmp_path, table_name="temp")
