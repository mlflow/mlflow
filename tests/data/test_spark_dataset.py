import pytest
import pandas as pd
import mlflow.data
from mlflow.data.spark_dataset import SparkDataset
from mlflow.data.spark_dataset_source import SparkDatasetSource
from mlflow.data.delta_dataset_source import DeltaDatasetSource
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


def test_from_pandas_spark_datasource(spark_session, tmp_path):
    df = pd.DataFrame([[1, 2, 3], [1, 2, 3]], columns=["a", "b", "c"])
    df_spark = spark_session.createDataFrame(df)
    path = str(tmp_path / "temp.parquet")
    df_spark.write.parquet(path)

    mlflow_df = mlflow.data.from_spark(df_spark, path=path)

    assert isinstance(mlflow_df, SparkDataset)
    # TODO: figure out how to compare spark df
    # assert mlflow_df.df.equals(df)
    assert mlflow_df.schema == _infer_schema(df)
    assert mlflow_df.profile == {}

    assert isinstance(mlflow_df.source, SparkDatasetSource)
