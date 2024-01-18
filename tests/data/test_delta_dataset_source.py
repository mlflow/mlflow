import json
from unittest import mock

import pandas as pd
import pytest

from mlflow.data.dataset_source_registry import get_dataset_source_from_json
from mlflow.data.delta_dataset_source import DeltaDatasetSource
from mlflow.exceptions import MlflowException


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


def test_delta_dataset_source_from_path(spark_session, tmp_path):
    df = pd.DataFrame([[1, 2, 3], [1, 2, 3]], columns=["a", "b", "c"])
    df_spark = spark_session.createDataFrame(df)
    path = str(tmp_path / "temp.delta")
    df_spark.write.format("delta").mode("overwrite").save(path)

    delta_datasource = DeltaDatasetSource(path=path)
    loaded_df_spark = delta_datasource.load()
    assert loaded_df_spark.count() == df_spark.count()
    assert delta_datasource.to_json() == json.dumps(
        {
            "path": path,
        }
    )

    reloaded_source = get_dataset_source_from_json(
        delta_datasource.to_json(), source_type=delta_datasource._get_source_type()
    )
    assert isinstance(reloaded_source, DeltaDatasetSource)
    assert type(delta_datasource) == type(reloaded_source)
    assert reloaded_source.to_json() == delta_datasource.to_json()


def test_delta_dataset_source_from_table(spark_session, tmp_path):
    df = pd.DataFrame([[1, 2, 3], [1, 2, 3]], columns=["a", "b", "c"])
    df_spark = spark_session.createDataFrame(df)
    df_spark.write.format("delta").mode("overwrite").saveAsTable(
        "default.temp_delta", path=tmp_path
    )

    delta_datasource = DeltaDatasetSource(delta_table_name="temp_delta")
    loaded_df_spark = delta_datasource.load()
    assert loaded_df_spark.count() == df_spark.count()
    assert delta_datasource.to_json() == json.dumps(
        {
            "delta_table_name": "temp_delta",
        }
    )

    reloaded_source = get_dataset_source_from_json(
        delta_datasource.to_json(), source_type=delta_datasource._get_source_type()
    )
    assert isinstance(reloaded_source, DeltaDatasetSource)
    assert type(delta_datasource) == type(reloaded_source)
    assert reloaded_source.to_json() == delta_datasource.to_json()


def test_delta_dataset_source_from_table_versioned(spark_session, tmp_path):
    df = pd.DataFrame([[1, 2, 3], [1, 2, 3]], columns=["a", "b", "c"])
    df_spark = spark_session.createDataFrame(df)
    df_spark.write.format("delta").mode("overwrite").saveAsTable(
        "default.temp_delta_versioned", path=tmp_path
    )

    df2 = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "c"])
    df2_spark = spark_session.createDataFrame(df2)
    df2_spark.write.format("delta").mode("overwrite").saveAsTable(
        "default.temp_delta_versioned", path=tmp_path
    )

    delta_datasource = DeltaDatasetSource(
        delta_table_name="temp_delta_versioned", delta_table_version=1
    )
    loaded_df_spark = delta_datasource.load()
    assert loaded_df_spark.count() == df2_spark.count()
    assert delta_datasource.to_json() == json.dumps(
        {
            "delta_table_name": "temp_delta_versioned",
            "delta_table_version": 1,
        }
    )

    reloaded_source = get_dataset_source_from_json(
        delta_datasource.to_json(), source_type=delta_datasource._get_source_type()
    )
    assert isinstance(reloaded_source, DeltaDatasetSource)
    assert type(delta_datasource) == type(reloaded_source)
    assert reloaded_source.to_json() == delta_datasource.to_json()


def test_delta_dataset_source_too_many_inputs(spark_session, tmp_path):
    df = pd.DataFrame([[1, 2, 3], [1, 2, 3]], columns=["a", "b", "c"])
    df_spark = spark_session.createDataFrame(df)
    df_spark.write.format("delta").mode("overwrite").saveAsTable(
        "default.temp_delta_too_many_inputs", path=tmp_path
    )

    with pytest.raises(MlflowException, match='Must specify exactly one of "path" or "table_name"'):
        DeltaDatasetSource(path=tmp_path, delta_table_name="temp_delta_too_many_inputs")


def test_uc_table_id_retrieval_works(spark_session, tmp_path):
    def mock_resolve_table_name(table_name):
        if table_name == "temp_delta_versioned_with_id":
            return "default.temp_delta_versioned_with_id"
        return table_name

    def mock_lookup_table_id(table_name):
        if table_name == "default.temp_delta_versioned_with_id":
            return "uc_table_id_1"
        return None

    def mock_is_databricks_uc_table():
        return True

    with mock.patch(
        "mlflow.data.delta_dataset_source.DeltaDatasetSource._resolve_table_name",
        side_effect=mock_resolve_table_name,
    ):
        with mock.patch(
            "mlflow.data.delta_dataset_source.DeltaDatasetSource._lookup_table_id",
            side_effect=mock_lookup_table_id,
        ):
            with mock.patch(
                "mlflow.data.delta_dataset_source.DeltaDatasetSource._is_databricks_uc_table",
                side_effect=mock_is_databricks_uc_table,
            ):
                df = pd.DataFrame([[1, 2, 3], [1, 2, 3]], columns=["a", "b", "c"])
                df_spark = spark_session.createDataFrame(df)
                df_spark.write.format("delta").mode("overwrite").saveAsTable(
                    "default.temp_delta_versioned_with_id", path=tmp_path
                )

                df2 = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "c"])
                df2_spark = spark_session.createDataFrame(df2)
                df2_spark.write.format("delta").mode("overwrite").saveAsTable(
                    "default.temp_delta_versioned_with_id", path=tmp_path
                )

                delta_datasource = DeltaDatasetSource(
                    delta_table_name="temp_delta_versioned_with_id", delta_table_version=1
                )
                loaded_df_spark = delta_datasource.load()
                assert loaded_df_spark.count() == df2_spark.count()
                assert delta_datasource.to_json() == json.dumps(
                    {
                        "delta_table_name": "default.temp_delta_versioned_with_id",
                        "delta_table_version": 1,
                        "is_databricks_uc_table": True,
                        "delta_table_id": "uc_table_id_1",
                    }
                )


def test_uc_table_id_retrieval_throws(spark_session, tmp_path):
    def mock_resolve_table_name(table_name):
        if table_name == "temp_delta_versioned_with_id_throws":
            return "default.temp_delta_versioned_with_id_throws"
        return table_name

    def mock_lookup_table_id(table_name):
        if table_name == "default.temp_delta_versioned_with_id_throws":
            raise Exception("Table id fails.")
        return None

    def mock_is_databricks_uc_table():
        return True

    with mock.patch(
        "mlflow.data.delta_dataset_source.DeltaDatasetSource._resolve_table_name",
        side_effect=mock_resolve_table_name,
    ):
        with mock.patch(
            "mlflow.data.delta_dataset_source.DeltaDatasetSource._lookup_table_id",
            side_effect=mock_lookup_table_id,
        ):
            with mock.patch(
                "mlflow.data.delta_dataset_source.DeltaDatasetSource._is_databricks_uc_table",
                side_effect=mock_is_databricks_uc_table,
            ):
                df = pd.DataFrame([[1, 2, 3], [1, 2, 3]], columns=["a", "b", "c"])
                df_spark = spark_session.createDataFrame(df)
                df_spark.write.format("delta").mode("overwrite").saveAsTable(
                    "default.temp_delta_versioned_with_id_throws", path=tmp_path
                )

                df2 = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "c"])
                df2_spark = spark_session.createDataFrame(df2)
                df2_spark.write.format("delta").mode("overwrite").saveAsTable(
                    "default.temp_delta_versioned_with_id_throws", path=tmp_path
                )

                delta_datasource = DeltaDatasetSource(
                    delta_table_name="temp_delta_versioned_with_id_throws", delta_table_version=1
                )
                loaded_df_spark = delta_datasource.load()
                assert loaded_df_spark.count() == df2_spark.count()
                assert delta_datasource.to_json() == json.dumps(
                    {
                        "delta_table_name": "default.temp_delta_versioned_with_id_throws",
                        "delta_table_version": 1,
                        "is_databricks_uc_table": True,
                    }
                )
