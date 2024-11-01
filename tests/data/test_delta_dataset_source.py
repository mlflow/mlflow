import json
from unittest import mock

import pandas as pd
import pytest

from mlflow.data.dataset_source_registry import get_dataset_source_from_json
from mlflow.data.delta_dataset_source import DeltaDatasetSource
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_managed_catalog_messages_pb2 import GetTable, GetTableResponse
from mlflow.utils.proto_json_utils import message_to_json


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
    assert delta_datasource.to_dict()["path"] == path

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
    assert delta_datasource.to_dict()["delta_table_name"] == "temp_delta"

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
    config = delta_datasource.to_dict()
    assert config["delta_table_name"] == "temp_delta_versioned"
    assert config["delta_table_version"] == 1

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
    def mock_resolve_table_name(table_name, spark):
        if table_name == "temp_delta_versioned_with_id":
            return "default.temp_delta_versioned_with_id"
        return table_name

    def mock_lookup_table_id(table_name):
        if table_name == "default.temp_delta_versioned_with_id":
            return "uc_table_id_1"
        return None

    with (
        mock.patch(
            "mlflow.data.delta_dataset_source.get_full_name_from_sc",
            side_effect=mock_resolve_table_name,
        ),
        mock.patch(
            "mlflow.data.delta_dataset_source.DeltaDatasetSource._lookup_table_id",
            side_effect=mock_lookup_table_id,
        ),
        mock.patch(
            "mlflow.data.delta_dataset_source._get_active_spark_session",
            return_value=None,
        ),
        mock.patch(
            "mlflow.data.delta_dataset_source.DeltaDatasetSource._is_databricks_uc_table",
            return_value=True,
        ),
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


def _args(endpoint, json_body):
    return {
        "host_creds": None,
        "endpoint": f"/api/2.0/unity-catalog/tables/{endpoint}",
        "method": "GET",
        "json_body": json_body,
        "response_proto": GetTableResponse,
    }


@pytest.mark.parametrize(
    ("call_endpoint_response", "expected_lookup_response", "test_table_name"),
    [
        (None, None, "delta_table_1"),
        (Exception("Exception from call_endpoint"), None, "delta_table_2"),
        (GetTableResponse(table_id="uc_table_id_1"), "uc_table_id_1", "delta_table_3"),
    ],
)
def test_lookup_table_id(
    call_endpoint_response, expected_lookup_response, test_table_name, tmp_path
):
    def mock_resolve_table_name(table_name, spark):
        if table_name == test_table_name:
            return f"default.{test_table_name}"
        return table_name

    def mock_call_endpoint(host_creds, endpoint, method, json_body, response_proto):
        if isinstance(call_endpoint_response, Exception):
            raise call_endpoint_response
        return call_endpoint_response

    with (
        mock.patch(
            "mlflow.data.delta_dataset_source.get_full_name_from_sc",
            side_effect=mock_resolve_table_name,
        ),
        mock.patch(
            "mlflow.data.delta_dataset_source._get_active_spark_session",
            return_value=None,
        ),
        mock.patch(
            "mlflow.data.delta_dataset_source.get_databricks_host_creds",
            return_value=None,
        ),
        mock.patch(
            "mlflow.data.delta_dataset_source.DeltaDatasetSource._is_databricks_uc_table",
            return_value=True,
        ),
        mock.patch(
            "mlflow.data.delta_dataset_source.call_endpoint",
            side_effect=mock_call_endpoint,
        ) as mock_endpoint,
    ):
        delta_datasource = DeltaDatasetSource(
            delta_table_name=test_table_name, delta_table_version=1
        )
        assert delta_datasource._lookup_table_id(test_table_name) == expected_lookup_response
        req_body = message_to_json(GetTable(full_name_arg=test_table_name))
        call_args = _args(test_table_name, req_body)
        mock_endpoint.assert_any_call(**call_args)


@pytest.mark.parametrize(
    ("table_name", "expected_result"),
    [
        ("default.test", True),
        ("hive_metastore.test", False),
        ("spark_catalog.test", False),
        ("samples.test", False),
    ],
)
def test_is_databricks_uc_table(table_name, expected_result):
    with (
        mock.patch(
            "mlflow.data.delta_dataset_source.get_full_name_from_sc",
            return_value=table_name,
        ),
        mock.patch(
            "mlflow.data.delta_dataset_source._get_active_spark_session",
            return_value=None,
        ),
    ):
        delta_datasource = DeltaDatasetSource(delta_table_name=table_name, delta_table_version=1)
        assert delta_datasource._is_databricks_uc_table() == expected_result
