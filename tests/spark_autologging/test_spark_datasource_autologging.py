import time

import pytest
from unittest import mock

import mlflow
import mlflow.spark
import tempfile
import os
import shutil

from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, IntegerType, StringType, StructField
from mlflow._spark_autologging import _SPARK_TABLE_INFO_TAG_NAME

from tests.tracking.test_rest_tracking import BACKEND_URIS
from tests.tracking.test_rest_tracking import tracking_server_uri  # pylint: disable=unused-import
from tests.tracking.test_rest_tracking import mlflow_client  # pylint: disable=unused-import
from tests.spark_autologging.utils import _assert_spark_data_logged
from tests.spark_autologging.utils import spark_session  # pylint: disable=unused-import
from tests.spark_autologging.utils import format_to_file_path  # pylint: disable=unused-import
from tests.spark_autologging.utils import data_format  # pylint: disable=unused-import
from tests.spark_autologging.utils import file_path  # pylint: disable=unused-import
from tests.spark_autologging.utils import _get_or_create_spark_session  # pylint: disable=unused-import
from tests.spark_autologging.utils import _get_mlflow_spark_jar_path


def pytest_generate_tests(metafunc):
    """
    Automatically parametrize each each fixture/test that depends on `backend_store_uri` with the
    list of backend store URIs.
    """
    if "backend_store_uri" in metafunc.fixturenames:
        metafunc.parametrize("backend_store_uri", BACKEND_URIS)


@pytest.fixture()
def http_tracking_uri_mock():
    mlflow.set_tracking_uri("http://some-cool-uri")
    yield
    mlflow.set_tracking_uri(None)


def _get_expected_table_info_row(path, data_format, version=None):
    expected_path = "file:%s" % path
    if version is None:
        return "path={path},format={format}".format(path=expected_path, format=data_format)
    return "path={path},version={version},format={format}".format(
        path=expected_path, version=version, format=data_format
    )


# Note that the following tests run one-after-the-other and operate on the SAME spark_session
#   (it is not reset between tests)

@pytest.mark.large
def test_autologging_of_datasources_with_different_formats(spark_session, format_to_file_path):
    mlflow.spark.autolog()
    for data_format, file_path in format_to_file_path.items():
        base_df = (
            spark_session.read.format(data_format)
            .option("header", "true")
            .option("inferSchema", "true")
            .load(file_path)
        )
        base_df.createOrReplaceTempView("temptable")
        table_df0 = spark_session.table("temptable")
        table_df1 = spark_session.sql("SELECT number1, number2 from temptable LIMIT 5")
        dfs = [
            base_df,
            table_df0,
            table_df1,
            base_df.filter("number1 > 0"),
            base_df.select("number1"),
            base_df.limit(2),
            base_df.filter("number1 > 0").select("number1").limit(2),
        ]

        for df in dfs:
            with mlflow.start_run():
                run_id = mlflow.active_run().info.run_id
                df.collect()
                time.sleep(1)
            run = mlflow.get_run(run_id)
            _assert_spark_data_logged(run=run, path=file_path, data_format=data_format)


@pytest.mark.large
def test_autologging_does_not_throw_on_api_failures(
    spark_session, format_to_file_path, mlflow_client
):
    # pylint: disable=unused-argument
    mlflow.spark.autolog()

    def failing_req_mock(*args, **kwargs):
        raise Exception("API request failed!")

    with mlflow.start_run():
        with mock.patch("mlflow.utils.rest_utils.http_request") as http_request_mock:
            http_request_mock.side_effect = failing_req_mock
            data_format = list(format_to_file_path.keys())[0]
            file_path = format_to_file_path[data_format]
            df = (
                spark_session.read.format(data_format)
                .option("header", "true")
                .option("inferSchema", "true")
                .load(file_path)
            )
            df.collect()
            df.filter("number1 > 0").collect()
            df.limit(2).collect()
            df.collect()
            time.sleep(1)


@pytest.mark.large
def test_autologging_dedups_multiple_reads_of_same_datasource(spark_session, format_to_file_path):
    mlflow.spark.autolog()
    data_format = list(format_to_file_path.keys())[0]
    file_path = format_to_file_path[data_format]
    df = (
        spark_session.read.format(data_format)
        .option("header", "true")
        .option("inferSchema", "true")
        .load(file_path)
    )
    with mlflow.start_run():
        run_id = mlflow.active_run().info.run_id
        df.collect()
        df.filter("number1 > 0").collect()
        df.limit(2).collect()
        df.collect()
        time.sleep(1)
    run = mlflow.get_run(run_id)
    _assert_spark_data_logged(run=run, path=file_path, data_format=data_format)
    # Test context provider flow
    df.filter("number1 > 0").collect()
    df.limit(2).collect()
    df.collect()
    with mlflow.start_run():
        run_id2 = mlflow.active_run().info.run_id
    time.sleep(1)
    run2 = mlflow.get_run(run_id2)
    _assert_spark_data_logged(run=run2, path=file_path, data_format=data_format)


@pytest.mark.large
def test_autologging_multiple_reads_same_run(spark_session, format_to_file_path):
    mlflow.spark.autolog()
    with mlflow.start_run():
        for data_format, file_path in format_to_file_path.items():
            run_id = mlflow.active_run().info.run_id
            df = spark_session.read.format(data_format).load(file_path)
            df.collect()
            time.sleep(1)
        run = mlflow.get_run(run_id)
        assert _SPARK_TABLE_INFO_TAG_NAME in run.data.tags
        table_info_tag = run.data.tags[_SPARK_TABLE_INFO_TAG_NAME]
        assert table_info_tag == "\n".join(
            [
                _get_expected_table_info_row(path, data_format)
                for data_format, path in format_to_file_path.items()
            ]
        )


@pytest.mark.large
def test_autologging_does_not_start_run(spark_session, format_to_file_path):
    try:
        mlflow.spark.autolog()
        data_format = list(format_to_file_path.keys())[0]
        file_path = format_to_file_path[data_format]
        df = (
            spark_session.read.format(data_format)
            .option("header", "true")
            .option("inferSchema", "true")
            .load(file_path)
        )
        df.collect()
        time.sleep(1)
        active_run = mlflow.active_run()
        assert active_run is None
        assert len(mlflow.search_runs()) == 0
    finally:
        mlflow.end_run()


@pytest.mark.large
@pytest.mark.usefixtures("mlflow_client")
def test_autologging_slow_api_requests(spark_session, format_to_file_path):
    import mlflow.utils.rest_utils

    orig = mlflow.utils.rest_utils.http_request

    def _slow_api_req_mock(*args, **kwargs):
        if kwargs.get("method") == "POST":
            print("Sleeping, %s, %s" % (args, kwargs))
            time.sleep(1)
        return orig(*args, **kwargs)

    mlflow.spark.autolog()
    with mlflow.start_run():
        # Mock slow API requests to log Spark datasource information
        with mock.patch("mlflow.utils.rest_utils.http_request") as http_request_mock:
            http_request_mock.side_effect = _slow_api_req_mock
            run_id = mlflow.active_run().info.run_id
            for data_format, file_path in format_to_file_path.items():
                df = (
                    spark_session.read.format(data_format)
                    .option("header", "true")
                    .option("inferSchema", "true")
                    .load(file_path)
                )
                df.collect()
        # Sleep a bit prior to ending the run to guarantee that the Python process can pick up on
        # datasource read events (simulate the common case of doing work, e.g. model training,
        # on the DataFrame after reading from it)
        time.sleep(1)

    # Python subscriber threads should pick up the active run at the time they're notified
    # & make API requests against that run, even if those requests are slow.
    time.sleep(5)
    run = mlflow.get_run(run_id)
    assert _SPARK_TABLE_INFO_TAG_NAME in run.data.tags
    table_info_tag = run.data.tags[_SPARK_TABLE_INFO_TAG_NAME]
    assert table_info_tag == "\n".join(
        [
            _get_expected_table_info_row(path, data_format)
            for data_format, path in format_to_file_path.items()
        ]
    )


@pytest.mark.large
def test_enabling_autologging_does_not_throw_when_spark_hasnt_been_started(spark_session):
    spark_session.stop()
    mlflow.spark.autolog()
