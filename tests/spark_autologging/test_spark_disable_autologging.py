import time

import pytest

import mlflow
import mlflow.spark

from tests.tracking.test_rest_tracking import BACKEND_URIS
from tests.tracking.test_rest_tracking import tracking_server_uri  # pylint: disable=unused-import
from tests.tracking.test_rest_tracking import mlflow_client  # pylint: disable=unused-import
from tests.spark_autologging.utils import (
    _assert_spark_data_logged,
    _assert_spark_data_not_logged,
)
from tests.spark_autologging.utils import spark_session  # pylint: disable=unused-import
from tests.spark_autologging.utils import format_to_file_path  # pylint: disable=unused-import
from tests.spark_autologging.utils import data_format  # pylint: disable=unused-import
from tests.spark_autologging.utils import file_path  # pylint: disable=unused-import


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
def test_autologging_disabled_logging_datasource_with_different_formats(
    spark_session, format_to_file_path
):
    mlflow.spark.autolog(disable=True)
    for data_format, file_path in format_to_file_path.items():
        df = (
            spark_session.read.format(data_format)
            .option("header", "true")
            .option("inferSchema", "true")
            .load(file_path)
        )

        with mlflow.start_run():
            run_id = mlflow.active_run().info.run_id
            df.collect()
            time.sleep(1)
        run = mlflow.get_run(run_id)
        _assert_spark_data_not_logged(run=run)


@pytest.mark.large
def test_autologging_disabled_logging_with_or_without_active_run(
    spark_session, format_to_file_path
):
    mlflow.spark.autolog(disable=True)
    data_format = list(format_to_file_path.keys())[0]
    file_path = format_to_file_path[data_format]
    df = (
        spark_session.read.format(data_format)
        .option("header", "true")
        .option("inferSchema", "true")
        .load(file_path)
    )

    # Reading data source before starting a run
    df.filter("number1 > 0").collect()
    df.limit(2).collect()
    df.collect()

    # If there was any tag info collected it will be logged here
    with mlflow.start_run():
        run_id = mlflow.active_run().info.run_id
    time.sleep(1)

    # Confirm nothing was logged.
    run = mlflow.get_run(run_id)
    _assert_spark_data_not_logged(run=run)

    # Reading data source during an active run
    with mlflow.start_run():
        run_id = mlflow.active_run().info.run_id
        df.collect()
        time.sleep(1)
    run = mlflow.get_run(run_id)
    _assert_spark_data_not_logged(run=run)


@pytest.mark.large
def test_autologging_disabled_then_enabled(spark_session, format_to_file_path):
    mlflow.spark.autolog(disable=True)
    data_format = list(format_to_file_path.keys())[0]
    file_path = format_to_file_path[data_format]
    df = (
        spark_session.read.format(data_format)
        .option("header", "true")
        .option("inferSchema", "true")
        .load(file_path)
    )
    # Logging is disabled here.
    with mlflow.start_run():
        run_id = mlflow.active_run().info.run_id
        df.collect()
        time.sleep(1)
    run = mlflow.get_run(run_id)
    _assert_spark_data_not_logged(run=run)

    # Logging is enabled here.
    mlflow.spark.autolog(disable=False)
    with mlflow.start_run():
        run_id = mlflow.active_run().info.run_id
        df.filter("number1 > 0").collect()
        time.sleep(1)
    run = mlflow.get_run(run_id)
    _assert_spark_data_logged(run=run, path=file_path, data_format=data_format)


@pytest.mark.large
def test_enabling_autologging_does_not_throw_when_spark_hasnt_been_started(spark_session):
    spark_session.stop()
    mlflow.spark.autolog(disable=True)
