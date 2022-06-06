import time

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

import mlflow
import mlflow.spark

from tests.spark.autologging.utils import _assert_spark_data_logged
from tests.spark.autologging.utils import spark_session  # pylint: disable=unused-import
from tests.spark.autologging.utils import format_to_file_path  # pylint: disable=unused-import
from tests.spark.autologging.utils import file_path, data_format  # pylint: disable=unused-import


@pytest.fixture()
def http_tracking_uri_mock():
    mlflow.set_tracking_uri("http://some-cool-uri")
    yield
    mlflow.set_tracking_uri(None)


def _fit_sklearn(pandas_df):
    x = pandas_df.values
    y = np.array([4] * len(x))
    LinearRegression().fit(x, y)
    # Sleep to allow time for datasource read event to fire asynchronously from the JVM & for
    # the Python-side event handler to run & log a tag to the current active run.
    # This race condition (& the risk of dropping datasource read events for short-lived runs)
    # is known and documented in
    # https://mlflow.org/docs/latest/python_api/mlflow.spark.html#mlflow.spark.autolog
    time.sleep(5)


def _fit_sklearn_model_with_active_run(pandas_df):
    run_id = mlflow.active_run().info.run_id
    _fit_sklearn(pandas_df)
    return mlflow.get_run(run_id)


def _fit_sklearn_model_no_active_run(pandas_df):
    orig_runs = mlflow.search_runs()
    orig_run_ids = set(orig_runs["run_id"])
    _fit_sklearn(pandas_df)
    new_runs = mlflow.search_runs()
    new_run_ids = set(new_runs["run_id"])
    assert len(new_run_ids) == len(orig_run_ids) + 1
    run_id = (new_run_ids - orig_run_ids).pop()
    return mlflow.get_run(run_id)


def _fit_sklearn_model(pandas_df):
    active_run = mlflow.active_run()
    if active_run:
        return _fit_sklearn_model_with_active_run(pandas_df)
    else:
        return _fit_sklearn_model_no_active_run(pandas_df)


def test_spark_autologging_with_sklearn_autologging(spark_session, data_format, file_path):
    assert mlflow.active_run() is None
    mlflow.spark.autolog()
    mlflow.sklearn.autolog()
    df = (
        spark_session.read.format(data_format)
        .option("header", "true")
        .option("inferSchema", "true")
        .load(file_path)
        .select("number1", "number2")
    )
    pandas_df = df.toPandas()
    run = _fit_sklearn_model(pandas_df)
    _assert_spark_data_logged(run, file_path, data_format)
    assert mlflow.active_run() is None


def test_spark_sklearn_autologging_context_provider(spark_session, data_format, file_path):
    mlflow.spark.autolog()
    mlflow.sklearn.autolog()

    df = (
        spark_session.read.format(data_format)
        .option("header", "true")
        .option("inferSchema", "true")
        .load(file_path)
        .select("number1", "number2")
    )
    pandas_df = df.toPandas()

    # DF info should be logged to the first run (it should be added to our context provider after
    # the toPandas() call above & then logged here)
    with mlflow.start_run():
        run = _fit_sklearn_model(pandas_df)
    _assert_spark_data_logged(run, file_path, data_format)

    with mlflow.start_run():
        pandas_df2 = df.filter("number1 > 0").toPandas()
        run2 = _fit_sklearn_model(pandas_df2)
    assert run2.info.run_id != run.info.run_id
    _assert_spark_data_logged(run2, file_path, data_format)
    time.sleep(1)
    assert mlflow.active_run() is None


def test_spark_and_sklearn_autologging_all_runs_managed(spark_session, data_format, file_path):
    mlflow.spark.autolog()
    mlflow.sklearn.autolog()
    for _ in range(2):
        with mlflow.start_run():
            df = (
                spark_session.read.format(data_format)
                .option("header", "true")
                .option("inferSchema", "true")
                .load(file_path)
                .select("number1", "number2")
            )
            pandas_df = df.toPandas()
            run = _fit_sklearn_model(pandas_df)
        _assert_spark_data_logged(run, file_path, data_format)
    assert mlflow.active_run() is None
