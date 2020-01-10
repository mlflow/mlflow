import time

from keras.layers import Dense
from keras.models import Sequential

import numpy as np
import pytest

import mlflow
import mlflow.spark
import mlflow.keras
import mlflow.tensorflow

from tests.projects.utils import tracking_uri_mock  # pylint: disable=unused-import
from tests.spark_autologging.utils import _assert_spark_data_logged
from tests.spark_autologging.utils import spark_session  # pylint: disable=unused-import
from tests.spark_autologging.utils import format_to_file_path  # pylint: disable=unused-import
from tests.spark_autologging.utils import file_path, data_format  # pylint: disable=unused-import


@pytest.fixture()
def http_tracking_uri_mock():
    mlflow.set_tracking_uri("http://some-cool-uri")
    yield
    mlflow.set_tracking_uri(None)


def _fit_keras(pandas_df, epochs):
    x = pandas_df.values
    y = np.array([4] * len(x))
    keras_model = Sequential()
    keras_model.add(Dense(1))
    keras_model.compile(loss='mean_squared_error', optimizer='SGD')
    keras_model.fit(x, y, epochs=epochs)
    time.sleep(2)


def _fit_keras_model_with_active_run(pandas_df, epochs):
    run_id = mlflow.active_run().info.run_id
    _fit_keras(pandas_df, epochs)
    run_id = run_id
    return mlflow.get_run(run_id)


def _fit_keras_model_no_active_run(pandas_df, epochs):
    orig_runs = mlflow.search_runs()
    orig_run_ids = set(orig_runs['run_id'])
    _fit_keras(pandas_df, epochs)
    new_runs = mlflow.search_runs()
    new_run_ids = set(new_runs['run_id'])
    assert len(new_run_ids) == len(orig_run_ids) + 1
    run_id = (new_run_ids - orig_run_ids).pop()
    return mlflow.get_run(run_id)


def _fit_keras_model(pandas_df, epochs):
    active_run = mlflow.active_run()
    if active_run:
        return _fit_keras_model_with_active_run(pandas_df, epochs)
    else:
        return _fit_keras_model_no_active_run(pandas_df, epochs)


@pytest.mark.large
def test_spark_autologging_with_keras_autologging(
        spark_session, data_format, file_path, tracking_uri_mock):
    # pylint: disable=unused-argument
    assert mlflow.active_run() is None
    mlflow.spark.autolog()
    mlflow.keras.autolog()
    df = spark_session.read.format(data_format).option("header", "true"). \
        option("inferSchema", "true").load(file_path).select("number1", "number2")
    pandas_df = df.toPandas()
    run = _fit_keras_model(pandas_df, epochs=1)
    _assert_spark_data_logged(run, file_path, data_format)
    assert mlflow.active_run() is None


@pytest.mark.large
def test_spark_keras_autologging_context_provider(
        spark_session, tracking_uri_mock, data_format, file_path):
    # pylint: disable=unused-argument
    mlflow.spark.autolog()
    mlflow.keras.autolog()

    df = spark_session.read.format(data_format).option("header", "true"). \
        option("inferSchema", "true").load(file_path).select("number1", "number2")
    pandas_df = df.toPandas()

    # DF info should be logged to the first run (it should be added to our context provider after
    # the toPandas() call above & then logged here)
    with mlflow.start_run():
        run = _fit_keras_model(pandas_df, epochs=1)
    _assert_spark_data_logged(run, file_path, data_format)

    with mlflow.start_run():
        pandas_df2 = df.filter("number1 > 0").toPandas()
        run2 = _fit_keras_model(pandas_df2, epochs=1)
    assert run2.info.run_id != run.info.run_id
    _assert_spark_data_logged(run2, file_path, data_format)
    time.sleep(1)
    assert mlflow.active_run() is None


@pytest.mark.large
def test_spark_and_keras_autologging_all_runs_managed(
        spark_session, tracking_uri_mock, data_format, file_path):
    # pylint: disable=unused-argument
    mlflow.spark.autolog()
    mlflow.keras.autolog()
    for _ in range(2):
        with mlflow.start_run():
            df = spark_session.read.format(data_format).option("header", "true"). \
                option("inferSchema", "true").load(file_path).select("number1", "number2")
            pandas_df = df.toPandas()
            run = _fit_keras_model(pandas_df, epochs=1)
        _assert_spark_data_logged(run, file_path, data_format)
    assert mlflow.active_run() is None
