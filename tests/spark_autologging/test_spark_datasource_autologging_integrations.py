import os
import shutil
import tempfile
import time

from keras.models import Sequential
from keras.layers import Layer, Dense

import numpy as np
import pytest
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, IntegerType, StringType, StructField

import mlflow
import mlflow.spark
import mlflow.keras
import mlflow.tensorflow

from mlflow.tracking.context.spark_autologging_context import _SPARK_TABLE_INFO_TAG_NAME
from tests.projects.utils import tracking_uri_mock  # pylint: disable=unused-import

def _get_mlflow_spark_jar_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pardir = os.path.pardir
    jar_dir = os.path.join(current_dir, pardir, pardir, "mlflow", "java", "spark", "target")
    jar_filenames = [fname for fname in os.listdir(jar_dir) if ".jar" in fname
                     and "sources" not in fname and "javadoc" not in fname]
    res = os.path.abspath(os.path.join(jar_dir, jar_filenames[0]))
    return res

@pytest.fixture(scope="module", autouse=True)
def spark_session():
    jar_path = _get_mlflow_spark_jar_path()
    session = SparkSession.builder \
        .config("spark.jars", jar_path)\
        .master("local[*]") \
        .getOrCreate()
    #.config("spark.jars", "/Users/sid.murching/code/mlflow/mlflow/java/client/target/mlflow-client-1.4.1-SNAPSHOT.jar") \
    yield session
    session.stop()

@pytest.fixture()
def http_tracking_uri_mock():
    mlflow.set_tracking_uri("http://some-cool-uri")
    yield
    mlflow.set_tracking_uri(None)


@pytest.fixture(scope="module")
def format_to_file_path(spark_session):
    rows = [
        Row(8, 32, "bat"),
        Row(64, 40, "mouse"),
        Row(-27, 55, "horse")
    ]
    schema = StructType([
        StructField("number2", IntegerType()),
        StructField("number1", IntegerType()),
        StructField("word", StringType())
    ])
    rdd = spark_session.sparkContext.parallelize(rows)
    df = spark_session.createDataFrame(rdd, schema)
    format_to_file_path = {}
    tempdir = tempfile.mkdtemp()
    for format in ["csv", "parquet", "json"]:
        format_to_file_path[format] = os.path.join(tempdir, "test-data-%s" % format)

    for format, file_path in format_to_file_path.items():
        df.write.option("header", "true").format(format).save(file_path)
    yield format_to_file_path
    shutil.rmtree(tempdir)

@pytest.fixture(scope="module")
def format(format_to_file_path):
    format, _ = sorted(list(format_to_file_path.items()))[0]
    return format


@pytest.fixture(scope="module")
def file_path(format_to_file_path):
    _, file_path = sorted(list(format_to_file_path.items()))[0]
    return file_path


def _get_expected_table_info_row(path, format, version=None):
    expected_path = "file:%s" % path
    if version is None:
        return "path={path},format={format}".format(path=expected_path, format=format)
    return "path={path},version={version},format={format}".format(
        path=expected_path, version=version, format=format)


def _assert_spark_data_logged(run, path, format, version=None):
    assert _SPARK_TABLE_INFO_TAG_NAME in run.data.tags
    table_info_tag = run.data.tags[_SPARK_TABLE_INFO_TAG_NAME]
    assert table_info_tag == _get_expected_table_info_row(path, format, version)

def _fit_keras(pandas_df, epochs):
    x = pandas_df.values
    y = np.array([4] * len(x))
    keras_model = Sequential()
    keras_model.add(Dense(1))
    keras_model.compile(loss='mean_squared_error', optimizer='SGD')
    keras_model.fit(x, y, epochs=epochs)
    time.sleep(1)

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


def test_spark_autologging_with_keras_autologging(
        spark_session, format, file_path, tracking_uri_mock):  # pylint: disable=unused-argument
    mlflow.spark.autolog()
    mlflow.keras.autolog()
    df = spark_session.read.format(format).option("header", "true"). \
        option("inferSchema", "true").load(file_path).select("number1", "number2")
    pandas_df = df.toPandas()
    run = _fit_keras_model(pandas_df, epochs=1)
    _assert_spark_data_logged(run, file_path, format)


def test_spark_keras_autologging_context_provider(spark_session, tracking_uri_mock, format, file_path):
    mlflow.spark.autolog()
    mlflow.keras.autolog()

    df = spark_session.read.format(format).option("header", "true"). \
        option("inferSchema", "true").load(file_path).select("number1", "number2")
    pandas_df = df.toPandas()

    # DF info should be logged to the first run (it should be added to our context provider after
    # the toPandas() call above & then logged here)
    with mlflow.start_run():
        run = _fit_keras_model(pandas_df, epochs=1)
    _assert_spark_data_logged(run, file_path, format)

    with mlflow.start_run():
        pandas_df2 = df.filter("number1 > 0").toPandas()
        run2 = _fit_keras_model(pandas_df2, epochs=1)
    assert run2.info.run_id != run.info.run_id
    _assert_spark_data_logged(run2, file_path, format)
    time.sleep(1)
    assert mlflow.active_run() is None


def test_spark_and_keras_autologging_no_active_run_mgmt(
        spark_session, tracking_uri_mock, format, file_path):
    mlflow.spark.autolog()
    mlflow.keras.autolog()
    df = spark_session.read.format(format).option("header", "true"). \
        option("inferSchema", "true").load(file_path).select("number1", "number2")
    pandas_df = df.toPandas()
    run = _fit_keras_model(pandas_df, epochs=1)
    _assert_spark_data_logged(run, file_path, format)
    assert mlflow.active_run() is None


def test_spark_and_keras_autologging_all_runs_managed(spark_session, tracking_uri_mock, format, file_path):
    mlflow.spark.autolog()
    mlflow.keras.autolog()
    for _ in range(2):
        with mlflow.start_run():
            df = spark_session.read.format(format).option("header", "true"). \
                option("inferSchema", "true").load(file_path).select("number1", "number2")
            pandas_df = df.toPandas()
            run = _fit_keras_model(pandas_df, epochs=1)
        _assert_spark_data_logged(run, file_path, format)
    assert mlflow.active_run() is None
