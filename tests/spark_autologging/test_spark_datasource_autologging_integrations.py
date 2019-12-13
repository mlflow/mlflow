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

from mlflow._spark_autologging import _SPARK_TABLE_INFO_TAG_NAME
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


def test_spark_autologging_with_keras_autologging(
        spark_session, format, file_path, tracking_uri_mock):  # pylint: disable=unused-argument

    mlflow.spark.autolog()
    mlflow.keras.autolog()
    df = spark_session.read.format(format).option("header", "true"). \
        option("inferSchema", "true").load(file_path).select("number1", "number2")

    # Start here
    pandas_df = df.toPandas()
    import datetime
    print("Read DF in Python at %s" % datetime.datetime.now())
    # import time
    # time.sleep(1)
    x = pandas_df.values
    y = np.array([4, 5, 6])
    keras_model = Sequential()
    keras_model.add(Dense(1))
    keras_model.compile(loss='mean_squared_error', optimizer='SGD')
    # Start here
    keras_model.fit(x, y, epochs=1)
    time.sleep(1)
    all_runs = mlflow.search_runs()
    assert len(all_runs) == 1
    run_id = mlflow.active_run().info.run_id
    run = mlflow.get_run(run_id)
    print(run.data.metrics, run.data.params, run.data.tags)
    assert 'epochs' in run.data.params
    assert run.data.params['epochs'] == '1'
    assert 'callbacks' not in run.data.params
    assert 'validation_data' not in run.data.params
    assert _SPARK_TABLE_INFO_TAG_NAME in run.data.tags
    table_info_tag = run.data.tags[_SPARK_TABLE_INFO_TAG_NAME]
    assert table_info_tag == _get_expected_table_info_row(file_path, format)


def test_api_usage0(spark_session, tracking_uri_mock, format, file_path):
    mlflow.spark.autolog()
    mlflow.keras.autolog()

    # Test constructing DF, collecting it within a run, using same DF in subsequent fit() call
    df = spark_session.read.format(format).option("header", "true"). \
        option("inferSchema", "true").load(file_path).select("number1", "number2")

    # Start here
    with mlflow.start_run():
        pandas_df = df.toPandas()
        x = pandas_df.values
        y = np.array([4, 5, 6])
        keras_model = Sequential()
        keras_model.add(Dense(1))
        keras_model.compile(loss='mean_squared_error', optimizer='SGD')
        keras_model.fit(x, y, epochs=1)

    # Run is ended, so no active run, so Spark DF may not get logged
    pandas_df2 = df.filter("number1 > 0").toPandas()
    keras_model.fit(x=pandas_df2.values, y=y, epochs=1)


def test_api_usage1(spark_session, tracking_uri_mock, format, file_path):
    mlflow.spark.autolog()
    mlflow.keras.autolog()
    # Test constructing DF, collecting it within a run, using same DF in subsequent fit() call
    df = spark_session.read.format(format).option("header", "true"). \
        option("inferSchema", "true").load(file_path).select("number1", "number2")
    x = df.toPandas().values
    y = np.array([4, 5, 6])

    keras_model = Sequential()
    keras_model.add(Dense(1))
    keras_model.compile(loss='mean_squared_error', optimizer='SGD')
    keras_model.fit(x=x, y=y, epochs=1)

    # Compute second DF - does select() call DataFrame constructor? Basically depending on the
    # method implementation, we may or may not try to start a run again...
    # Also, there are other types other than DataFrame, for example there's GroupedData when
    # you do df.groupBy(), should we patch the constructor for that too?
    pandas_df2 = df.select("betterCols").toPandas()
    keras_model.fit(pandas_df2)


def test_api_usage2():
    mlflow.spark.autolog()
    mlflow.keras.autolog()
    # Test constructing DF, starting run afterwards
    df = ...
    pandas_df = df.toPandas()
    with mlflow.start_run():
        keras_model.fit(pandas_df)
    mlflow.end_run()

