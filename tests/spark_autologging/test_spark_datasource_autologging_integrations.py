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
import tensorflow

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
    for format in ["csv"]:#, "parquet", "json"]:
        format_to_file_path[format] = os.path.join(tempdir, "test-data-%s" % format)

    for format, file_path in format_to_file_path.items():
        df.write.option("header", "true").format(format).save(file_path)
    yield format_to_file_path
    shutil.rmtree(tempdir)

def _get_expected_table_info_row(path, format, version=None):
    expected_path = "file:%s" % path
    if version is None:
        return "path={path},format={format}".format(path=expected_path, format=format)
    return "path={path},version={version},format={format}".format(
        path=expected_path, version=version, format=format)


def test_spark_autologging_with_keras_autologging(spark_session, format_to_file_path, tracking_uri_mock):  # pylint: disable=unused-argument
    mlflow.spark.autolog()
    mlflow.keras.autolog()
    format, path = list(format_to_file_path.items())[0]
    df = spark_session.read.format(format).option("header", "true"). \
        option("inferSchema", "true").load(path).select("number1", "number2")
    pandas_df = df.toPandas()
    # import time
    # time.sleep(1)
    x = pandas_df.values
    y = np.array([4, 5, 6])
    keras_model = Sequential()
    keras_model.add(Dense(1))
    keras_model.compile(loss='mean_squared_error', optimizer='SGD')
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
    assert table_info_tag == _get_expected_table_info_row(path, format)

