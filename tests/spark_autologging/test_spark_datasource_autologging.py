import mock
import os
import shutil
import tempfile
import time

import pytest
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, IntegerType, StringType, StructField

import mlflow
import mlflow.spark
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
    yield session
    session.stop()

@pytest.fixture()
def http_tracking_uri_mock():
    mlflow.set_tracking_uri("http://some-cool-uri")
    yield
    mlflow.set_tracking_uri(None)


@pytest.fixture(scope="module", autouse=True)
def format_to_file_path(spark_session):
    rows = [
        Row(8, "bat"),
        Row(64, "mouse"),
        Row(-27, "horse")
    ]
    schema = StructType([
        StructField("number", IntegerType()),
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


def test_autologging_of_datasources_with_different_formats(spark_session, tracking_uri_mock, format_to_file_path):
    mlflow.spark.autolog()
    for format, file_path in format_to_file_path.items():
        base_df = spark_session.read.format(format).option("header", "true").\
            option("inferSchema", "true").load(file_path)
        dfs = [
            base_df,
            base_df.filter("number > 0"),
            base_df.select("number"),
            base_df.limit(2),
            base_df.filter("number > 0").select("number").limit(2)]

        for df in dfs:
            with mlflow.start_run():
                run_id = mlflow.active_run().info.run_id
                df.collect()
            time.sleep(1)
            run = mlflow.get_run(run_id)
            assert _SPARK_TABLE_INFO_TAG_NAME in run.data.tags
            table_info_tag = run.data.tags[_SPARK_TABLE_INFO_TAG_NAME]
            assert table_info_tag == _get_expected_table_info_row(file_path, format)


def test_autologging_does_not_throw_on_api_failures(
        spark_session, format_to_file_path, http_tracking_uri_mock): # pylint: disable=unused-import
    mlflow.spark.autolog()
    def failing_req_mock(*args, **kwargs):
        raise Exception("API request failed!")

    with mock.patch('mlflow.utils.rest_utils.http_request') as http_request_mock:
        mlflow.set_tracking_uri("http://some-cool-url")
        http_request_mock.side_effect = failing_req_mock
        format = list(format_to_file_path.keys())[0]
        file_path = format_to_file_path[format]
        df = spark_session.read.format(format).option("header", "true"). \
            option("inferSchema", "true").load(file_path)
        df.collect()
        df.filter("number > 0").collect()
        df.limit(2).collect()
        df.collect()
        time.sleep(1)


def test_autologging_dedups_multiple_reads_of_same_datasource(
        spark_session, format_to_file_path, tracking_uri_mock):
    mlflow.spark.autolog()
    format = list(format_to_file_path.keys())[0]
    file_path = format_to_file_path[format]
    df = spark_session.read.format(format).option("header", "true"). \
        option("inferSchema", "true").load(file_path)
    with mlflow.start_run():
        run_id = mlflow.active_run().info.run_id
        df.collect()
        df.filter("number > 0").collect()
        df.limit(2).collect()
        df.collect()
    time.sleep(1)
    run = mlflow.get_run(run_id)
    assert _SPARK_TABLE_INFO_TAG_NAME in run.data.tags
    table_info_tag = run.data.tags[_SPARK_TABLE_INFO_TAG_NAME]
    assert table_info_tag == _get_expected_table_info_row(path=file_path, format=format)


def test_autologging_multiple_reads_same_run(spark_session, tracking_uri_mock, format_to_file_path):
    mlflow.spark.autolog()
    with mlflow.start_run():
        for format, file_path in format_to_file_path.items():
            run_id = mlflow.active_run().info.run_id
            df = spark_session.read.format(format).option("header", "true"). \
                option("inferSchema", "true").load(file_path)
            df.collect()
            time.sleep(1)
        run = mlflow.get_run(run_id)
        assert _SPARK_TABLE_INFO_TAG_NAME in run.data.tags
        table_info_tag = run.data.tags[_SPARK_TABLE_INFO_TAG_NAME]
        assert table_info_tag == "\n".join([
            _get_expected_table_info_row(path, format)
            for format, path in format_to_file_path.items()
        ])


def test_autologging_starts_run_if_none_active(spark_session, format_to_file_path, tracking_uri_mock):
    try:
        mlflow.spark.autolog()
        format = list(format_to_file_path.keys())[0]
        file_path = format_to_file_path[format]
        df = spark_session.read.format(format).option("header", "true"). \
            option("inferSchema", "true").load(file_path)
        df.collect()
        time.sleep(1)
        print("@SID mlflow.active_run(): %s" % mlflow.active_run())
        run_id = mlflow.active_run().info.run_id
        run = mlflow.get_run(run_id)
        assert _SPARK_TABLE_INFO_TAG_NAME in run.data.tags
        table_info_tag = run.data.tags[_SPARK_TABLE_INFO_TAG_NAME]
        assert table_info_tag == _get_expected_table_info_row(path=file_path, format=format)
    finally:
        mlflow.end_run()


def test_enabling_autologging_does_not_throw_when_spark_hasnt_been_started(spark_session, tracking_uri_mock):
    spark_session.stop()
    mlflow.spark.autolog()