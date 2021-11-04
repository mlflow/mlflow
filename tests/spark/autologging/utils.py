import os
import pytest
import shutil
import tempfile

from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, IntegerType, StringType, StructField

from mlflow._spark_autologging import _SPARK_TABLE_INFO_TAG_NAME


def _get_mlflow_spark_jar_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pardir = os.path.pardir
    jar_dir = os.path.join(current_dir, pardir, pardir, "mlflow", "java", "spark", "target")
    jar_filenames = [
        fname
        for fname in os.listdir(jar_dir)
        if ".jar" in fname and "sources" not in fname and "javadoc" not in fname
    ]
    res = os.path.abspath(os.path.join(jar_dir, jar_filenames[0]))
    return res


def _get_expected_table_info_row(path, data_format, version=None):
    expected_path = "file:%s" % path
    if version is None:
        return "path={path},format={data_format}".format(
            path=expected_path, data_format=data_format
        )
    return "path={path},version={version},format={data_format}".format(
        path=expected_path, version=version, data_format=data_format
    )


def _assert_spark_data_logged(run, path, data_format, version=None):
    assert _SPARK_TABLE_INFO_TAG_NAME in run.data.tags
    table_info_tag = run.data.tags[_SPARK_TABLE_INFO_TAG_NAME]
    assert table_info_tag == _get_expected_table_info_row(path, data_format, version)


def _assert_spark_data_not_logged(run):
    assert _SPARK_TABLE_INFO_TAG_NAME not in run.data.tags


def _get_or_create_spark_session(jars=None):
    jar_path = jars if jars is not None else _get_mlflow_spark_jar_path()
    return SparkSession.builder.config("spark.jars", jar_path).master("local[*]").getOrCreate()


@pytest.fixture(scope="module")
def spark_session():
    jar_path = _get_mlflow_spark_jar_path()
    session = SparkSession.builder.config("spark.jars", jar_path).master("local[*]").getOrCreate()
    yield session
    session.stop()


@pytest.fixture(scope="module")
def data_format(format_to_file_path):
    res, _ = sorted(list(format_to_file_path.items()))[0]
    return res


@pytest.fixture(scope="module")
def file_path(format_to_file_path):
    _, file_path = sorted(list(format_to_file_path.items()))[0]
    return file_path


@pytest.fixture(scope="module")
def format_to_file_path(spark_session):
    rows = [Row(8, 32, "bat"), Row(64, 40, "mouse"), Row(-27, 55, "horse")]
    schema = StructType(
        [
            StructField("number2", IntegerType()),
            StructField("number1", IntegerType()),
            StructField("word", StringType()),
        ]
    )
    rdd = spark_session.sparkContext.parallelize(rows)
    df = spark_session.createDataFrame(rdd, schema)
    res = {}
    tempdir = tempfile.mkdtemp()
    for data_format in ["csv", "parquet", "json"]:
        res[data_format] = os.path.join(tempdir, "test-data-%s" % data_format)

    for data_format, file_path in res.items():
        df.write.option("header", "true").format(data_format).save(file_path)
    yield res
    shutil.rmtree(tempdir)
