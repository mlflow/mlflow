import os

import pyspark
from packaging.version import Version
from pyspark.sql import SparkSession

import mlflow
from mlflow.spark.autologging import _SPARK_TABLE_INFO_TAG_NAME


def _get_mlflow_spark_jar_path():
    spark_dir = "spark_2.13" if Version(pyspark.__version__).major >= 4 else "spark_2.12"
    jar_dir = os.path.join(os.path.dirname(mlflow.__file__), "java", spark_dir, "target")
    jar_filenames = [
        fname
        for fname in os.listdir(jar_dir)
        if ".jar" in fname and "sources" not in fname and "javadoc" not in fname
    ]
    return os.path.abspath(os.path.join(jar_dir, jar_filenames[0]))


def _get_expected_table_info_row(path, data_format, version=None):
    expected_path = f"file:{path}"
    if version is None:
        return f"path={expected_path},format={data_format}"
    return f"path={expected_path},version={version},format={data_format}"


def _assert_spark_data_logged(run, path, data_format, version=None):
    assert _SPARK_TABLE_INFO_TAG_NAME in run.data.tags
    table_info_tag = run.data.tags[_SPARK_TABLE_INFO_TAG_NAME]
    expected_tag = _get_expected_table_info_row(path, data_format, version)
    assert table_info_tag == expected_tag, f"Got: {table_info_tag} Expected: {expected_tag}"


def _assert_spark_data_not_logged(run):
    assert _SPARK_TABLE_INFO_TAG_NAME not in run.data.tags


def _get_or_create_spark_session(jars=None):
    jar_path = jars if jars is not None else _get_mlflow_spark_jar_path()
    return SparkSession.builder.config("spark.jars", jar_path).master("local[*]").getOrCreate()
