import os

from pyspark.sql import SparkSession

import mlflow
from mlflow._spark_autologging import _SPARK_TABLE_INFO_TAG_NAME


def _get_mlflow_spark_jar_path():
    jar_dir = os.path.join(os.path.dirname(mlflow.__file__), "java", "spark", "target")
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


def _disable_pin_thread():
    # PYSPARK_PIN_THREAD is set to true by default since Pyspark 3.2.0, which causes
    # issues with Py4J callbacks, so we ask users to set it to false.
    # We have to set this before creating the SparkSession.
    os.environ["PYSPARK_PIN_THREAD"] = "false"


def _get_or_create_spark_session(jars=None):
    _disable_pin_thread()

    jar_path = jars if jars is not None else _get_mlflow_spark_jar_path()
    return SparkSession.builder.config("spark.jars", jar_path).master("local[*]").getOrCreate()
