import os

import pytest
from pyspark.sql import SparkSession

import mlflow
import mlflow.spark
from mlflow._spark_autologging import _get_current_listener
from tests.projects.utils import tracking_uri_mock  # pylint: disable=unused-import


def _get_mlflow_spark_jar_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pardir = os.path.pardir
    jar_dir = os.path.join(current_dir, pardir, pardir, "mlflow", "java", "spark", "target")
    jar_filenames = [fname for fname in os.listdir(jar_dir) if ".jar" in fname
                     and "sources" not in fname and "javadoc" not in fname]
    res = os.path.abspath(os.path.join(jar_dir, jar_filenames[0]))
    return res


def _get_or_create_spark_session():
    jar_path = _get_mlflow_spark_jar_path()
    return SparkSession.builder \
        .config("spark.jars", jar_path) \
        .master("local[*]") \
        .getOrCreate()


@pytest.fixture()
def spark_session():
    session = _get_or_create_spark_session()
    yield session
    session.stop()


@pytest.mark.large
def test_enabling_autologging_does_not_throw_when_spark_hasnt_been_started(
        spark_session, tracking_uri_mock):
    # pylint: disable=unused-argument
    spark_session.stop()
    mlflow.spark.autolog()


@pytest.mark.large
def test_enabling_autologging_successful_across_sessions(spark_session, tracking_uri_mock):
    # pylint: disable=unused-argument
    mlflow.spark.autolog()
    orig_listener = _get_current_listener()
    spark_session.stop()
    _get_or_create_spark_session()
    mlflow.spark.autolog()
    new_listener = _get_current_listener()
    assert orig_listener.replId() != new_listener.replId()
