import mlflow
import pytest
from pkg_resources import resource_filename

from pyspark.sql import SparkSession
from tests.spark_autologging.utils import _get_mlflow_spark_jar_path


@pytest.fixture(scope="module")
def spark_session_with_custom_allowlist():
    jar_path = _get_mlflow_spark_jar_path()
    session = (
        SparkSession.builder.config("spark.jars", jar_path)
        .master("local[*]")
        .config(
            "spark.mlflow.pysparkml.autolog.logModelAllowlistFile",
            resource_filename(__name__, "custom_log_model_allowlist.txt"),
        )
        .getOrCreate()
    )
    yield session
    session.stop()


def test_custom_log_model_allowlist(spark_session_with_custom_allowlist):
    mlflow.pyspark.ml.autolog()
    assert mlflow.pyspark.ml._log_model_allowlist == {
        "pyspark.ml.regression.LinearRegressionModel",
        "pyspark.ml.classification.NaiveBayesModel",
    }
