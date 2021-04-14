import os
import mlflow
import pytest
from pkg_resources import resource_filename

from pyspark.sql import SparkSession
from tests.spark_autologging.utils import _get_mlflow_spark_jar_path


# Put this test in separate module because it require a spark context
# with a special conf and the conf is immutable in runtime.
def test_custom_log_model_allowlist(tmpdir):  # pylint: disable=unused-argument
    allowlist_file_path = os.path.join(tmpdir, "allowlist")
    with open(allowlist_file_path, "w") as f:
        f.write(
            "pyspark.ml.regression.LinearRegressionModel\npyspark.ml.classification.NaiveBayesModel\n"
        )

    jar_path = _get_mlflow_spark_jar_path()
    spark_session = (
        SparkSession.builder.config("spark.jars", jar_path)
        .config("spark.mlflow.pysparkml.autolog.logModelAllowlistFile", allowlist_file_path)
        .master("local[*]")
        .getOrCreate()
    )

    mlflow.pyspark.ml.autolog()
    assert mlflow.pyspark.ml._log_model_allowlist == {
        "pyspark.ml.regression.LinearRegressionModel",
        "pyspark.ml.classification.NaiveBayesModel",
    }

    spark_session.stop()
