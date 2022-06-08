import os
import mlflow
from pkg_resources import resource_filename

from pyspark.sql import SparkSession


# Put this test in separate module because it require a spark context
# with a special conf and the conf is immutable in runtime.
def test_custom_log_model_allowlist(tmpdir):
    allowlist_file_path = os.path.join(tmpdir, "allowlist")
    with open(allowlist_file_path, "w") as f:
        f.write("pyspark.ml.regression.LinearRegressionModel\n")
        f.write("pyspark.ml.classification.NaiveBayesModel\n")

    spark_session = (
        SparkSession.builder.config(
            "spark.mlflow.pysparkml.autolog.logModelAllowlistFile", allowlist_file_path
        )
        .master("local[*]")
        .getOrCreate()
    )

    mlflow.pyspark.ml.autolog()
    assert mlflow.pyspark.ml._log_model_allowlist == {
        "pyspark.ml.regression.LinearRegressionModel",
        "pyspark.ml.classification.NaiveBayesModel",
    }

    spark_session.stop()


def test_log_model_allowlist_from_url():

    allowlist_file_path = "https://raw.githubusercontent.com/mlflow/mlflow/master/mlflow/pyspark/ml/log_model_allowlist.txt"

    spark_session = (
        SparkSession.builder.config(
            "spark.mlflow.pysparkml.autolog.logModelAllowlistFile", allowlist_file_path
        )
        .master("local[*]")
        .getOrCreate()
    )

    mlflow.pyspark.ml.autolog()

    allowlist = set()
    builtin_allowlist_file = resource_filename("mlflow.pyspark.ml", "log_model_allowlist.txt")
    with open(builtin_allowlist_file) as f:
        for line in f:
            stripped = line.strip()
            is_blankline_or_comment = stripped == "" or stripped.startswith("#")
            if not is_blankline_or_comment:
                allowlist.add(stripped)

    assert mlflow.pyspark.ml._log_model_allowlist == allowlist

    spark_session.stop()
