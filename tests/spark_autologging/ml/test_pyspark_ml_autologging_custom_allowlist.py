import os
import pytest
import mlflux

from pyspark.sql import SparkSession

pytestmark = pytest.mark.large


# Put this test in separate module because it require a spark context
# with a special conf and the conf is immutable in runtime.
def test_custom_log_model_allowlist(tmpdir):
    allowlist_file_path = os.path.join(tmpdir, "allowlist")
    with open(allowlist_file_path, "w") as f:
        f.write("pyspark.ml.regression.LinearRegressionModel\n")
        f.write("pyspark.ml.classification.NaiveBayesModel\n")

    spark_session = (
        SparkSession.builder.config(
            "spark.mlflux.pysparkml.autolog.logModelAllowlistFile", allowlist_file_path
        )
        .master("local[*]")
        .getOrCreate()
    )

    mlflux.pyspark.ml.autolog()
    assert mlflux.pyspark.ml._log_model_allowlist == {
        "pyspark.ml.regression.LinearRegressionModel",
        "pyspark.ml.classification.NaiveBayesModel",
    }

    spark_session.stop()
