import mlflow
import pytest
from pkg_resources import resource_filename

from tests.spark_autologging.utils import spark_session  # pylint: disable=unused-import


@pytest.mark.parametrize(
    "spark_session",
    [
        {
            "spark.mlflow.pysparkml.autolog.logModelAllowlistFile": resource_filename(
                __name__, "custom_log_model_allowlist.txt"
            )
        }
    ],
    indirect=["spark_session"],
)
def test_custom_log_model_allowlist(spark_session):  # pylint: disable=unused-argument
    mlflow.pyspark.ml.autolog()
    assert mlflow.pyspark.ml._log_model_allowlist == {
        "pyspark.ml.regression.LinearRegressionModel",
        "pyspark.ml.classification.NaiveBayesModel",
    }
