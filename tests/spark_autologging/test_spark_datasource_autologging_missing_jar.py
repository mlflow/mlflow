import pytest

import mlflow.spark
from mlflow.exceptions import MlflowException

from tests.spark_autologging.utils import _get_or_create_spark_session
from tests.projects.utils import tracking_uri_mock  # pylint: disable=unused-import


@pytest.mark.large
def test_enabling_autologging_throws_for_missing_jar(tracking_uri_mock):
    # pylint: disable=unused-argument
    spark_session = _get_or_create_spark_session(jars="")
    try:
        with pytest.raises(MlflowException) as exc:
            mlflow.spark.autolog()
        assert "Please ensure you have the mlflow-spark JAR attached" in exc.value.message
    finally:
        spark_session.stop()
