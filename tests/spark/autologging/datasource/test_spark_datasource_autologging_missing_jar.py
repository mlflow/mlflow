import pytest

import mlflow.spark
from mlflow.exceptions import MlflowException

from tests.spark.autologging.utils import _get_or_create_spark_session


def test_enabling_autologging_throws_for_missing_jar():
    with _get_or_create_spark_session(jars=""):
        with pytest.raises(MlflowException, match="ensure you have the mlflow-spark JAR attached"):
            mlflow.spark.autolog()
