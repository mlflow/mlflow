import pytest

import mlflux.spark
from mlflux.exceptions import MlflowException

from tests.spark_autologging.utils import _get_or_create_spark_session


@pytest.mark.large
def test_enabling_autologging_throws_for_missing_jar():
    # pylint: disable=unused-argument
    spark_session = _get_or_create_spark_session(jars="")
    try:
        with pytest.raises(MlflowException) as exc:
            mlflux.spark.autolog()
        assert "ensure you have the mlflux-spark JAR attached" in exc.value.message
    finally:
        spark_session.stop()
