from pyspark.ml.param import Param as SparkMLParam
from pyspark.ml.util import Identifiable

from mlflow.entities import Param


def test_spark_integration():
    key = SparkMLParam(Identifiable(), "name", "doc")
    value = 123
    param = Param(key, value)
    assert param.key == "name"
    assert param.value == "123"
