import mlflow


def test_enabling_autologging_does_not_throw_when_spark_hasnt_been_started():
    from pyspark.sql import SparkSession

    assert SparkSession.getActiveSession() is None

    mlflow.spark.autolog(disable=True)
