import mlflow


def test_enabling_autologging_does_not_throw_when_spark_hasnt_been_started():
    mlflow.spark.autolog(disable=True)
