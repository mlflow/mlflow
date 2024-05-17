import mlflow
import mlflow.spark


def test_enabling_autologging_does_not_throw_when_spark_hasnt_been_started(spark_session):
    spark_session.stop()
    mlflow.spark.autolog(disable=True)
