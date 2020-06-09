from pyspark.sql import SparkSession


def _get_active_spark_session():
    try:
        return SparkSession.builder.getActiveSession()
    except Exception:  # pylint: disable=broad-except
        return SparkSession._instantiatedSession
