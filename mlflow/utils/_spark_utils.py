def _get_active_spark_session():
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        # Return None if user doesn't have PySpark installed
        return None
    try:
        # getActiveSession() only exists in Spark 3.0 and above
        return SparkSession.getActiveSession()
    except Exception:
        # Fall back to this internal field for Spark 2.x and below.
        return SparkSession._instantiatedSession
