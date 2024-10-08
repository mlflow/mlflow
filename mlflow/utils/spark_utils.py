def is_spark_connect_mode():
    try:
        from pyspark.sql.utils import is_remote
    except ImportError:
        return False
    return is_remote()
