def is_spark_connect_mode():
    try:
        from pyspark.sql.utils import is_remote
    except ImportError:
        return False
    return is_remote()


def get_spark_dataframe_type():
    if is_spark_connect_mode():
        from pyspark.sql.connect.dataframe import DataFrame as SparkDataFrame
    else:
        from pyspark.sql import DataFrame as SparkDataFrame

    return SparkDataFrame
