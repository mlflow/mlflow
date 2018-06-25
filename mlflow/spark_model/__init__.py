import pandas

from mlflow import tracking


def load_udf(spark, path, run_id=None, result_type='double'):
    """Returns a Spark UDF that can be used to invoke the python-function formatted model.

    Note that parameters passed to the UDF will be forwarded to the model as a DataFrame
    where the names are simply ordinals (0, 1, ...).

    Example:
        predict = mlflow.pyfunc.spark_model(spark, "/my/local/model")
        df.withColumn("prediction", predict("name", "age")).show()

    Args:
        spark (SparkSession): a SparkSession object
        path (str): A path containing a pyfunc model.
        result_type (str): Spark UDF type returned by the model's prediction method. Default double
        :param run_id:
    """

    if run_id:
        path = tracking._get_model_log_dir(path, run_id)

    # Scope Spark import to this method so users don't need pyspark to use non-Spark-related
    # functionality.
    from mlflow.spark_model.spark_model_cache import SparkModelCache
    from pyspark.sql.functions import pandas_udf

    archive_path = SparkModelCache.add_local_model(spark, path)

    def predict(*args):
        model = SparkModelCache.get_or_load(archive_path)
        schema = {str(i): arg for i, arg in enumerate(args)}
        pdf = pandas.DataFrame(schema)
        result = model.predict(pdf)
        return pandas.Series(result)

    return pandas_udf(predict, result_type)
