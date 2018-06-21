import pandas

from mlflow import tracking
from mlflow.models import Model

FLAVOR_UDF = "spark_udf"
LOADER_MODULE = "loader_module"


def add_spark_udf_to_model(model, loader_module="mlflow.pyfunc", result_type="double"):
    """ Add pyfunc spec to the model configuration.

    Defines pyfunc configuration schema. Caller can use this to create a valid pyfunc model flavor
    out of an existing directory structure. For example, other model flavors can use this to specify
    how to use their output as a pyfunc.

    NOTE: all paths are relative to the exported model root directory.

    :param loader_module:
    :param model: Existing servable
    :param data: to the model data
    :param code: path to the code dependencies
    :param env: conda environment
    :return: updated model configuration.
    """
    return model.add_flavor(FLAVOR_UDF, loader_module=loader_module, result_type=result_type)


def load_udf(spark, path, run_id=None):
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
    model = Model.load(path)
    if FLAVOR_UDF not in model.flavors:
        raise Exception("This model does not have {} flavor".format(FLAVOR_UDF))
    conf = model.flavors[FLAVOR_UDF]
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

    return pandas_udf(predict, conf['result_type'])

