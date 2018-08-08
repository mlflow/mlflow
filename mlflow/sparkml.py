from __future__ import absolute_import

import os

import pyspark
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.base import Transformer

from mlflow.utils.exception import SaveModelException 

FLAVOR_NAME = "sparkml"


def add_to_model(mlflow_model, path, spark_model):
    # TODO(czumar): How does this work with the mlflow_model artifact path?

    """
    :param mlflow_model: MLFlow model config to which this flavor is being added.
    :param path: Path of the MLFlow model to which this flavor is being added.
    :param spark_Model: Spark PipelineModel to be saved. This model must be MLeap-compatible and
                  cannot contain any custom transformers.
    """
    if not isinstance(spark_model, Transformer):
        raise SaveModelException("Unexpected type {}. SparkML model works only with Transformers".
                format(str(type(spark_model))))
    if not isinstance(spark_model, PipelineModel):
        raise SaveModelException("Not a PipelineModel." 
                                 " SparkML can currently only save PipelineModels.")

    spark_path_full = os.path.join(path, "sparkml")
    sparkml_datapath_sub = os.path.join("sparkml", "model")
    sparkml_datapath_full = os.path.join(path, sparkml_datapath_sub)
    if os.path.exists(spark_path_full):
        raise SaveModelException("SparkML model data path already exists at: {path}".format(
            path=spark_path_full))
    os.makedirs(spark_path_full)
    spark_model.save(sparkml_datapath_full)
    mlflow_model.add_flavor(FLAVOR_NAME, 
                            pyspark_version=pyspark.version.__version__, 
                            model_data=sparkml_datapath_sub)


def load_model(model_path, flavor_conf):
    """
    :param model_path: Path to an MLFlow model that contains the SparkML flavor.
    :param flavor_conf: The SparkML flavor configuration associated with the MLFlow model.
    :return: SparkML model.
    :rtype: pyspark.ml.pipeline.PipelineModel
    """
    return PipelineModel.load(os.path.join(model_path, flavor_conf['model_data']))


def load_pyfunc(path):
    """
    Load the model as PyFunc.
    :param path: Local path
    :return: The model as PyFunc.
    """
    spark = pyspark.sql.SparkSession.builder.config("spark.python.worker.reuse", True) \
        .master("local[1]").getOrCreate()
    return _PyFuncModelWrapper(spark, PipelineModel.load(path))


class _PyFuncModelWrapper(object):
    """
    Wrapper around Spark MLlib PipelineModel providing interface for scoring pandas DataFrame.
    """
    def __init__(self, spark, spark_model):
        self.spark = spark
        self.spark_model = spark_model

    def predict(self, pandas_df):
        """
        Generate predictions given input data in Pandas DataFrame.

        :param pandas_df:
        :return: list with model predictions
        :rtype: list
        """
        spark_df = self.spark.createDataFrame(pandas_df)
        return [x.prediction for x in
                self.spark_model.transform(spark_df).select("prediction").collect()]
