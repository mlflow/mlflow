"""
MLflow integration for SparkML models.

SparkML models are saved and loaded using native SparkML persistence.
The models can be exported as pyfunc for out-of Spark deployment or it can be loaded back as Spark
Transformer in order to score it in Spark. The pyfunc flavor instantiates SparkContext internally
and reads the input data as Spark DataFrame prior to scoring.

Note: SparkML model can not be exported as spark_udf as those two formats are incompatible.
"""

from __future__ import absolute_import

import os

import mlflow
from mlflow import pyfunc
from mlflow.environment import CondaEnvironment
from mlflow.models import Model

import pyspark
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.base import Transformer

FLAVOR_NAME = "sparkml"


def log_model(spark_model, artifact_path, *jars, env):
    """
    Log model using supplied flavor.

    :param spark_model: Model to be saved.
    :param artifact_path: Run-relative artifact path.
    :param env: Conda environment, defaults to minimum viable environment.
    :param jars: List of jars needed by the model.
    """
    return Model.log(artifact_path=artifact_path, flavor=mlflow.sparkml, spark_model=spark_model,
                     jars=jars, env=env)


def save_model(spark_model, path, mlflow_model=Model(), env=CondaEnvironment(), jars=None):
    """
    Save SparkML model at given local path.

    Uses SparkML persistence mechanism.

    :param spark_model: Model to be saved.
    :param path: Local path where the model is to be save.
    :param mlflow_model: MLflow model config.
    :param env: Conda environment, defaults to minimum viable environment.
    :param jars: List of jars needed by the model.
    """
    if jars:
        raise Exception("jar dependencies are not implemented")
    if not isinstance(spark_model, Transformer):
        raise Exception("Unexpected type {}. SparkML model works only with Transformers".format(
            str(type(spark_model))))
    if not isinstance(spark_model, PipelineModel):
        raise Exception("Not a PipelineModel. SparkML can currently only save PipelineModels.")
    spark_model.save(os.path.join(path, "model"))
    version = pyspark.version.__version__
    env.add_pip_dependency("pyspark", version)
    if jars:
        raise Exception("jar dependencies are not yet implemented")
    mlflow_model.add_flavor('sparkml', pyspark_version=version, model_data="model")
    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.sparkml", data="model", env="env.yml")
    env.save(os.path.join(path, "env.yml"))
    mlflow_model.save(os.path.join(path, "MLmodel"))


def load_model(path, run_id=None):
    """
    Load the SparkML model from the given path.
    :param run_id: Run ID. If provided its is combined with path to identify the model.
    :param path: Local filesystem path or Run-relative artifact path to the model.
    :return: SparkML model.
    """
    if run_id is not None:
        path = mlflow.tracking._get_model_log_dir(model_name=path, run_id=run_id)
    m = Model.load(path)
    if FLAVOR_NAME not in m.flavors:
        raise Exception("Model does not have {} flavor".format(FLAVOR_NAME))
    conf = m.flavors[FLAVOR_NAME]
    return PipelineModel.load(conf['model_data'])


def load_pyfunc(path):
    """
    Load the model as PyFunc.
    :param path: Local path
    :return: The model as PyFunc.
    """
    spark = pyspark.sql.SparkSession.builder.config(key="spark.python.worker.reuse", value=True) \
        .master("local[1]").getOrCreate()
    return SparkMLModel(spark, PipelineModel.load(path))


class SparkMLModel(object):
    """
    Wrapper around SparkML PipelineModel providing interface for scoring pandas DataFrame.
    """

    def __init__(self, spark, transformer):
        self.spark = spark
        self.transformer = transformer

    def predict(self, pandas_df):
        """
        Generate predictions given input data in Pandas DataFrame.

        :param pandas_df:
        :return: list with model predictions
        """
        spark_df = self.spark.createDataFrame(pandas_df)
        return [x.prediction for x in
                self.transformer.transform(spark_df).select("prediction").collect()]
