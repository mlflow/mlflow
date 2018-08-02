"""
MLflow integration for Spark MLlib models.

Spark MLlib models are saved and loaded using native Spark MLlib persistence.
The models can be exported as pyfunc for out-of Spark deployment or it can be loaded back as Spark
Transformer in order to score it in Spark. The pyfunc flavor instantiates SparkContext internally
and reads the input data as Spark DataFrame prior to scoring.
"""

from __future__ import absolute_import

import os
import shutil
import string
import random

import pyspark
from pyspark import SparkContext
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.base import Transformer

import mlflow
from mlflow import pyfunc
from mlflow.models import Model


FLAVOR_NAME = "spark"


def log_model(spark_model, artifact_path, conda_env=None, jars=None):
    """
    Log a Spark MLlib model as an MLflow artifact for the current run.

    :param spark_model: PipelineModel to be saved.
    :param artifact_path: Run relative artifact path.
    :param conda_env: Path to a Conda environment file. If provided, defines environment for the
    model. At minimum, it should specify python, pyspark, and mlflow with appropriate versions.
    :param jars: List of JARs needed by the model.
    """
    return Model.log(artifact_path=artifact_path, flavor=mlflow.spark, spark_model=spark_model,
                     jars=jars, conda_env=conda_env)


def _tmp_path(len):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(len))


class _DFS:
    _fs = None

    @classmethod
    def jvm(cls):
        return SparkContext._gateway.jvm

    @classmethod
    def fs(cls):
        if not cls._fs:
            sc = SparkContext.getOrCreate()
            cls._fs = cls.jvm().org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration())
        return cls._fs

    @classmethod
    def local_path(cls, path):
        return cls.jvm().org.apache.hadoop.fs.Path("file:" + os.path.abspath(path))

    @classmethod
    def remote_path(cls, path):
        return cls.fs().makeQualified(cls.jvm().org.apache.hadoop.fs.Path(path))


    @classmethod
    def copy_to_local(cls, src, dst):
        remote = cls.remote_path(src)
        cls.fs().copyToLocalFile(True, remote, cls.local_path(dst))

    @classmethod
    def copy_from_local(cls, src, dst):
       cls.fs().copyFromLocalFile(False, cls.local_path(src), cls.remote_path(dst))

    @classmethod
    def remove(cls, path):
        cls.fs().delete(cls.remote_path(path), True)


def save_model(spark_model, path, mlflow_model=Model(), input_df = None, conda_env=None, jars=None):
    """
    Save Spark MLlib PipelineModel at given local path.

    Uses Spark MLlib persistence mechanism.

    :param spark_model: Spark PipelineModel to be saved. Can save only PipelineModels.
    :param path: Local path where the model is to be saved.
    :param mlflow_model: MLflow model config this flavor is being added to.
    :param conda_env: Conda environment this model depends on.
    :param jars: List of JARs needed by the model.
    """
    if jars:
        raise Exception("jar dependencies are not implemented")
    if not isinstance(spark_model, Transformer):
        raise Exception("Unexpected type {}. SparkML model works only with Transformers".format(
            str(type(spark_model))))
    if not isinstance(spark_model, PipelineModel):
        raise Exception("Not a PipelineModel. SparkML can save only PipelineModels.")
    dfs_tmpdir = _tmp_path(32)
    spark_model.save(dfs_tmpdir)
    # Spark ML stores the model to DFS if funning on a cluster, copy it to local dir
    _DFS.copy_to_local(dfs_tmpdir, os.path.abspath(os.path.join(path, "model")))

    spark_model.save(os.path.join(path, "model"))
    pyspark_version = pyspark.version.__version__
    model_conda_env = None
    if conda_env:
        model_conda_env = os.path.basename(os.path.abspath(conda_env))
        shutil.copyfile(conda_env, os.path.join(path, model_conda_env))
    if jars:
        raise Exception("JAR dependencies are not yet implemented")
    mlflow_model.add_flavor(FLAVOR_NAME, pyspark_version=pyspark_version, model_data="model")
    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.spark", data="model",
                        env=model_conda_env)
    mlflow_model.save(os.path.join(path, "MLmodel"))


def load_model(path, run_id=None):
    """
    Load the Spark MLlib model from the path.

    :param run_id: Run ID. If provided, combined with ``path`` to identify the model.
    :param path: Local filesystem path or run-relative artifact path to the model.
    :return: SparkML model.
    :rtype: pyspark.ml.pipeline.PipelineModel
    """
    if run_id is not None:
        path = mlflow.tracking._get_model_log_dir(model_name=path, run_id=run_id)
    m = Model.load(os.path.join(path, 'MLmodel'))
    if FLAVOR_NAME not in m.flavors:
        raise Exception("Model does not have {} flavor".format(FLAVOR_NAME))
    conf = m.flavors[FLAVOR_NAME]
    model_path = os.path.join(path, conf['model_data'])
    dfs_tmp_path = _tmp_path(32)
    # Spark ML expects the model to be on the current DFS, copy the model dir to DFS first
    _DFS.copy_from_local(model_path, dfs_tmp_path)
    pipeline_model = PipelineModel.load(dfs_tmp_path)
    _DFS.remove(dfs_tmp_path)
    return pipeline_model


def load_pyfunc(path):
    """
    Load a Python Function model from a local file.

    :param path: Local path.
    :return: The model as PyFunc.
    """
    spark = pyspark.sql.SparkSession.builder.config("spark.python.worker.reuse", True) \
        .master("local[1]").getOrCreate()
    # We do not need any DFS here as pyfunc should create its own SparkContext with no executors
    return _PyFuncModelWrapper(spark, PipelineModel.load("file:" + os.path.abspath(path)))


class _PyFuncModelWrapper(object):
    """
    Wrapper around Spark MLlib PipelineModel providing interface for scoring pandas DataFrame.
    """
    def __init__(self, spark, spark_model):
        self.spark = spark
        self.spark_model = spark_model

    def predict(self, pandas_df):
        """
        Generate predictions given input data in pandas DataFrame.

        :param pandas_df: pandas Dataframe containing input data.
        :return: List with model predictions.
        :rtype: list
        """
        spark_df = self.spark.createDataFrame(pandas_df)
        return [x.prediction for x in
                self.spark_model.transform(spark_df).select("prediction").collect()]
