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

import pyspark
from pyspark import SparkContext
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.base import Transformer

import mlflow
from mlflow import pyfunc
from mlflow.models import Model

FLAVOR_NAME = "spark"

# Default temporary directory on DFS. Used to write / read from Saprk ML models.
DFS_TMP = "/tmp/mlflow"


def log_model(spark_model, artifact_path, conda_env=None, jars=None, dfs_tmpdir=DFS_TMP):
    """
    Log a Spark MLlib model as an MLflow artifact for the current run.

    :param spark_model: PipelineModel to be saved.
    :param artifact_path: Run relative artifact path.
    :param conda_env: Path to a Conda environment file. If provided, defines environment for the
        model. At minimum, it should specify python, pyspark, and mlflow with appropriate versions.
    :param jars: List of JARs needed by the model.
    :param dfs_tmpdir: Temporary directory path on Distributed (Hadoop) File System (DFS) or local
        filesystem if running in local mode. The model will be writen in this destination and than
        copied into the model's artifact directory. This is necessary as Spark ML models
        read / write from / to DFS if running on a cluster. All temporary files created on the
        DFS will be removed if this operation completes successfully.
    """
    return Model.log(artifact_path=artifact_path, flavor=mlflow.spark, spark_model=spark_model,
                     jars=jars, conda_env=conda_env, dfs_tmpdir=dfs_tmpdir)


def _tmp_path(dfs_tmp):
    import uuid
    return os.path.join(dfs_tmp, str(uuid.uuid4()))


class _HadoopFileSystem:
    """
    Interface to org.apache.hadoop.fs.FileSystem.

    Spark ML models expect to read from and write to Hadoop FileSystem when running on a cluster.
    Sine MLflow works on local directories, we need this interface to copy teh files between
    the current DFS and local dir.
    """

    def __init__(self):
        raise Exception("This class should not be instantiated")
    _filesystem = None
    _conf = None

    @classmethod
    def _jvm(cls):
        return SparkContext._gateway.jvm

    @classmethod
    def _fs(cls):
        if not cls._filesystem:
            sc = SparkContext.getOrCreate()
            cls._conf = sc._jsc.hadoopConfiguration()
            cls._filesystem = cls._jvm().org.apache.hadoop.fs.FileSystem.get(cls._conf)
        return cls._filesystem

    @classmethod
    def _local_path(cls, path):
        return cls._jvm().org.apache.hadoop.fs.Path(os.path.abspath(path))

    @classmethod
    def _remote_path(cls, path):
        return cls._jvm().org.apache.hadoop.fs.Path(path)

    @classmethod
    def copy_to_local_file(cls, src, dst, removeSrc):
        cls._fs().copyToLocalFile(removeSrc, cls._remote_path(src), cls._local_path(dst))

    @classmethod
    def copy_from_local_file(cls, src, dst, removeSrc):
        cls._fs().copyFromLocalFile(removeSrc, cls._local_path(src), cls._remote_path(dst))

    @classmethod
    def delete(cls, path):
        cls._fs().delete(cls._remote_path(path), True)


def save_model(spark_model, path, mlflow_model=Model(), conda_env=None, jars=None,
               dfs_tmpdir=DFS_TMP):
    """
    Save Spark MLlib PipelineModel at given local path.

    Uses Spark MLlib persistence mechanism.

    :param spark_model: Spark PipelineModel to be saved. Can save only PipelineModels.
    :param path: Local path where the model is to be saved.
    :param mlflow_model: MLflow model config this flavor is being added to.
    :param conda_env: Conda environment this model depends on.
    :param jars: List of JARs needed by the model.
    :param dfs_tmpdir: Temporary directory path on Distributed (Hadoop) File System (DFS) or local
        filesystem if running in local mode. The model will be writen in this destination and than
        copied to requested local path. This is necessary as Spark ML models
        read / write from / to DFS if running on a cluster. All temporary files created on
        the DFS will be removed if this operation completes successfully.
    """
    if jars:
        raise Exception("jar dependencies are not implemented")
    if not isinstance(spark_model, Transformer):
        raise Exception("Unexpected type {}. SparkML model works only with Transformers".format(
            str(type(spark_model))))
    if not isinstance(spark_model, PipelineModel):
        raise Exception("Not a PipelineModel. SparkML can save only PipelineModels.")
    # Spark ML stores the model on DFS if running on a cluster
    # Save it to a DFS temp dir first and copy it to local path
    tmp_path = _tmp_path(dfs_tmpdir)
    spark_model.save(tmp_path)
    model_path = os.path.abspath(os.path.join(path, "model"))
    _HadoopFileSystem.copy_to_local_file(tmp_path, model_path, removeSrc=True)
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


def load_model(path, run_id=None, dfs_tmpdir=DFS_TMP):
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
    tmp_path = _tmp_path(dfs_tmpdir)
    try:
        # Spark ML expects the model to be stored on DFS
        # Copy the model to a temp DFS location first.
        _HadoopFileSystem.copy_from_local_file(model_path, tmp_path, removeSrc=False)
        pipeline_model = PipelineModel.load(tmp_path)
        return pipeline_model
    finally:
        _HadoopFileSystem.delete(tmp_path)


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
