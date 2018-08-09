"""
MLflow integration of the Spark MLlib serialization tool.

This module provides utilities for saving and loading models 
using the native Spark MLlib (SparkML) persistence mechanism.
"""

from __future__ import absolute_import


import os


import pyspark
from pyspark import SparkContext
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.base import Transformer


from mlflow.utils.exception import SaveModelException 

FLAVOR_NAME = "sparkml"

# Default temporary directory on DFS. Used to write / read from Spark ML models.
DFS_TMP = "/tmp/mlflow"


def add_to_model(mlflow_model, path, spark_model, dfs_tmpdir):
    """
    :param mlflow_model: MLFlow model config to which this flavor is being added.
    :param path: Path of the MLFlow model to which this flavor is being added.
    :param spark_Model: Spark PipelineModel to be saved. This model must be MLeap-compatible and
                  cannot contain any custom transformers.
    """
    if not isinstance(spark_model, Transformer):
        raise SaveModelException("Unexpected type {}." 
                                 " Spark MLLib serialization only works with Spark Transformers".
                format(str(type(spark_model))))
    if not isinstance(spark_model, PipelineModel):
        raise SaveModelException("Not a PipelineModel." 
                                 " Spark MLLib can currently only save PipelineModels.")

    spark_path_full = os.path.join(path, "sparkml")
    sparkml_datapath_sub = os.path.join("sparkml", "model")
    sparkml_datapath_full = os.path.join(path, sparkml_datapath_sub)
    if os.path.exists(spark_path_full):
        raise SaveModelException("SparkML model data path already exists at: {path}".format(
            path=spark_path_full))
    os.makedirs(spark_path_full)

    tmp_path = _tmp_path(dfs_tmpdir)
    spark_model.save(tmp_path)
    _HadoopFileSystem.copy_to_local_file(tmp_path, sparkml_datapath_full, removeSrc=True) 

    mlflow_model.add_flavor(FLAVOR_NAME, 
                            pyspark_version=pyspark.version.__version__, 
                            model_data=sparkml_datapath_sub)


def load_model(model_path, flavor_conf, dfs_tmpdir):
    """
    :param model_path: Path to an MLFlow model that contains the SparkML flavor.
    :param flavor_conf: The SparkML flavor configuration associated with the MLFlow model.
    :param dfs_tmpdir: Temporary directory path on Distributed (Hadoop) File System (DFS) or local
        filesystem if running in local mode. The model will be read from this path and then 
        copied into the model's artifact directory. This is necessary because Spark ML models
        read from / write to / DFS if running on a cluster. All temporary files created on the
        DFS will be removed if this operation completes successfully.
    :return: SparkML model.
    :rtype: pyspark.ml.pipeline.PipelineModel
    """
    model_data_path = os.path.join(model_path, flavor_conf['model_data'])
    tmp_path = _tmp_path(dfs_tmpdir)
    try:
        # Spark ML expects the model to be stored on DFS
        # Copy the model to a temp DFS location first.
        _HadoopFileSystem.copy_from_local_file(model_data_path, tmp_path, removeSrc=False)
        pipeline_model = PipelineModel.load(tmp_path)
        return pipeline_model
    finally:
        _HadoopFileSystem.delete(tmp_path)


def load_pyfunc(path):
    """
    Load the model as PyFunc.
    :param path: Local path
    :return: The model as PyFunc.
    """
    spark = pyspark.sql.SparkSession.builder.config("spark.python.worker.reuse", True) \
        .master("local[1]").getOrCreate()
    # We do not need any DFS here as pyfunc should create its own SparkContext with no executors
    return _PyFuncModelWrapper(spark, PipelineModel.load("file:" + os.path.abspath(path)))


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


