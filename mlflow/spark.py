"""
The ``mlflow.spark`` module provides an API for logging and loading Spark MLlib models. This module
exports Spark MLlib models with the following flavors:

Spark MLlib (native) format
    Allows models to be loaded as Spark Transformers for scoring in a Spark session.
    Models with this flavor can be loaded as PySpark PipelineModel objects in Python.
    This is the main flavor and is always produced.
:py:mod:`mlflow.pyfunc`
    Supports deployment outside of Spark by instantiating a SparkContext and reading
    input data as a Spark DataFrame prior to scoring. Also supports deployment in Spark
    as a Spark UDF. Models with this flavor can be loaded as Python functions
    for performing inference. This flavor is always produced.
:py:mod:`mlflow.mleap`
    Enables high-performance deployment outside of Spark by leveraging MLeap's
    custom dataframe and pipeline representations. Models with this flavor *cannot* be loaded
    back as Python objects. Rather, they must be deserialized in Java using the
    ``mlflow/java`` package. This flavor is produced only if you specify
    MLeap-compatible arguments.
"""

from __future__ import absolute_import

import os
import shutil

import pyspark
from pyspark import SparkContext
from pyspark.ml.pipeline import PipelineModel

import mlflow
from mlflow import pyfunc, mleap
from mlflow.models import Model
from mlflow.utils.logging_utils import eprint

FLAVOR_NAME = "spark"

# Default temporary directory on DFS. Used to write / read from Spark ML models.
DFS_TMP = "/tmp/mlflow"


def log_model(spark_model, artifact_path, conda_env=None, jars=None, dfs_tmpdir=None,
              sample_input=None):
    """
    Log a Spark MLlib model as an MLflow artifact for the current run. This uses the
    MLlib persistence format, and the logged model will have the Spark flavor.

    :param spark_model: PipelineModel to be saved.
    :param artifact_path: Run relative artifact path.
    :param conda_env: Path to a Conda environment file. If provided, defines environment for the
                      model. At minimum, it should specify python, pyspark, and mlflow with
                      appropriate versions.
    :param jars: List of JARs needed by the model.
    :param dfs_tmpdir: Temporary directory path on Distributed (Hadoop) File System (DFS) or local
                       filesystem if running in local mode. The model will be writen in this
                       destination and then copied into the model's artifact directory. This is
                       necessary as Spark ML models read from and write to DFS if running on a
                       cluster. If this operation completes successfully, all temporary files
                       created on the DFS are removed. Defaults to ``/tmp/mlflow``.
    :param sample_input: A sample input used to add the MLeap flavor to the model.
                         This must be a PySpark DataFrame that the model can evaluate. If
                         ``sample_input`` is ``None``, the MLeap flavor is not added.

    >>> from pyspark.ml import Pipeline
    >>> from pyspark.ml.classification import LogisticRegression
    >>> from pyspark.ml.feature import HashingTF, Tokenizer
    >>> training = spark.createDataFrame([
    ...   (0, "a b c d e spark", 1.0),
    ...   (1, "b d", 0.0),
    ...   (2, "spark f g h", 1.0),
    ...   (3, "hadoop mapreduce", 0.0) ], ["id", "text", "label"])
    >>> tokenizer = Tokenizer(inputCol="text", outputCol="words")
    >>> hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
    >>> lr = LogisticRegression(maxIter=10, regParam=0.001)
    >>> pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])
    >>> model = pipeline.fit(training)
    >>> mlflow.spark.log_model(model, "spark-model")
    """
    return Model.log(artifact_path=artifact_path, flavor=mlflow.spark, spark_model=spark_model,
                     jars=jars, conda_env=conda_env, dfs_tmpdir=dfs_tmpdir,
                     sample_input=sample_input)


def _tmp_path(dfs_tmp):
    import uuid
    return os.path.join(dfs_tmp, str(uuid.uuid4()))


class _HadoopFileSystem:
    """
    Interface to org.apache.hadoop.fs.FileSystem.

    Spark ML models expect to read from and write to Hadoop FileSystem when running on a cluster.
    Since MLflow works on local directories, we need this interface to copy the files between
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
    def copy_to_local_file(cls, src, dst, remove_src):
        cls._fs().copyToLocalFile(remove_src, cls._remote_path(src), cls._local_path(dst))

    @classmethod
    def copy_from_local_file(cls, src, dst, remove_src):
        cls._fs().copyFromLocalFile(remove_src, cls._local_path(src), cls._remote_path(dst))

    @classmethod
    def qualified_local_path(cls, path):
        return cls._fs().makeQualified(cls._local_path(path)).toString()

    @classmethod
    def maybe_copy_from_local_file(cls, src, dst):
        """
        Conditionally copy the file to the Hadoop DFS.
        The file is copied iff the configuration has distributed filesystem.

        :return: If copied, return new target location, otherwise return (absolute) source path.
        """
        local_path = cls._local_path(src)
        qualified_local_path = cls._fs().makeQualified(local_path).toString()
        if qualified_local_path == "file:" + local_path.toString():
            return local_path.toString()
        cls.copy_from_local_file(src, dst, remove_src=False)
        eprint("Copied SparkML model to %s" % dst)
        return dst

    @classmethod
    def delete(cls, path):
        cls._fs().delete(cls._remote_path(path), True)


def save_model(spark_model, path, mlflow_model=Model(), conda_env=None, jars=None,
               dfs_tmpdir=None, sample_input=None):
    """
    Save a Spark MLlib PipelineModel to a local path.

    By default, this function saves models using the Spark MLlib persistence mechanism.
    Additionally, if a sample input is specified using the ``sample_input`` parameter, the model
    is also serialized in MLeap format and the MLeap flavor is added.

    :param spark_model: Spark PipelineModel to be saved. Can save only PipelineModels.
    :param path: Local path where the model is to be saved.
    :param mlflow_model: MLflow model config this flavor is being added to.
    :param conda_env: Conda environment this model depends on.
    :param jars: List of JARs needed by the model.
    :param dfs_tmpdir: Temporary directory path on Distributed (Hadoop) File System (DFS) or local
                       filesystem if running in local mode. The model will be written in this
                       destination and then copied to the requested local path. This is necessary
                       as Spark ML models read from and write to DFS if running on a cluster. All
                       temporary files created on the DFS will be removed if this operation
                       completes successfully. Defaults to ``/tmp/mlflow``.
    :param sample_input: A sample input that will be used to add the MLeap flavor to the model.
                         This must be a PySpark DataFrame that the model can evaluate. If
                         ``sample_input`` is ``None``, the MLeap flavor is not added.

    >>> from mlflow import spark
    >>> from pyspark.ml.pipeline.PipelineModel
    >>>
    >>> #your pyspark.ml.pipeline.PipelineModel type
    >>> model = ...
    >>> mlflow.spark.save_model(model, "spark-model")
    """
    if jars:
        raise Exception("jar dependencies are not implemented")

    if sample_input is not None:
        mleap.add_to_model(mlflow_model, path, spark_model, sample_input)

    if not isinstance(spark_model, PipelineModel):
        raise Exception("Not a PipelineModel. SparkML can only save PipelineModels.")

    # Spark ML stores the model on DFS if running on a cluster
    # Save it to a DFS temp dir first and copy it to local path
    if dfs_tmpdir is None:
        dfs_tmpdir = DFS_TMP
    tmp_path = _tmp_path(dfs_tmpdir)
    spark_model.save(tmp_path)
    sparkml_data_path_sub = "sparkml"
    sparkml_data_path = os.path.abspath(os.path.join(path, sparkml_data_path_sub))
    _HadoopFileSystem.copy_to_local_file(tmp_path, sparkml_data_path, remove_src=True)
    pyspark_version = pyspark.version.__version__
    model_conda_env = None
    if conda_env:
        model_conda_env = os.path.basename(os.path.abspath(conda_env))
        shutil.copyfile(conda_env, os.path.join(path, model_conda_env))
    mlflow_model.add_flavor(FLAVOR_NAME, pyspark_version=pyspark_version,
                            model_data=sparkml_data_path_sub)
    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.spark", data=sparkml_data_path_sub,
                        env=model_conda_env)
    mlflow_model.save(os.path.join(path, "MLmodel"))


def _load_model(model_path, dfs_tmpdir=None):
    if dfs_tmpdir is None:
        dfs_tmpdir = DFS_TMP
    tmp_path = _tmp_path(dfs_tmpdir)
    # Spark ML expects the model to be stored on DFS
    # Copy the model to a temp DFS location first. We cannot delete this file, as
    # Spark may read from it at any point.
    model_path = _HadoopFileSystem.maybe_copy_from_local_file(model_path, tmp_path)
    return PipelineModel.load(model_path)


def load_model(path, run_id=None, dfs_tmpdir=None):
    """
    Load the Spark MLlib model from the path.

    :param path: Local filesystem path or run-relative artifact path to the model.
    :param run_id: Run ID. If provided, combined with ``path`` to identify the model.
    :param dfs_tmpdir: Temporary directory path on Distributed (Hadoop) File System (DFS) or local
                       filesystem if running in local mode. The model will be loaded from this
                       destination. Defaults to ``/tmp/mlflow``.
    :return: pyspark.ml.pipeline.PipelineModel

    >>> from mlflow import spark
    >>> model = mlflow.spark.load_model("spark-model")
    >>> # Prepare test documents, which are unlabeled (id, text) tuples.
    >>> test = spark.createDataFrame([
    ...   (4, "spark i j k"),
    ...   (5, "l m n"),
    ...   (6, "spark hadoop spark"),
    ...   (7, "apache hadoop")], ["id", "text"])
    >>>  # Make predictions on test documents.
    >>> prediction = model.transform(test)
    """
    if run_id is not None:
        path = mlflow.tracking.utils._get_model_log_dir(model_name=path, run_id=run_id)
    m = Model.load(os.path.join(path, 'MLmodel'))
    if FLAVOR_NAME not in m.flavors:
        raise Exception("Model does not have {} flavor".format(FLAVOR_NAME))
    conf = m.flavors[FLAVOR_NAME]
    model_path = os.path.join(path, conf['model_data'])
    return _load_model(model_path=model_path, dfs_tmpdir=dfs_tmpdir)


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.
    """
    # NOTE: The getOrCreate() call below may change settings of the active session which we do not
    # intend to do here. In particular, setting master to local[1] can break distributed clusters.
    # To avoid this problem, we explicitly check for an active session. This is not ideal but there
    # is no good workaround at the moment.
    spark = pyspark.sql.SparkSession._instantiatedSession
    if spark is None:
        spark = pyspark.sql.SparkSession.builder.config("spark.python.worker.reuse", True)\
            .master("local[1]").getOrCreate()
    return _PyFuncModelWrapper(spark, _load_model(model_path=path))


class _PyFuncModelWrapper(object):
    """
    Wrapper around Spark MLlib PipelineModel providing interface for scoring pandas DataFrame.
    """

    def __init__(self, spark, spark_model):
        self.spark = spark
        self.spark_model = spark_model

    def predict(self, pandas_df):
        """
        Generate predictions given input data in a pandas DataFrame.

        :param pandas_df: pandas DataFrame containing input data.
        :return: List with model predictions.
        """
        spark_df = self.spark.createDataFrame(pandas_df)
        return [x.prediction for x in
                self.spark_model.transform(spark_df).select("prediction").collect()]
