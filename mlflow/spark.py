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
import yaml
import logging

from py4j.protocol import Py4JJavaError
import pyspark
from pyspark import SparkContext
from pyspark.ml.pipeline import PipelineModel

import mlflow
from mlflow import pyfunc, mleap
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.utils.file_utils import TempDir

FLAVOR_NAME = "spark"

# Default temporary directory on DFS. Used to write / read from Spark ML models.
DFS_TMP = "/tmp/mlflow"
_SPARK_MODEL_PATH_SUB = "sparkml"

DEFAULT_CONDA_ENV = _mlflow_conda_env(
    additional_conda_deps=[
        "pyspark={}".format(pyspark.__version__),
    ],
    additional_pip_deps=None,
    additional_conda_channels=None,
)


_logger = logging.getLogger(__name__)


def log_model(spark_model, artifact_path, conda_env=None, jars=None, dfs_tmpdir=None,
              sample_input=None):
    """
    Log a Spark MLlib model as an MLflow artifact for the current run. This uses the
    MLlib persistence format, and the logged model will have the Spark flavor.

    :param spark_model: PipelineModel to be saved.
    :param artifact_path: Run relative artifact path.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this decribes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in ``mlflow.spark.DEFAULT_CONDA_ENV``. If `None`, the default
                      ``mlflow.spark.DEFAULT_CONDA_ENV`` environment will be added to the model.
                      The following is an *example* dictionary representation of a Conda
                      environment::

                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults'],
                            'dependencies': [
                                'python=3.7.0',
                                'pyspark=2.3.0'
                            ]
                        }

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
    _validate_model(spark_model, jars)
    run_id = mlflow.tracking.fluent._get_or_start_run().info.run_uuid
    run_root_artifact_uri = mlflow.get_artifact_uri()
    # If the artifact URI is a local filesystem path, defer to Model.log() to persist the model,
    # since Spark may not be able to write directly to the driver's filesystem. For example,
    # writing to `file:/uri` will write to the local filesystem from each executor, which will
    # be incorrect on multi-node clusters - to avoid such issues we just use the Model.log() path
    # here.
    if mlflow.tracking.utils._is_local_uri(run_root_artifact_uri):
        return Model.log(artifact_path=artifact_path, flavor=mlflow.spark, spark_model=spark_model,
                         jars=jars, conda_env=conda_env, dfs_tmpdir=dfs_tmpdir,
                         sample_input=sample_input)
    # If Spark cannot write directly to the artifact repo, defer to Model.log() to persist the
    # model
    model_dir = os.path.join(run_root_artifact_uri, artifact_path)
    try:
        spark_model.save(os.path.join(model_dir, _SPARK_MODEL_PATH_SUB))
    except Py4JJavaError:
        return Model.log(artifact_path=artifact_path, flavor=mlflow.spark, spark_model=spark_model,
                         jars=jars, conda_env=conda_env, dfs_tmpdir=dfs_tmpdir,
                         sample_input=sample_input)

    # Otherwise, override the default model log behavior and save model directly to artifact repo
    mlflow_model = Model(artifact_path=artifact_path, run_id=run_id)
    with TempDir() as tmp:
        tmp_model_metadata_dir = tmp.path()
        _save_model_metadata(
            tmp_model_metadata_dir, spark_model, mlflow_model, sample_input, conda_env)
        mlflow.tracking.fluent.log_artifacts(tmp_model_metadata_dir, artifact_path)


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
            cls._filesystem = cls._jvm().org.apache.hadoop.fs.FileSystem.get(cls._conf())
        return cls._filesystem

    @classmethod
    def _conf(cls):
        sc = SparkContext.getOrCreate()
        return sc._jsc.hadoopConfiguration()

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
        _logger.info("Copied SparkML model to %s", dst)
        return dst

    @classmethod
    def delete(cls, path):
        cls._fs().delete(cls._remote_path(path), True)


def _save_model_metadata(dst_dir, spark_model, mlflow_model, sample_input, conda_env):
    """
    Saves model metadata into the passed-in directory. The persisted metadata assumes that a
    model can be loaded from a relative path to the metadata file (currently hard-coded to
    "sparkml").
    """
    if sample_input is not None:
        mleap.add_to_model(mlflow_model, dst_dir, spark_model, sample_input)

    pyspark_version = pyspark.version.__version__

    conda_env_subpath = "conda.yaml"
    if conda_env is None:
        conda_env = DEFAULT_CONDA_ENV
    elif not isinstance(conda_env, dict):
        with open(conda_env, "r") as f:
            conda_env = yaml.safe_load(f)
    with open(os.path.join(dst_dir, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    mlflow_model.add_flavor(FLAVOR_NAME, pyspark_version=pyspark_version,
                            model_data=_SPARK_MODEL_PATH_SUB)
    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.spark", data=_SPARK_MODEL_PATH_SUB,
                        env=conda_env_subpath)
    mlflow_model.save(os.path.join(dst_dir, "MLmodel"))


def _validate_model(spark_model, jars):
    if not isinstance(spark_model, PipelineModel):
        raise MlflowException("Not a PipelineModel. SparkML can only save PipelineModels.",
                              INVALID_PARAMETER_VALUE)
    if jars:
        raise MlflowException("JAR dependencies are not implemented", INVALID_PARAMETER_VALUE)


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
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this decribes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in ``mlflow.spark.DEFAULT_CONDA_ENV``. If `None`, the default
                      ``mlflow.spark.DEFAULT_CONDA_ENV`` environment will be added to the model.
                      The following is an *example* dictionary representation of a Conda
                      environment::

                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults'],
                            'dependencies': [
                                'python=3.7.0',
                                'pyspark=2.3.0'
                            ]
                        }

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
    _validate_model(spark_model, jars)
    # Spark ML stores the model on DFS if running on a cluster
    # Save it to a DFS temp dir first and copy it to local path
    if dfs_tmpdir is None:
        dfs_tmpdir = DFS_TMP
    tmp_path = _tmp_path(dfs_tmpdir)
    spark_model.save(tmp_path)
    sparkml_data_path = os.path.abspath(os.path.join(path, _SPARK_MODEL_PATH_SUB))
    _HadoopFileSystem.copy_to_local_file(tmp_path, sparkml_data_path, remove_src=True)
    _save_model_metadata(
        dst_dir=path, spark_model=spark_model, mlflow_model=mlflow_model,
        sample_input=sample_input, conda_env=conda_env)


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
    path = os.path.abspath(path)
    flavor_conf = _get_flavor_configuration(model_path=path, flavor_name=FLAVOR_NAME)
    spark_model_artifacts_path = os.path.join(path, flavor_conf['model_data'])
    return _load_model(model_path=spark_model_artifacts_path, dfs_tmpdir=dfs_tmpdir)


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
