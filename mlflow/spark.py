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
import os
import yaml
import logging
import re
import traceback

import mlflow
from mlflow import pyfunc, mleap
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import ModelInputExample, _save_example
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.utils.file_utils import TempDir
from mlflow.utils.uri import is_local_uri, append_to_uri_path
from mlflow.utils.model_utils import _get_flavor_configuration_from_uri
from mlflow.utils import experimental


FLAVOR_NAME = "spark"

# Default temporary directory on DFS. Used to write / read from Spark ML models.
DFS_TMP = "/tmp/mlflow"
_SPARK_MODEL_PATH_SUB = "sparkml"

_logger = logging.getLogger(__name__)


def _format_exception(ex):
    return "".join(traceback.format_exception(type(ex), ex, ex.__traceback__))


def get_default_conda_env():
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`. This Conda environment
             contains the current version of PySpark that is installed on the caller's
             system. ``dev`` versions of PySpark are replaced with stable versions in
             the resulting Conda environment (e.g., if you are running PySpark version
             ``2.4.5.dev0``, invoking this method produces a Conda environment with a
             dependency on PySpark version ``2.4.5``).
    """
    import pyspark
    # Strip the suffix from `dev` versions of PySpark, which are not
    # available for installation from Anaconda or PyPI
    pyspark_version = re.sub(r"(\.?)dev.*", "", pyspark.__version__)

    return _mlflow_conda_env(
        additional_conda_deps=[
            "pyspark={}".format(pyspark_version),
        ],
        additional_pip_deps=None,
        additional_conda_channels=None)


def log_model(spark_model, artifact_path, conda_env=None, dfs_tmpdir=None,
              sample_input=None, registered_model_name=None,
              signature: ModelSignature=None, input_example: ModelInputExample=None):
    """
    Log a Spark MLlib model as an MLflow artifact for the current run. This uses the
    MLlib persistence format and produces an MLflow Model with the Spark flavor.

    Note: If no run is active, it will instantiate a run to obtain a run_id.

    :param spark_model: Spark model to be saved - MLflow can only save descendants of
                        pyspark.ml.Model which implement MLReadable and MLWritable.
    :param artifact_path: Run relative artifact path.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this decsribes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in :func:`get_default_conda_env()`. If `None`, the default
                      :func:`get_default_conda_env()` environment is added to the model.
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
    :param dfs_tmpdir: Temporary directory path on Distributed (Hadoop) File System (DFS) or local
                       filesystem if running in local mode. The model is written in this
                       destination and then copied into the model's artifact directory. This is
                       necessary as Spark ML models read from and write to DFS if running on a
                       cluster. If this operation completes successfully, all temporary files
                       created on the DFS are removed. Defaults to ``/tmp/mlflow``.
    :param sample_input: A sample input used to add the MLeap flavor to the model.
                         This must be a PySpark DataFrame that the model can evaluate. If
                         ``sample_input`` is ``None``, the MLeap flavor is not added.
    :param registered_model_name: (Experimental) If given, create a model version under
                                  ``registered_model_name``, also creating a registered model if one
                                  with the given name does not exist.

    :param signature: (Experimental) :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:

                      .. code-block:: python

                        from mlflow.models.signature import infer_signature
                        train = df.drop_column("target_label")
                        predictions = ... # compute model predictions
                        signature = infer_signature(train, predictions)
    :param input_example: (Experimental) Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.



    .. code-block:: python
        :caption: Example

        from pyspark.ml import Pipeline
        from pyspark.ml.classification import LogisticRegression
        from pyspark.ml.feature import HashingTF, Tokenizer
        training = spark.createDataFrame([
            (0, "a b c d e spark", 1.0),
            (1, "b d", 0.0),
            (2, "spark f g h", 1.0),
            (3, "hadoop mapreduce", 0.0) ], ["id", "text", "label"])
        tokenizer = Tokenizer(inputCol="text", outputCol="words")
        hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
        lr = LogisticRegression(maxIter=10, regParam=0.001)
        pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])
        model = pipeline.fit(training)
        mlflow.spark.log_model(model, "spark-model")
    """
    from py4j.protocol import Py4JJavaError

    _validate_model(spark_model)
    from pyspark.ml import PipelineModel
    if not isinstance(spark_model, PipelineModel):
        spark_model = PipelineModel([spark_model])
    run_id = mlflow.tracking.fluent._get_or_start_run().info.run_id
    run_root_artifact_uri = mlflow.get_artifact_uri()
    # If the artifact URI is a local filesystem path, defer to Model.log() to persist the model,
    # since Spark may not be able to write directly to the driver's filesystem. For example,
    # writing to `file:/uri` will write to the local filesystem from each executor, which will
    # be incorrect on multi-node clusters - to avoid such issues we just use the Model.log() path
    # here.
    if is_local_uri(run_root_artifact_uri):
        return Model.log(artifact_path=artifact_path, flavor=mlflow.spark, spark_model=spark_model,
                         conda_env=conda_env, dfs_tmpdir=dfs_tmpdir, sample_input=sample_input,
                         registered_model_name=registered_model_name)
    # If Spark cannot write directly to the artifact repo, defer to Model.log() to persist the
    # model
    model_dir = os.path.join(run_root_artifact_uri, artifact_path)
    try:
        spark_model.save(os.path.join(model_dir, _SPARK_MODEL_PATH_SUB))
    except Py4JJavaError:
        return Model.log(artifact_path=artifact_path, flavor=mlflow.spark, spark_model=spark_model,
                         conda_env=conda_env, dfs_tmpdir=dfs_tmpdir, sample_input=sample_input,
                         registered_model_name=registered_model_name,
                         signature=signature, input_example=input_example)

    # Otherwise, override the default model log behavior and save model directly to artifact repo
    mlflow_model = Model(artifact_path=artifact_path, run_id=run_id)
    with TempDir() as tmp:
        tmp_model_metadata_dir = tmp.path()
        _save_model_metadata(
            tmp_model_metadata_dir, spark_model, mlflow_model, sample_input, conda_env,
            signature=signature, input_example=input_example)
        mlflow.tracking.fluent.log_artifacts(tmp_model_metadata_dir, artifact_path)
        if registered_model_name is not None:
            mlflow.register_model("runs:/%s/%s" % (run_id, artifact_path), registered_model_name)


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
        from pyspark import SparkContext

        return SparkContext._gateway.jvm

    @classmethod
    def _fs(cls):
        if not cls._filesystem:
            cls._filesystem = cls._jvm().org.apache.hadoop.fs.FileSystem.get(cls._conf())
        return cls._filesystem

    @classmethod
    def _conf(cls):
        from pyspark import SparkContext

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
    def _try_file_exists(cls, dfs_path):
        try:
            return cls._fs().exists(dfs_path)
        except Exception as ex:  # pylint: disable=broad-except
            _logger.warning(
                "Unexpected exception while checking if model uri is visible on "
                "DFS: %s", ex)
        return False

    @classmethod
    def maybe_copy_from_uri(cls, src_uri, dst_path):
        """
        Conditionally copy the file to the Hadoop DFS from the source uri.
        In case the file is already on the Hadoop DFS do nothing.

        :return: If copied, return new target location, otherwise return source uri.
        """
        try:
            # makeQualified throws if wrong schema / uri
            dfs_path = cls._fs().makeQualified(cls._remote_path(src_uri))
            if cls._try_file_exists(dfs_path):
                _logger.info("File '%s' is already on DFS, copy is not necessary.", src_uri)
                return src_uri
        except Exception:  # pylint: disable=broad-except
            _logger.info("URI '%s' does not point to the current DFS.", src_uri)
        _logger.info("File '%s' not found on DFS. Will attempt to upload the file.", src_uri)
        return cls.maybe_copy_from_local_file(_download_artifact_from_uri(src_uri), dst_path)

    @classmethod
    def delete(cls, path):
        cls._fs().delete(cls._remote_path(path), True)


def _save_model_metadata(dst_dir, spark_model, mlflow_model, sample_input, conda_env,
                         signature=None, input_example=None):
    """
    Saves model metadata into the passed-in directory. The persisted metadata assumes that a
    model can be loaded from a relative path to the metadata file (currently hard-coded to
    "sparkml").
    """
    import pyspark

    if sample_input is not None:
        mleap.add_to_model(mlflow_model=mlflow_model, path=dst_dir, spark_model=spark_model,
                           sample_input=sample_input)
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, dst_dir)
    conda_env_subpath = "conda.yaml"
    if conda_env is None:
        conda_env = get_default_conda_env()
    elif not isinstance(conda_env, dict):
        with open(conda_env, "r") as f:
            conda_env = yaml.safe_load(f)
    with open(os.path.join(dst_dir, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    mlflow_model.add_flavor(FLAVOR_NAME, pyspark_version=pyspark.__version__,
                            model_data=_SPARK_MODEL_PATH_SUB)
    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.spark", data=_SPARK_MODEL_PATH_SUB,
                        env=conda_env_subpath)
    mlflow_model.save(os.path.join(dst_dir, MLMODEL_FILE_NAME))


def _validate_model(spark_model):
    from pyspark.ml.util import MLReadable, MLWritable
    from pyspark.ml import Model as PySparkModel
    if not isinstance(spark_model, PySparkModel) \
            or not isinstance(spark_model, MLReadable) \
            or not isinstance(spark_model, MLWritable):
        raise MlflowException(
            "Cannot serialize this model. MLflow can only save descendants of pyspark.Model"
            "that implement MLWritable and MLReadable.",
            INVALID_PARAMETER_VALUE)


def save_model(spark_model, path, mlflow_model=None, conda_env=None,
               dfs_tmpdir=None, sample_input=None,
               signature: ModelSignature=None, input_example: ModelInputExample=None):
    """
    Save a Spark MLlib Model to a local path.

    By default, this function saves models using the Spark MLlib persistence mechanism.
    Additionally, if a sample input is specified using the ``sample_input`` parameter, the model
    is also serialized in MLeap format and the MLeap flavor is added.

    :param spark_model: Spark model to be saved - MLflow can only save descendants of
                        pyspark.ml.Model which implement MLReadable and MLWritable.
    :param path: Local path where the model is to be saved.
    :param mlflow_model: MLflow model config this flavor is being added to.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this decsribes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in :func:`get_default_conda_env()`. If `None`, the default
                      :func:`get_default_conda_env()` environment is added to the model.
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
    :param dfs_tmpdir: Temporary directory path on Distributed (Hadoop) File System (DFS) or local
                       filesystem if running in local mode. The model is be written in this
                       destination and then copied to the requested local path. This is necessary
                       as Spark ML models read from and write to DFS if running on a cluster. All
                       temporary files created on the DFS are removed if this operation
                       completes successfully. Defaults to ``/tmp/mlflow``.
    :param sample_input: A sample input that is used to add the MLeap flavor to the model.
                         This must be a PySpark DataFrame that the model can evaluate. If
                         ``sample_input`` is ``None``, the MLeap flavor is not added.

    :param signature: (Experimental) :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:

                      .. code-block:: python

                        from mlflow.models.signature import infer_signature
                        train = df.drop_column("target_label")
                        predictions = ... # compute model predictions
                        signature = infer_signature(train, predictions)
    :param input_example: (Experimental) Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.



    .. code-block:: python
        :caption: Example

        from mlflow import spark
        from pyspark.ml.pipeline.PipelineModel

        # your pyspark.ml.pipeline.PipelineModel type
        model = ...
        mlflow.spark.save_model(model, "spark-model")
    """
    _validate_model(spark_model)
    from pyspark.ml import PipelineModel
    if not isinstance(spark_model, PipelineModel):
        spark_model = PipelineModel([spark_model])
    if mlflow_model is None:
        mlflow_model = Model()
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
        sample_input=sample_input, conda_env=conda_env,
        signature=signature, input_example=input_example)


def _load_model(model_uri, dfs_tmpdir=None):
    from pyspark.ml.pipeline import PipelineModel
    if dfs_tmpdir is None:
        dfs_tmpdir = DFS_TMP
    tmp_path = _tmp_path(dfs_tmpdir)
    # Spark ML expects the model to be stored on DFS
    # Copy the model to a temp DFS location first. We cannot delete this file, as
    # Spark may read from it at any point.
    model_path = _HadoopFileSystem.maybe_copy_from_uri(model_uri, tmp_path)
    return PipelineModel.load(model_path)


def load_model(model_uri, dfs_tmpdir=None):
    """
    Load the Spark MLlib model from the path.

    :param model_uri: The location, in URI format, of the MLflow model, for example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
                      - ``models:/<model_name>/<model_version>``
                      - ``models:/<model_name>/<stage>``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
                      artifact-locations>`_.
    :param dfs_tmpdir: Temporary directory path on Distributed (Hadoop) File System (DFS) or local
                       filesystem if running in local mode. The model is loaded from this
                       destination. Defaults to ``/tmp/mlflow``.
    :return: pyspark.ml.pipeline.PipelineModel

    .. code-block:: python
        :caption: Example

        from mlflow import spark
        model = mlflow.spark.load_model("spark-model")
        # Prepare test documents, which are unlabeled (id, text) tuples.
        test = spark.createDataFrame([
            (4, "spark i j k"),
            (5, "l m n"),
            (6, "spark hadoop spark"),
            (7, "apache hadoop")], ["id", "text"])
        # Make predictions on test documents
        prediction = model.transform(test)
    """
    if RunsArtifactRepository.is_runs_uri(model_uri):
        runs_uri = model_uri
        model_uri = RunsArtifactRepository.get_underlying_uri(model_uri)
        _logger.info("'%s' resolved as '%s'", runs_uri, model_uri)
    elif ModelsArtifactRepository.is_models_uri(model_uri):
        runs_uri = model_uri
        model_uri = ModelsArtifactRepository.get_underlying_uri(model_uri)
        _logger.info("'%s' resolved as '%s'", runs_uri, model_uri)
    flavor_conf = _get_flavor_configuration_from_uri(model_uri, FLAVOR_NAME)
    model_uri = append_to_uri_path(model_uri, flavor_conf["model_data"])
    return _load_model(model_uri=model_uri, dfs_tmpdir=dfs_tmpdir)


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.

    :param path: Local filesystem path to the MLflow Model with the ``spark`` flavor.
    """
    # NOTE: The getOrCreate() call below may change settings of the active session which we do not
    # intend to do here. In particular, setting master to local[1] can break distributed clusters.
    # To avoid this problem, we explicitly check for an active session. This is not ideal but there
    # is no good workaround at the moment.
    import pyspark

    spark = pyspark.sql.SparkSession._instantiatedSession
    if spark is None:
        spark = pyspark.sql.SparkSession.builder.config("spark.python.worker.reuse", True) \
            .master("local[1]").getOrCreate()
    return _PyFuncModelWrapper(spark, _load_model(model_uri=path))


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


@experimental
def autolog():
    """
    Enables automatic logging of Spark datasource paths, versions (if applicable), and formats
    when they are read. This method is not threadsafe and assumes a
    `SparkSession
    <https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.SparkSession>`_
    already exists with the
    `mlflow-spark JAR
    <http://mlflow.org/docs/latest/tracking.html#automatic-logging-from-spark-experimental>`_
    attached. It should be called on the Spark driver, not on the executors (i.e. do not call
    this method within a function parallelized by Spark). This API requires Spark 3.0 or above.

    Datasource information is logged under the current active MLflow run. If no active run
    exists, datasource information is cached in memory & logged to the next-created active run
    (but not to successive runs). Note that autologging of Spark ML (MLlib) models is not currently
    supported via this API. Datasource-autologging is best-effort, meaning that if Spark is under
    heavy load or MLflow logging fails for any reason (e.g., if the MLflow server is unavailable),
    logging may be dropped.

    For any unexpected issues with autologging, check Spark driver and executor logs in addition
    to stderr & stdout generated from your MLflow code - datasource information is pulled from
    Spark, so logs relevant to debugging may show up amongst the Spark logs.

    .. code-block:: python
        :caption: Example

        import mlflow.spark
        from pyspark.sql import SparkSession
        # Create and persist some dummy data
        spark = (SparkSession.builder
                    .config("spark.jars.packages", "org.mlflow.mlflow-spark")
                    .getOrCreate())
        df = spark.createDataFrame([
                (4, "spark i j k"),
                (5, "l m n"),
                (6, "spark hadoop spark"),
                (7, "apache hadoop")], ["id", "text"])
        import tempfile
        tempdir = tempfile.mkdtemp()
        df.write.format("csv").save(tempdir)
        # Enable Spark datasource autologging.
        mlflow.spark.autolog()
        loaded_df = spark.read.format("csv").load(tempdir)
        # Call collect() to trigger a read of the Spark datasource. Datasource info
        # (path and format)is automatically logged to an MLflow run.
        loaded_df.collect()
        shutil.rmtree(tempdir) # clean up tempdir
    """
    from mlflow import _spark_autologging
    _spark_autologging.autolog()
