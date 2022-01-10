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
import logging
import posixpath
import re
import shutil
import uuid
import yaml

import mlflow
from mlflow import pyfunc, mleap
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import ModelInputExample, _save_example
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import (
    _mlflow_conda_env,
    _validate_env_arguments,
    _process_pip_requirements,
    _process_conda_env,
    _CONDA_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement
from mlflow.utils.docstring_utils import format_docstring, LOG_MODEL_PARAM_DOCS
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.utils.file_utils import TempDir, write_to
from mlflow.utils.uri import (
    is_local_uri,
    append_to_uri_path,
    dbfs_hdfs_uri_to_fuse_path,
    is_valid_dbfs_uri,
)
from mlflow.utils import databricks_utils
from mlflow.utils.model_utils import _get_flavor_configuration_from_uri
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.utils.autologging_utils import autologging_integration, safe_patch


FLAVOR_NAME = "spark"

# Default temporary directory on DFS. Used to write / read from Spark ML models.
DFS_TMP = "/tmp/mlflow"
_SPARK_MODEL_PATH_SUB = "sparkml"

_logger = logging.getLogger(__name__)


def get_default_pip_requirements():
    """
    :return: A list of default pip requirements for MLflow Models produced by this flavor.
             Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
             that, at minimum, contains these requirements.
    """
    # Strip the suffix from `dev` versions of PySpark, which are not
    # available for installation from Anaconda or PyPI
    pyspark_req = re.sub(r"(\.?)dev.*$", "", _get_pinned_requirement("pyspark"))
    return [pyspark_req]


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
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name="pyspark"))
def log_model(
    spark_model,
    artifact_path,
    conda_env=None,
    dfs_tmpdir=None,
    sample_input=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
):
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
    :param registered_model_name: If given, create a model version under
                                  ``registered_model_name``, also creating a registered model if one
                                  with the given name does not exist.

    :param signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
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
    :param input_example: Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.
    :param await_registration_for: Number of seconds to wait for the model version to finish
                            being created and is in ``READY`` status. By default, the function
                            waits for five minutes. Specify 0 or None to skip waiting.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}

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
    from py4j.protocol import Py4JError

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
        return Model.log(
            artifact_path=artifact_path,
            flavor=mlflow.spark,
            spark_model=spark_model,
            conda_env=conda_env,
            dfs_tmpdir=dfs_tmpdir,
            sample_input=sample_input,
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=input_example,
            await_registration_for=await_registration_for,
            pip_requirements=pip_requirements,
            extra_pip_requirements=extra_pip_requirements,
        )
    model_dir = os.path.join(run_root_artifact_uri, artifact_path)
    # Try to write directly to the artifact repo via Spark. If this fails, defer to Model.log()
    # to persist the model
    try:
        spark_model.save(posixpath.join(model_dir, _SPARK_MODEL_PATH_SUB))
    except Py4JError:
        return Model.log(
            artifact_path=artifact_path,
            flavor=mlflow.spark,
            spark_model=spark_model,
            conda_env=conda_env,
            dfs_tmpdir=dfs_tmpdir,
            sample_input=sample_input,
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=input_example,
            await_registration_for=await_registration_for,
            pip_requirements=pip_requirements,
            extra_pip_requirements=extra_pip_requirements,
        )

    # Otherwise, override the default model log behavior and save model directly to artifact repo
    mlflow_model = Model(artifact_path=artifact_path, run_id=run_id)
    with TempDir() as tmp:
        tmp_model_metadata_dir = tmp.path()
        _save_model_metadata(
            tmp_model_metadata_dir,
            spark_model,
            mlflow_model,
            sample_input,
            conda_env,
            signature=signature,
            input_example=input_example,
        )
        mlflow.tracking.fluent.log_artifacts(tmp_model_metadata_dir, artifact_path)
        if registered_model_name is not None:
            mlflow.register_model(
                "runs:/%s/%s" % (run_id, artifact_path),
                registered_model_name,
                await_registration_for,
            )


def _tmp_path(dfs_tmp):
    return posixpath.join(dfs_tmp, str(uuid.uuid4()))


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
        except Exception as ex:
            # Log a debug-level message, since existence checks may raise exceptions
            # in normal operating circumstances that do not warrant warnings
            _logger.debug(
                "Unexpected exception while checking if model uri is visible on " "DFS: %s", ex
            )
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
        except Exception:
            _logger.info("URI '%s' does not point to the current DFS.", src_uri)
        _logger.info("File '%s' not found on DFS. Will attempt to upload the file.", src_uri)
        return cls.maybe_copy_from_local_file(_download_artifact_from_uri(src_uri), dst_path)

    @classmethod
    def delete(cls, path):
        cls._fs().delete(cls._remote_path(path), True)


def _save_model_metadata(
    dst_dir,
    spark_model,
    mlflow_model,
    sample_input,
    conda_env,
    signature=None,
    input_example=None,
    pip_requirements=None,
    extra_pip_requirements=None,
):
    """
    Saves model metadata into the passed-in directory. The persisted metadata assumes that a
    model can be loaded from a relative path to the metadata file (currently hard-coded to
    "sparkml").
    """
    import pyspark

    if sample_input is not None:
        mleap.add_to_model(
            mlflow_model=mlflow_model,
            path=dst_dir,
            spark_model=spark_model,
            sample_input=sample_input,
        )
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, dst_dir)

    mlflow_model.add_flavor(
        FLAVOR_NAME, pyspark_version=pyspark.__version__, model_data=_SPARK_MODEL_PATH_SUB
    )
    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.spark",
        data=_SPARK_MODEL_PATH_SUB,
        env=_CONDA_ENV_FILE_NAME,
    )
    mlflow_model.save(os.path.join(dst_dir, MLMODEL_FILE_NAME))

    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements()
            # To ensure `_load_pyfunc` can successfully load the model during the dependency
            # inference, `mlflow_model.save` must be called beforehand to save an MLmodel file.
            inferred_reqs = mlflow.models.infer_pip_requirements(
                dst_dir,
                FLAVOR_NAME,
                fallback=default_reqs,
            )
            default_reqs = sorted(set(inferred_reqs).union(default_reqs))
        else:
            default_reqs = None
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(
            default_reqs,
            pip_requirements,
            extra_pip_requirements,
        )
    else:
        conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)

    with open(os.path.join(dst_dir, _CONDA_ENV_FILE_NAME), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    # Save `constraints.txt` if necessary
    if pip_constraints:
        write_to(os.path.join(dst_dir, _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))

    # Save `requirements.txt`
    write_to(os.path.join(dst_dir, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))


def _validate_model(spark_model):
    from pyspark.ml.util import MLReadable, MLWritable
    from pyspark.ml import Model as PySparkModel

    if (
        not isinstance(spark_model, PySparkModel)
        or not isinstance(spark_model, MLReadable)
        or not isinstance(spark_model, MLWritable)
    ):
        raise MlflowException(
            "Cannot serialize this model. MLflow can only save descendants of pyspark.Model"
            "that implement MLWritable and MLReadable.",
            INVALID_PARAMETER_VALUE,
        )


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name="pyspark"))
def save_model(
    spark_model,
    path,
    mlflow_model=None,
    conda_env=None,
    dfs_tmpdir=None,
    sample_input=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
):
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

    :param signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
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
    :param input_example: Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}

    .. code-block:: python
        :caption: Example

        from mlflow import spark
        from pyspark.ml.pipeline.PipelineModel

        # your pyspark.ml.pipeline.PipelineModel type
        model = ...
        mlflow.spark.save_model(model, "spark-model")
    """
    _validate_model(spark_model)
    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

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
    # We're copying the Spark model from DBFS to the local filesystem if (a) the temporary DFS URI
    # we saved the Spark model to is a DBFS URI ("dbfs:/my-directory"), or (b) if we're running
    # on a Databricks cluster and the URI is schemeless (e.g. looks like a filesystem absolute path
    # like "/my-directory")
    copying_from_dbfs = is_valid_dbfs_uri(tmp_path) or (
        databricks_utils.is_in_cluster() and posixpath.abspath(tmp_path) == tmp_path
    )
    if copying_from_dbfs and databricks_utils.is_dbfs_fuse_available():
        tmp_path_fuse = dbfs_hdfs_uri_to_fuse_path(tmp_path)
        shutil.move(src=tmp_path_fuse, dst=sparkml_data_path)
    else:
        _HadoopFileSystem.copy_to_local_file(tmp_path, sparkml_data_path, remove_src=True)
    _save_model_metadata(
        dst_dir=path,
        spark_model=spark_model,
        mlflow_model=mlflow_model,
        sample_input=sample_input,
        conda_env=conda_env,
        signature=signature,
        input_example=input_example,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
    )


def _shutil_copytree_without_file_permissions(src_dir, dst_dir):
    """
    Copies the directory src_dir into dst_dir, without preserving filesystem permissions
    """
    for (dirpath, dirnames, filenames) in os.walk(src_dir):
        for dirname in dirnames:
            relative_dir_path = os.path.relpath(os.path.join(dirpath, dirname), src_dir)
            # For each directory <dirname> immediately under <dirpath>, create an equivalently-named
            # directory under the destination directory
            abs_dir_path = os.path.join(dst_dir, relative_dir_path)
            os.mkdir(abs_dir_path)
        for filename in filenames:
            # For each file with name <filename> immediately under <dirpath>, copy that file to
            # the appropriate location in the destination directory
            file_path = os.path.join(dirpath, filename)
            relative_file_path = os.path.relpath(file_path, src_dir)
            abs_file_path = os.path.join(dst_dir, relative_file_path)
            shutil.copyfile(file_path, abs_file_path)


def _load_model_databricks(model_uri, dfs_tmpdir):
    from pyspark.ml.pipeline import PipelineModel

    # Download model saved to remote URI to local filesystem
    local_model_path = _download_artifact_from_uri(model_uri)
    # Spark ML expects the model to be stored on DFS
    # Copy the model to a temp DFS location first. We cannot delete this file, as
    # Spark may read from it at any point.
    fuse_dfs_tmpdir = dbfs_hdfs_uri_to_fuse_path(dfs_tmpdir)
    os.mkdir(fuse_dfs_tmpdir)
    # Workaround for inability to use shutil.copytree with DBFS FUSE due to permission-denied
    # errors on passthrough-enabled clusters when attempting to copy permission bits for directories
    _shutil_copytree_without_file_permissions(src_dir=local_model_path, dst_dir=fuse_dfs_tmpdir)
    return PipelineModel.load(dfs_tmpdir)


def _load_model(model_uri, dfs_tmpdir_base=None):
    from pyspark.ml.pipeline import PipelineModel

    if dfs_tmpdir_base is None:
        dfs_tmpdir_base = DFS_TMP
    dfs_tmpdir = _tmp_path(dfs_tmpdir_base)
    if databricks_utils.is_in_cluster() and databricks_utils.is_dbfs_fuse_available():
        return _load_model_databricks(model_uri, dfs_tmpdir)
    model_uri = _HadoopFileSystem.maybe_copy_from_uri(model_uri, dfs_tmpdir)
    return PipelineModel.load(model_uri)


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
    return _load_model(model_uri=model_uri, dfs_tmpdir_base=dfs_tmpdir)


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
        # NB: If there is no existing Spark context, create a new local one.
        # NB: We're disabling caching on the new context since we do not need it and we want to
        # avoid overwriting cache of underlying Spark cluster when executed on a Spark Worker
        # (e.g. as part of spark_udf).
        spark = (
            pyspark.sql.SparkSession.builder.config("spark.python.worker.reuse", True)
            .config("spark.databricks.io.cache.enabled", False)
            # In Spark 3.1 and above, we need to set this conf explicitly to enable creating
            # a SparkSession on the workers
            .config("spark.executor.allowSparkContext", "true")
            .master("local[1]")
            .getOrCreate()
        )
    return _PyFuncModelWrapper(spark, _load_model(model_uri=path))


class _PyFuncModelWrapper:
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
        return [
            x.prediction
            for x in self.spark_model.transform(spark_df).select("prediction").collect()
        ]


@autologging_integration(FLAVOR_NAME)
def autolog(disable=False, silent=False):  # pylint: disable=unused-argument
    """
    Enables (or disables) and configures logging of Spark datasource paths, versions
    (if applicable), and formats when they are read. This method is not threadsafe and assumes a
    `SparkSession
    <https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.SparkSession>`_
    already exists with the
    `mlflow-spark JAR
    <http://mlflow.org/docs/latest/tracking.html#automatic-logging-from-spark-experimental>`_
    attached. It should be called on the Spark driver, not on the executors (i.e. do not call
    this method within a function parallelized by Spark). This API requires Spark 3.0 or above.

    Datasource information is cached in memory and logged to all subsequent MLflow runs,
    including the active MLflow run (if one exists when the data is read). Note that autologging of
    Spark ML (MLlib) models is not currently supported via this API. Datasource autologging is
    best-effort, meaning that if Spark is under heavy load or MLflow logging fails for any reason
    (e.g., if the MLflow server is unavailable), logging may be dropped.

    For any unexpected issues with autologging, check Spark driver and executor logs in addition
    to stderr & stdout generated from your MLflow code - datasource information is pulled from
    Spark, so logs relevant to debugging may show up amongst the Spark logs.

    .. code-block:: python
        :caption: Example

        import mlflow.spark
        import os
        import shutil
        from pyspark.sql import SparkSession
        # Create and persist some dummy data
        # Note: On environments like Databricks with pre-created SparkSessions,
        # ensure the org.mlflow:mlflow-spark:1.11.0 is attached as a library to
        # your cluster
        spark = (SparkSession.builder
                    .config("spark.jars.packages", "org.mlflow:mlflow-spark:1.11.0")
                    .master("local[*]")
                    .getOrCreate())
        df = spark.createDataFrame([
                (4, "spark i j k"),
                (5, "l m n"),
                (6, "spark hadoop spark"),
                (7, "apache hadoop")], ["id", "text"])
        import tempfile
        tempdir = tempfile.mkdtemp()
        df.write.csv(os.path.join(tempdir, "my-data-path"), header=True)
        # Enable Spark datasource autologging.
        mlflow.spark.autolog()
        loaded_df = spark.read.csv(os.path.join(tempdir, "my-data-path"),
                        header=True, inferSchema=True)
        # Call toPandas() to trigger a read of the Spark datasource. Datasource info
        # (path and format) is logged to the current active run, or the
        # next-created MLflow run if no run is currently active
        with mlflow.start_run() as active_run:
            pandas_df = loaded_df.toPandas()

    :param disable: If ``True``, disables the Spark datasource autologging integration.
                    If ``False``, enables the Spark datasource autologging integration.
    :param silent: If ``True``, suppress all event logs and warnings from MLflow during Spark
                   datasource autologging. If ``False``, show all events and warnings during Spark
                   datasource autologging.
    """
    from mlflow.utils._spark_utils import _get_active_spark_session
    from mlflow._spark_autologging import _listen_for_spark_activity
    from pyspark.sql import SparkSession
    from pyspark import SparkContext

    def __init__(original, self, *args, **kwargs):
        original(self, *args, **kwargs)

        _listen_for_spark_activity(self._sc)

    safe_patch(FLAVOR_NAME, SparkSession, "__init__", __init__, manage_run=False)

    active_session = _get_active_spark_session()
    if active_session is not None:
        # We know SparkContext exists here already, so get it
        sc = SparkContext.getOrCreate()

        _listen_for_spark_activity(sc)
