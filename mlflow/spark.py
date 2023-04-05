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
import yaml
from packaging.version import Version

import mlflow
from mlflow import environment_variables, pyfunc, mleap
from mlflow.environment_variables import MLFLOW_DFS_TMP
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import ModelInputExample, _save_example
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.tracking.artifact_utils import (
    _download_artifact_from_uri,
    _get_root_uri_and_artifact_path,
)
from mlflow.utils.environment import (
    _mlflow_conda_env,
    _validate_env_arguments,
    _process_pip_requirements,
    _process_conda_env,
    _CONDA_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _PythonEnv,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement
from mlflow.utils.docstring_utils import format_docstring, LOG_MODEL_PARAM_DOCS
from mlflow.store.artifact.databricks_artifact_repo import DatabricksArtifactRepository
from mlflow.utils.file_utils import TempDir, write_to, shutil_copytree_without_file_permissions
from mlflow.utils.uri import (
    is_local_uri,
    append_to_uri_path,
    dbfs_hdfs_uri_to_fuse_path,
    is_valid_dbfs_uri,
    is_databricks_acled_artifacts_uri,
    get_databricks_profile_uri_from_artifact_uri,
    generate_tmp_dfs_path,
)
from mlflow.utils import databricks_utils
from mlflow.utils.model_utils import (
    _get_flavor_configuration_from_uri,
    _validate_and_copy_code_paths,
    _add_code_from_conf_to_system_path,
)
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.utils.autologging_utils import autologging_integration, safe_patch


FLAVOR_NAME = "spark"

_SPARK_MODEL_PATH_SUB = "sparkml"
_MLFLOWDBFS_SCHEME = "mlflowdbfs"

_logger = logging.getLogger(__name__)


def get_default_pip_requirements():
    """
    :return: A list of default pip requirements for MLflow Models produced by this flavor.
             Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
             that, at minimum, contains these requirements.
    """
    import pyspark

    # Strip the suffix from `dev` versions of PySpark, which are not
    # available for installation from Anaconda or PyPI
    pyspark_req = re.sub(r"(\.?)dev.*$", "", _get_pinned_requirement("pyspark"))
    reqs = [pyspark_req]
    if Version(pyspark.__version__) <= Version("3.3.2"):
        # Versions of PySpark <= 3.3.2 are incompatible with pandas >= 2
        reqs.append("pandas<2")
    return reqs


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
    code_paths=None,
    dfs_tmpdir=None,
    sample_input=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
):
    """
    Log a Spark MLlib model as an MLflow artifact for the current run. This uses the
    MLlib persistence format and produces an MLflow Model with the Spark flavor.

    Note: If no run is active, it will instantiate a run to obtain a run_id.

    :param spark_model: Spark model to be saved - MLflow can only save descendants of
                        pyspark.ml.Model or pyspark.ml.Transformer which implement
                        MLReadable and MLWritable.
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
                                'python=3.8.15',
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
                        predictions = ...  # compute model predictions
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
    :param metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.

                     .. Note:: Experimental: This parameter may change or be removed in a future
                                             release without warning.
    :return: A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
             metadata of the logged model.

    .. code-block:: python
        :caption: Example

        from pyspark.ml import Pipeline
        from pyspark.ml.classification import LogisticRegression
        from pyspark.ml.feature import HashingTF, Tokenizer

        training = spark.createDataFrame(
            [
                (0, "a b c d e spark", 1.0),
                (1, "b d", 0.0),
                (2, "spark f g h", 1.0),
                (3, "hadoop mapreduce", 0.0),
            ],
            ["id", "text", "label"],
        )
        tokenizer = Tokenizer(inputCol="text", outputCol="words")
        hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
        lr = LogisticRegression(maxIter=10, regParam=0.001)
        pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])
        model = pipeline.fit(training)
        mlflow.spark.log_model(model, "spark-model")
    """

    _validate_model(spark_model)
    from pyspark.ml import PipelineModel

    if not isinstance(spark_model, PipelineModel):
        spark_model = PipelineModel([spark_model])
    run_id = mlflow.tracking.fluent._get_or_start_run().info.run_id
    run_root_artifact_uri = mlflow.get_artifact_uri()
    remote_model_path = None
    if _should_use_mlflowdbfs(run_root_artifact_uri):
        remote_model_path = append_to_uri_path(
            run_root_artifact_uri, artifact_path, _SPARK_MODEL_PATH_SUB
        )
        mlflowdbfs_path = _mlflowdbfs_path(run_id, artifact_path)
        with databricks_utils.MlflowCredentialContext(
            get_databricks_profile_uri_from_artifact_uri(run_root_artifact_uri)
        ):
            try:
                spark_model.save(mlflowdbfs_path)
            except Exception as e:
                raise MlflowException("failed to save spark model via mlflowdbfs") from e

    # If the artifact URI is a local filesystem path, defer to Model.log() to persist the model,
    # since Spark may not be able to write directly to the driver's filesystem. For example,
    # writing to `file:/uri` will write to the local filesystem from each executor, which will
    # be incorrect on multi-node clusters.
    # If the artifact URI is not a local filesystem path we attempt to write directly to the
    # artifact repo via Spark. If this fails, we defer to Model.log().
    elif is_local_uri(run_root_artifact_uri) or not _maybe_save_model(
        spark_model,
        append_to_uri_path(run_root_artifact_uri, artifact_path),
    ):
        return Model.log(
            artifact_path=artifact_path,
            flavor=mlflow.spark,
            spark_model=spark_model,
            conda_env=conda_env,
            code_paths=code_paths,
            dfs_tmpdir=dfs_tmpdir,
            sample_input=sample_input,
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=input_example,
            await_registration_for=await_registration_for,
            pip_requirements=pip_requirements,
            extra_pip_requirements=extra_pip_requirements,
            metadata=metadata,
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
            code_paths,
            signature=signature,
            input_example=input_example,
            pip_requirements=pip_requirements,
            extra_pip_requirements=extra_pip_requirements,
            remote_model_path=remote_model_path,
        )
        mlflow.tracking.fluent.log_artifacts(tmp_model_metadata_dir, artifact_path)
        mlflow.tracking.fluent._record_logged_model(mlflow_model)
        if registered_model_name is not None:
            mlflow.register_model(
                f"runs:/{run_id}/{artifact_path}",
                registered_model_name,
                await_registration_for,
            )
        return mlflow_model.get_model_info()


def _mlflowdbfs_path(run_id, artifact_path):
    if artifact_path.startswith("/"):
        raise MlflowException(
            f"artifact_path should be relative, found: {artifact_path}",
            INVALID_PARAMETER_VALUE,
        )
    return "{}:///artifacts?run_id={}&path=/{}".format(
        _MLFLOWDBFS_SCHEME, run_id, posixpath.join(artifact_path, _SPARK_MODEL_PATH_SUB)
    )


def _maybe_save_model(spark_model, model_dir):
    from py4j.protocol import Py4JError

    try:
        spark_model.save(posixpath.join(model_dir, _SPARK_MODEL_PATH_SUB))
        return True
    except Py4JError:
        return False


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
    def _stats(cls):
        return cls._jvm().org.apache.hadoop.fs.FileSystem.getGlobalStorageStatistics()

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
                "Unexpected exception while checking if model uri is visible on DFS: %s", ex
            )
        return False

    @classmethod
    def maybe_copy_from_uri(cls, src_uri, dst_path, local_model_path=None):
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
        return cls.maybe_copy_from_local_file(
            local_model_path or _download_artifact_from_uri(src_uri), dst_path
        )

    @classmethod
    def delete(cls, path):
        cls._fs().delete(cls._remote_path(path), True)

    @classmethod
    def is_filesystem_available(cls, scheme):
        return scheme in [stats.getScheme() for stats in cls._stats().iterator()]


def _should_use_mlflowdbfs(root_uri):
    # The `mlflowdbfs` scheme does not appear in the available schemes returned from
    # the Hadoop FileSystem API until a read call has been issued.
    from mlflow.utils._spark_utils import _get_active_spark_session

    if (
        not is_valid_dbfs_uri(root_uri)
        or not is_databricks_acled_artifacts_uri(root_uri)
        or not databricks_utils.is_in_databricks_runtime()
        or (environment_variables._DISABLE_MLFLOWDBFS.get() or "").lower() == "true"
    ):
        return False

    try:
        databricks_utils._get_dbutils()
    except Exception:
        # If dbutils is unavailable, indicate that mlflowdbfs is unavailable
        # because usage of mlflowdbfs depends on dbutils
        return False

    mlflowdbfs_read_exception_str = None
    try:
        _get_active_spark_session().read.load("mlflowdbfs:///artifact?run_id=foo&path=/bar")
    except Exception as e:
        # The load invocation is expected to throw an exception.
        mlflowdbfs_read_exception_str = str(e)

    try:
        return _HadoopFileSystem.is_filesystem_available(_MLFLOWDBFS_SCHEME)
    except Exception:
        # The HDFS filesystem logic used to determine mlflowdbfs availability on Databricks
        # clusters may not work on certain Databricks cluster types due to unavailability of
        # the _HadoopFileSystem.is_filesystem_available() API. As a temporary workaround,
        # we check the contents of the expected exception raised by a dummy mlflowdbfs
        # read for evidence that mlflowdbfs is available. If "MlflowdbfsClient" is present
        # in the exception contents, we can safely assume that mlflowdbfs is available because
        # `MlflowdbfsClient` is exclusively used by mlflowdbfs for performing MLflow
        # file storage operations
        #
        # TODO: Remove this logic once the _HadoopFileSystem.is_filesystem_available() check
        # below is determined to work on all Databricks cluster types
        return "MlflowdbfsClient" in (mlflowdbfs_read_exception_str or "")


def _save_model_metadata(
    dst_dir,
    spark_model,
    mlflow_model,
    sample_input,
    conda_env,
    code_paths,
    signature=None,
    input_example=None,
    pip_requirements=None,
    extra_pip_requirements=None,
    remote_model_path=None,
):
    """
    Saves model metadata into the passed-in directory.
    If mlflowdbfs is not used, the persisted metadata assumes that a model can be
    loaded from a relative path to the metadata file (currently hard-coded to "sparkml").
    If mlflowdbfs is used, remote_model_path should be provided, and the model needs to
    be loaded from the remote_model_path.
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

    code_dir_subpath = _validate_and_copy_code_paths(code_paths, dst_dir)
    mlflow_model.add_flavor(
        FLAVOR_NAME,
        pyspark_version=pyspark.__version__,
        model_data=_SPARK_MODEL_PATH_SUB,
        code=code_dir_subpath,
    )
    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.spark",
        data=_SPARK_MODEL_PATH_SUB,
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
    )
    mlflow_model.save(os.path.join(dst_dir, MLMODEL_FILE_NAME))

    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements()
            if remote_model_path:
                _logger.info(
                    "Inferring pip requirements by reloading the logged model from the databricks "
                    "artifact repository, which can be time-consuming. To speed up, explicitly "
                    "specify the conda_env or pip_requirements when calling log_model()."
                )
            # To ensure `_load_pyfunc` can successfully load the model during the dependency
            # inference, `mlflow_model.save` must be called beforehand to save an MLmodel file.
            inferred_reqs = mlflow.models.infer_pip_requirements(
                remote_model_path or dst_dir,
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

    _PythonEnv.current().to_yaml(os.path.join(dst_dir, _PYTHON_ENV_FILE_NAME))


def _validate_model(spark_model):
    from pyspark.ml.util import MLReadable, MLWritable
    from pyspark.ml import Model as PySparkModel
    from pyspark.ml import Transformer as PySparkTransformer

    if (
        (
            not isinstance(spark_model, PySparkModel)
            and not isinstance(spark_model, PySparkTransformer)
        )
        or not isinstance(spark_model, MLReadable)
        or not isinstance(spark_model, MLWritable)
    ):
        raise MlflowException(
            "Cannot serialize this model. MLflow can only save descendants of pyspark.ml.Model "
            "or pyspark.ml.Transformer that implement MLWritable and MLReadable.",
            INVALID_PARAMETER_VALUE,
        )


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name="pyspark"))
def save_model(
    spark_model,
    path,
    mlflow_model=None,
    conda_env=None,
    code_paths=None,
    dfs_tmpdir=None,
    sample_input=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
):
    """
    Save a Spark MLlib Model to a local path.

    By default, this function saves models using the Spark MLlib persistence mechanism.
    Additionally, if a sample input is specified using the ``sample_input`` parameter, the model
    is also serialized in MLeap format and the MLeap flavor is added.

    :param spark_model: Spark model to be saved - MLflow can only save descendants of
                        pyspark.ml.Model or pyspark.ml.Transformer which implement
                        MLReadable and MLWritable.
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
                                'python=3.8.15',
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
                        predictions = ...  # compute model predictions
                        signature = infer_signature(train, predictions)
    :param input_example: Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    :param metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.

                     .. Note:: Experimental: This parameter may change or be removed in a future
                                             release without warning.

    .. code-block:: python
        :caption: Example

        from mlflow import spark
        from pyspark.ml.pipeline import PipelineModel

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
    if metadata is not None:
        mlflow_model.metadata = metadata
    # Spark ML stores the model on DFS if running on a cluster
    # Save it to a DFS temp dir first and copy it to local path
    if dfs_tmpdir is None:
        dfs_tmpdir = MLFLOW_DFS_TMP.get()
    tmp_path = generate_tmp_dfs_path(dfs_tmpdir)
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
        code_paths=code_paths,
        signature=signature,
        input_example=input_example,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
    )


def _load_model_databricks(dfs_tmpdir, local_model_path):
    from pyspark.ml.pipeline import PipelineModel

    # Spark ML expects the model to be stored on DFS
    # Copy the model to a temp DFS location first. We cannot delete this file, as
    # Spark may read from it at any point.
    fuse_dfs_tmpdir = dbfs_hdfs_uri_to_fuse_path(dfs_tmpdir)
    os.makedirs(fuse_dfs_tmpdir)
    # Workaround for inability to use shutil.copytree with DBFS FUSE due to permission-denied
    # errors on passthrough-enabled clusters when attempting to copy permission bits for directories
    shutil_copytree_without_file_permissions(src_dir=local_model_path, dst_dir=fuse_dfs_tmpdir)
    return PipelineModel.load(dfs_tmpdir)


def _load_model(model_uri, dfs_tmpdir_base=None, local_model_path=None):
    from pyspark.ml.pipeline import PipelineModel

    dfs_tmpdir = generate_tmp_dfs_path(dfs_tmpdir_base or MLFLOW_DFS_TMP.get())
    if databricks_utils.is_in_cluster() and databricks_utils.is_dbfs_fuse_available():
        return _load_model_databricks(
            dfs_tmpdir, local_model_path or _download_artifact_from_uri(model_uri)
        )
    model_uri = _HadoopFileSystem.maybe_copy_from_uri(model_uri, dfs_tmpdir, local_model_path)
    return PipelineModel.load(model_uri)


def load_model(model_uri, dfs_tmpdir=None, dst_path=None):
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
    :param dst_path: The local filesystem path to which to download the model artifact.
                     This directory must already exist. If unspecified, a local output
                     path will be created.
    :return: pyspark.ml.pipeline.PipelineModel

    .. code-block:: python
        :caption: Example

        from mlflow import spark

        model = mlflow.spark.load_model("spark-model")
        # Prepare test documents, which are unlabeled (id, text) tuples.
        test = spark.createDataFrame(
            [(4, "spark i j k"), (5, "l m n"), (6, "spark hadoop spark"), (7, "apache hadoop")],
            ["id", "text"],
        )
        # Make predictions on test documents
        prediction = model.transform(test)
    """
    # This MUST be called prior to appending the model flavor to `model_uri` in order
    # for `artifact_path` to take on the correct value for model loading via mlflowdbfs.
    root_uri, artifact_path = _get_root_uri_and_artifact_path(model_uri)

    flavor_conf = _get_flavor_configuration_from_uri(model_uri, FLAVOR_NAME, _logger)
    local_mlflow_model_path = _download_artifact_from_uri(
        artifact_uri=model_uri, output_path=dst_path
    )
    _add_code_from_conf_to_system_path(local_mlflow_model_path, flavor_conf)

    if _should_use_mlflowdbfs(model_uri):
        from pyspark.ml.pipeline import PipelineModel

        mlflowdbfs_path = _mlflowdbfs_path(
            DatabricksArtifactRepository._extract_run_id(model_uri), artifact_path
        )
        with databricks_utils.MlflowCredentialContext(
            get_databricks_profile_uri_from_artifact_uri(root_uri)
        ):
            return PipelineModel.load(mlflowdbfs_path)

    sparkml_model_uri = append_to_uri_path(model_uri, flavor_conf["model_data"])
    local_sparkml_model_path = os.path.join(local_mlflow_model_path, flavor_conf["model_data"])
    return _load_model(
        model_uri=sparkml_model_uri,
        dfs_tmpdir_base=dfs_tmpdir,
        local_model_path=local_sparkml_model_path,
    )


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_model``.

    :param path: Local filesystem path to the MLflow Model with the ``spark`` flavor.
    """
    from mlflow.utils._spark_utils import (
        _create_local_spark_session_for_loading_spark_model,
        _get_active_spark_session,
    )

    # NOTE: The `_create_local_spark_session_for_loading_spark_model()` call below may change
    # settings of the active session which we do not intend to do here.
    # In particular, setting master to local[1] can break distributed clusters.
    # To avoid this problem, we explicitly check for an active session. This is not ideal but there
    # is no good workaround at the moment.
    spark = _get_active_spark_session()
    if spark is None:
        # NB: If there is no existing Spark context, create a new local one.
        # NB: We're disabling caching on the new context since we do not need it and we want to
        # avoid overwriting cache of underlying Spark cluster when executed on a Spark Worker
        # (e.g. as part of spark_udf).
        spark = _create_local_spark_session_for_loading_spark_model()
    return _PyFuncModelWrapper(spark, _load_model(model_uri=path))


def _find_and_set_features_col_as_vector_if_needed(spark_df, spark_model):
    """
    Finds the `featuresCol` column in spark_model and
    then tries to cast that column to `vector` type.
    This method is noop if the `featuresCol` is already of type `vector`
    or if it can't be cast to `vector` type
    Note:
    If a spark ML pipeline contains a single Estimator stage, it requires
    the input dataframe to contain features column of vector type.
    But the autologging for pyspark ML casts vector column to array<double> type
    for parity with the pd Dataframe. The following fix is required, which transforms
    that features column back to vector type so that the pipeline stages can correctly work.
    A valid scenario is if the auto-logged input example is directly used
    for prediction, which would otherwise fail without this transformation.

    :param spark_df: Input dataframe that contains `featuresCol`
    :param spark_model: A pipeline model or a single transformer that contains `featuresCol` param
    :return: A spark dataframe that contains features column of `vector` type.
    """
    from pyspark.sql.functions import udf
    from pyspark.ml.linalg import Vectors, VectorUDT
    from pyspark.sql import types as t

    def _find_stage_with_features_col(stage):
        if stage.hasParam("featuresCol"):

            def _array_to_vector(input_array):
                return Vectors.dense(input_array)

            array_to_vector_udf = udf(f=_array_to_vector, returnType=VectorUDT())
            features_col_name = stage.extractParamMap().get(stage.featuresCol)
            features_col_type = [
                _field
                for _field in spark_df.schema.fields
                if _field.name == features_col_name
                and _field.dataType
                in [t.ArrayType(t.DoubleType(), True), t.ArrayType(t.DoubleType(), False)]
            ]
            if len(features_col_type) == 1:
                return spark_df.withColumn(
                    features_col_name, array_to_vector_udf(features_col_name)
                )
        return spark_df

    if hasattr(spark_model, "stages"):
        for stage in reversed(spark_model.stages):
            return _find_stage_with_features_col(stage)
    return _find_stage_with_features_col(spark_model)


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
        from pyspark.ml import PipelineModel

        spark_df = _find_and_set_features_col_as_vector_if_needed(
            self.spark.createDataFrame(pandas_df), self.spark_model
        )
        prediction_column = "prediction"
        if isinstance(self.spark_model, PipelineModel) and self.spark_model.stages[-1].hasParam(
            "outputCol"
        ):
            from pyspark.sql import SparkSession

            spark = SparkSession.builder.getOrCreate()
            # do a transform with an empty input DataFrame
            # to get the schema of the transformed DataFrame
            transformed_df = self.spark_model.transform(spark.createDataFrame([], spark_df.schema))
            # Ensure prediction column doesn't already exist
            if prediction_column not in transformed_df.columns:
                # make sure predict work by default for Transformers
                self.spark_model.stages[-1].setOutputCol(prediction_column)
        return [
            x.prediction
            for x in self.spark_model.transform(spark_df).select(prediction_column).collect()
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
        spark = (
            SparkSession.builder.config("spark.jars.packages", "org.mlflow:mlflow-spark:1.11.0")
            .master("local[*]")
            .getOrCreate()
        )
        df = spark.createDataFrame(
            [(4, "spark i j k"), (5, "l m n"), (6, "spark hadoop spark"), (7, "apache hadoop")],
            ["id", "text"],
        )
        import tempfile

        tempdir = tempfile.mkdtemp()
        df.write.csv(os.path.join(tempdir, "my-data-path"), header=True)
        # Enable Spark datasource autologging.
        mlflow.spark.autolog()
        loaded_df = spark.read.csv(
            os.path.join(tempdir, "my-data-path"), header=True, inferSchema=True
        )
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
