"""
The ``mlflow.johnsnowlabs`` module provides an API for logging and loading Spark NLP and NLU models. This module
exports the following flavors:

The following Environment Variables must be present
- `SECRET`: The secret for the John Snow Labs Enterprise NLP Library
- `SPARK_NLP_LICENSE`: Your John Snow Labs Enterprise NLP License
- `AWS_ACCESS_KEY_ID`: Your AWS Secret ID for accessing John Snow Labs Enterprise Models
- `AWS_SECRET_ACCESS_KEY`: Your AWS Secret key for accessing John Snow Labs Enterprise Models


You can set them like this in Python

```python
import os
os.environ['SECRET'] = 'MY_SECRET'
os.environ['AWS_ACCESS_KEY_ID'] = 'MY_AWS_ACCESS_KEY_ID'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'MY_AWS_SECRET_ACCESS_KEY'
os.environ['SPARK_NLP_LICENSE'] = 'MY_SPARK_NLP_LICENSE'


Johnsnowlabs (native) format
    Allows models to be loaded as Spark Transformers for scoring in a Spark session.
    Models with this flavor can be loaded as NluPipelines, with underlying Spark MLlib PipelineModel
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
import json
import logging
import os
import posixpath
import shutil
import sys
from pathlib import Path
from urllib.request import urlopen

import johnsnowlabs.settings
import yaml
from johnsnowlabs import nlp
from johnsnowlabs import settings
from johnsnowlabs.auto_install.jsl_home import get_install_suite_from_jsl_home
from johnsnowlabs.py_models.jsl_secrets import JslSecrets

import mlflow
from mlflow import pyfunc, mleap
from mlflow.environment_variables import MLFLOW_DFS_TMP
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import ModelInputExample, _save_example
from mlflow.spark import _mlflowdbfs_path, _should_use_mlflowdbfs, _maybe_save_model, _HadoopFileSystem
from mlflow.store.artifact.databricks_artifact_repo import DatabricksArtifactRepository
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import (
    _download_artifact_from_uri,
    _get_root_uri_and_artifact_path,
)
from mlflow.utils import databricks_utils
from mlflow.utils.autologging_utils import autologging_integration, safe_patch
from mlflow.utils.docstring_utils import format_docstring, LOG_MODEL_PARAM_DOCS
from mlflow.utils.environment import (
    _mlflow_conda_env,
    _process_pip_requirements,
    _process_conda_env,
    _CONDA_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _PythonEnv,
)
from mlflow.utils.file_utils import TempDir, write_to, shutil_copytree_without_file_permissions
from mlflow.utils.model_utils import (
    _get_flavor_configuration_from_uri,
    _validate_and_copy_code_paths,
    _add_code_from_conf_to_system_path,
)
from mlflow.utils.uri import (
    is_local_uri,
    append_to_uri_path,
    dbfs_hdfs_uri_to_fuse_path,
    is_valid_dbfs_uri,
    get_databricks_profile_uri_from_artifact_uri,
    generate_tmp_dfs_path,
)

def validate_env_vars():
    if not all(os.environ.get(var) for var in JOHNSNOWLABS_ENV_VARS):
        raise Exception(
            'Please set SECRET environment variable to your secret key before saving or logging your model')

FLAVOR_NAME = "johnsnowlabs"
JOHNSNOWLABS_ENV_VARS = ('SECRET', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'SPARK_NLP_LICENSE')


def download_url(url, save_path):
    file_name = url.split("/")[-1]
    save_path = os.path.join(save_path, file_name)
    print(f"Downloading {file_name}")
    # Download the file from `url` and save it locally under `file_name`:
    with urlopen(url) as response, open(save_path, "wb") as out_file:
        shutil.copyfileobj(response, out_file)


def download_sparknlp_jar(out_folder):
    SPARK_NLP_JAR = f'spark-nlp-assembly-{settings.raw_version_nlp}.jar'
    download_url(
        f"https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/{SPARK_NLP_JAR}",
        out_folder)


def download_sparknlp_jsl_jar(out_folder, secret):
    SPARK_NLP_INTERNAL_JAR = f'spark-nlp-jsl-{settings.raw_version_medical}.jar'
    download_url(
        f"https://pypi.johnsnowlabs.com/{secret}/{SPARK_NLP_INTERNAL_JAR}",
        out_folder)


HEALTHCARE_wheel_URI = 'https://pypi.johnsnowlabs.com/{secret}/spark-nlp-jsl/spark_nlp_jsl-{SPARK_NLP_INTERNAL_VERSION}-py3-none-any.whl'
_JOHNSNOWLABS_MODEL_PATH_SUB = "jsl-model"
_MLFLOWDBFS_SCHEME = "mlflowdbfs"

_logger = logging.getLogger(__name__)


def get_default_pip_requirements():
    """
    :return: A list of default pip requirements for MLflow Models produced by this flavor.
             Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
             that, at minimum, contains these requirements.
    """
    if not 'SECRET' in os.environ:
        raise Exception(
            'Please set SECRET environment variable to your secret key before saving or logging your model')
    from johnsnowlabs import settings
    return [
        f'johnsnowlabs=={settings.raw_version_jsl_lib}',
        f"https://pypi.johnsnowlabs.com/{os.environ['SECRET']}/spark-nlp-jsl/spark_nlp_jsl-{settings.raw_version_medical}-py3-none-any.whl",
        f'pyspark=={settings.raw_version_pyspark}',
        'mlflow-tmp==2.2.26', # TODO REMOVE THIS after PR for johnsnowlabs flavor is merged
        'pandas==1.5.3',
        # 'mleap',  # TODO JUST ADD TO TEST CLASS?
        # 'pytest',  # TODO JUST ADD TO TEST CLASS?

    ]


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


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name="johnsnowlabs"))
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
        ### Secret Params

):
    """
    Log a Johnsnowlabs NLUPipeline, created via nlp.load() model as an MLflow artifact for the current run. This uses the
    MLlib persistence format and produces an MLflow Model with the Johnsnowlabs flavor.

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
    validate_env_vars()
    run_id = mlflow.tracking.fluent._get_or_start_run().info.run_id
    run_root_artifact_uri = mlflow.get_artifact_uri()
    remote_model_path = None
    if _should_use_mlflowdbfs(run_root_artifact_uri):
        remote_model_path = append_to_uri_path(
            run_root_artifact_uri, artifact_path, _JOHNSNOWLABS_MODEL_PATH_SUB
        )
        mlflowdbfs_path = _mlflowdbfs_path(run_id, artifact_path)
        with databricks_utils.MlflowCredentialContext(
                get_databricks_profile_uri_from_artifact_uri(run_root_artifact_uri)
        ):
            try:
                _unpack_and_save_model(spark_model, mlflowdbfs_path)

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
            flavor=mlflow.johnsnowlabs,
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

    # add the johnsnowlabs flavor
    # TODO dynamic pyspark version # pyspark.__version__p
    mlflow_model.add_flavor(
        FLAVOR_NAME,
        pyspark_version=johnsnowlabs.settings.raw_version_pyspark,
        model_data=_JOHNSNOWLABS_MODEL_PATH_SUB,
        code=code_dir_subpath,
    )

    # add the pyfunc flavor
    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.johnsnowlabs",
        data=_JOHNSNOWLABS_MODEL_PATH_SUB,
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
    )
    mlflow_model.save(os.path.join(dst_dir, MLMODEL_FILE_NAME))

    if conda_env is None:
        default_reqs = get_default_pip_requirements() if pip_requirements is None else None
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
    # TODO REMOVE THIS HACK PR IS MERGED
    pip_requirements = [r for r in pip_requirements if not r.startswith('mlflow==2.2')]
    write_to(os.path.join(dst_dir, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))

    _PythonEnv.current().to_yaml(os.path.join(dst_dir, _PYTHON_ENV_FILE_NAME))

    save_jars_and_lic(dst_dir)


def save_jars_and_lic(dst_dir):
    deps_data_path = os.path.join(dst_dir, _JOHNSNOWLABS_MODEL_PATH_SUB, "jars.jsl")
    Path(deps_data_path).mkdir(parents=True, exist_ok=True)

    import shutil
    suite = get_install_suite_from_jsl_home(False)
    if suite.hc.get_java_path():
        shutil.copyfile(suite.hc.get_java_path(), os.path.join(deps_data_path, 'hc_jar.jar'))
    if suite.nlp.get_java_path():
        shutil.copyfile(suite.nlp.get_java_path(), os.path.join(deps_data_path, 'os_jar.jar'))

    secrets = JslSecrets.build_or_try_find_secrets()
    if secrets.HC_LICENSE:
        with open(os.path.join(deps_data_path, 'license.json'), 'w') as f:
            f.write(secrets.json())


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name="johnsnowlabs"))
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
    Save a Spark johnsnowlabs Model to a local path.

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
    # _validate_model(spark_model)
    # _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)
    validate_env_vars()
    if mlflow_model is None:
        mlflow_model = Model()
    if metadata is not None:
        mlflow_model.metadata = metadata
    # Spark ML stores the model on DFS if running on a cluster
    # Save it to a DFS temp dir first and copy it to local path
    if dfs_tmpdir is None:
        dfs_tmpdir = MLFLOW_DFS_TMP.get()
    tmp_path = generate_tmp_dfs_path(dfs_tmpdir)

    # Spark Session already is running if we have a model, so no need to check or create
    _unpack_and_save_model(spark_model, tmp_path)
    sparkml_data_path = os.path.abspath(os.path.join(path, _JOHNSNOWLABS_MODEL_PATH_SUB))
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
    # Spark ML expects the model to be stored on DFS
    # Copy the model to a temp DFS location first. We cannot delete this file, as
    # Spark may read from it at any point.
    fuse_dfs_tmpdir = dbfs_hdfs_uri_to_fuse_path(dfs_tmpdir)
    os.makedirs(fuse_dfs_tmpdir)
    # Workaround for inability to use shutil.copytree with DBFS FUSE due to permission-denied
    # errors on passthrough-enabled clusters when attempting to copy permission bits for directories
    shutil_copytree_without_file_permissions(src_dir=local_model_path, dst_dir=fuse_dfs_tmpdir)
    return nlp.load(path=dfs_tmpdir)


def _load_model(model_uri, dfs_tmpdir_base=None, local_model_path=None):
    dfs_tmpdir = generate_tmp_dfs_path(dfs_tmpdir_base or MLFLOW_DFS_TMP.get())
    if databricks_utils.is_in_cluster() and databricks_utils.is_dbfs_fuse_available():
        return _load_model_databricks(
            dfs_tmpdir, local_model_path or _download_artifact_from_uri(model_uri)
        )
    # model_uri = _HadoopFileSystem.maybe_copy_from_uri(model_uri, dfs_tmpdir, local_model_path)
    if model_uri and not local_model_path:
        local_model_path = _download_artifact_from_uri(model_uri)
    get_or_create_sparksession(local_model_path)

    if _JOHNSNOWLABS_MODEL_PATH_SUB not in local_model_path:
        local_model_path = os.path.join(local_model_path, _JOHNSNOWLABS_MODEL_PATH_SUB)

    return nlp.load(path=local_model_path)


def load_model(model_uri, dfs_tmpdir=None, dst_path=None, **kwargs):
    """
    Load the Spark MLlib model from the path.

    :param model_uri: The location, in URI format, of the MLflow model, for example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
                      - ``models:/<model_name>/<model_version>``
                      - ``models:/<model_name>/<stage>``
                      - johnsnowlabs:/<nlu_ref>``

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
    validate_env_vars()
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


def _load_pyfunc(path, spark=None):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_model``.
    :param path: Local filesystem path to the MLflow Model with the ``johnsnowlabs`` flavor.
    :param spark: Optionally pass spark context when using pyfunc as UDF. required, because we cannot fetch the Sparkcontext inside of the Workernode which executes the UDF
    :return:
    """
    return _PyFuncModelWrapper(_load_model(model_uri=path), spark if spark else get_or_create_sparksession(path))


def get_or_create_sparksession(model_path=None):
    """
    1. Check if SparkSession running and get it
    2. If none exists, create a new one using jars in model_path
    3. If model_path not defined, rely on nlp.start() to create a new
    one using johnsnowlabs Jar resolution method
    See https://nlp.johnsnowlabs.com/docs/en/jsl/start-a-sparksession
    and https://nlp.johnsnowlabs.com/docs/en/jsl/install_advanced
    :param model_path:
    :return:
    """
    validate_env_vars()
    from mlflow.utils._spark_utils import _get_active_spark_session
    spark = _get_active_spark_session()
    if spark is None:
        spark_conf = {}
        spark_conf["spark.python.worker.reuse"] = "true"
        # disable task retry (i.e. make it fast fail)
        spark_conf["spark.task.maxFailures"] = "1"
        # Disable simplifying traceback from Python UDFs
        spark_conf["spark.sql.execution.pyspark.udf.simplifiedTraceback.enabled"] = "false"
        # Show jvm side stack trace.
        spark_conf["spark.sql.pyspark.jvmStacktrace.enabled"] = "true"
        spark_conf["spark.jars.excludes"] = "net.sourceforge.f2j:arpack_combined_all"
        spark_conf["spark.driver.extraJavaOptions"] = "-Dio.netty.tryReflectionSetAccessible=true"
        spark_conf["spark.executor.extraJavaOptions"] = "-Dio.netty.tryReflectionSetAccessible=true"

        os.environ['PYSPARK_PYTHON'] = sys.executable
        os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
        if model_path:
            jar_paths, license_path = fetch_deps_from_path(os.path.join(model_path))
            # jar_paths += get_mleap_jars().split(',')  # TODO when to load MLleap Jars
            if license_path:
                if 'JSL_NLP_LICENSE' not in os.environ:
                    os.environ.update({k: str(v) for k, v in json.load(open(license_path)).items() if v is not None})
                    os.environ['JSL_NLP_LICENSE'] = json.load(open(license_path))['HC_LICENSE']
            print("STARTING NEW  SESSION WITH JARS: ", jar_paths)
            spark = nlp.start(nlp=False, spark_nlp=False, jar_paths=jar_paths, json_license_path=license_path,
                              create_jsl_home_if_missing=False, spark_conf=spark_conf)
        else:
            spark = nlp.start()
    return spark


def create_sparksession_for_jsl_uri(**kwargs):
    # if using johnsnowlabs:/ uri, we need to start a sparksession
    # Support johnsnowlabs nlp.start() API to fetch a session
    nlp.start(**kwargs)


def fetch_deps_from_path(local_model_path):
    if _JOHNSNOWLABS_MODEL_PATH_SUB not in local_model_path:
        local_model_path = os.path.join(local_model_path, _JOHNSNOWLABS_MODEL_PATH_SUB, 'jars.jsl', )
    else:
        local_model_path = os.path.join(local_model_path, 'jars.jsl', )

    jar_paths = [os.path.join(local_model_path, file) for file in os.listdir(local_model_path) if '.jar' in file]
    license_path = [os.path.join(local_model_path, file) for file in os.listdir(local_model_path) if '.json' in file]
    license_path = license_path[0] if license_path else None
    return jar_paths, license_path


def _unpack_and_save_model(spark_model, dst):
    from pyspark.ml import PipelineModel
    if isinstance(spark_model, _PyFuncModelWrapper):
        spark_model = spark_model.spark_model
    if isinstance(spark_model, PipelineModel):
        spark_model.write().overwrite().save(dst)
    else:
        # nlu pipe
        spark_model.predict('Init')
        try:
            spark_model.vanilla_transformer_pipe.write().overwrite().save(dst)
        except:
            # for mlflowdbfs_path we cannot use overwrite, gives
            # org.apache.hadoop.fs.UnsupportedFileSystemException: No FileSystem for scheme "mlflowdbfs"
            spark_model.save(dst)


class _PyFuncModelWrapper:
    """
    Wrapper around NLUPipeline providing interface for scoring pandas DataFrame.
    """

    def __init__(self, spark_model, spark=None, ):
        # we have this ternary, so we support _PyFuncModelWrapper(nlu_ref)
        self.spark = spark if spark else get_or_create_sparksession()
        self.spark_model = spark_model

    def predict(self, text, output_level=''):
        """
        Generate predictions given input data in a pandas DataFrame.

        :param output_level:
        :param text: pandas DataFrame containing input data.
        :return: List with model predictions.
        """
        return self.spark_model.predict(text, output_level=output_level).reset_index().to_json()


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
    Johnsnowlabs models is not currently supported via this API. Datasource autologging is
    best-effort, meaning that if Spark is under heavy load or MLflow logging fails for any reason
    (e.g., if the MLflow server is unavailable), logging may be dropped.

    For any unexpected issues with autologging, check Spark driver and executor logs in addition
    to stderr & stdout generated from your MLflow code - datasource information is pulled from
    Spark, so logs relevant to debugging may show up amongst the Spark logs.

    .. code-block:: python
        :caption: Example

        import mlflow.johnsnowlabs
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
