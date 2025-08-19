"""
The ``mlflow.johnsnowlabs`` module provides an API for logging and loading Spark NLP and NLU models.
This module exports the following flavors:

Johnsnowlabs (native) format
    Allows models to be loaded as Spark Transformers for scoring in a Spark session.
    Models with this flavor can be loaded as NluPipelines, with underlying Spark MLlib PipelineModel
    This is the main flavor and is always produced.
:py:mod:`mlflow.pyfunc`
    Supports deployment outside of Spark by instantiating a SparkContext and reading
    input data as a Spark DataFrame prior to scoring. Also supports deployment in Spark
    as a Spark UDF. Models with this flavor can be loaded as Python functions
    for performing inference. This flavor is always produced.

This flavor gives you access to `20.000+ state-of-the-art enterprise NLP models in 200+ languages
<https://nlp.johnsnowlabs.com/models>`_ for medical, finance, legal and many more domains.
Features include: LLM's, Text Summarization, Question Answering, Named Entity Recognition, Relation
Extraction, Sentiment Analysis, Spell Checking, Image Classification, Automatic Speech Recognition
and much more, powered by the latest Transformer Architectures. The models are provided by
`John Snow Labs <https://www.johnsnowlabs.com/>`_ and requires a `John Snow Labs
<https://www.johnsnowlabs.com/>`_ Enterprise NLP License. `You can reach out to us
<https://www.johnsnowlabs.com/schedule-a-demo/>`_ for a research or industry license.

These keys must be present in your license json:

1. ``SECRET``: The secret for the John Snow Labs Enterprise NLP Library
2. ``SPARK_NLP_LICENSE``: Your John Snow Labs Enterprise NLP License
3. ``AWS_ACCESS_KEY_ID``: Your AWS Secret ID for accessing John Snow Labs Enterprise Models
4. ``AWS_SECRET_ACCESS_KEY``: Your AWS Secret key for accessing John Snow Labs Enterprise Models

You can set them using the following code:

.. code-block:: python

    import os
    import json

    # Write your raw license.json string into the 'JOHNSNOWLABS_LICENSE_JSON' env variable
    creds = {
        "AWS_ACCESS_KEY_ID": "...",
        "AWS_SECRET_ACCESS_KEY": "...",
        "SPARK_NLP_LICENSE": "...",
        "SECRET": "...",
    }
    os.environ["JOHNSNOWLABS_LICENSE_JSON"] = json.dumps(creds)
"""

import json
import logging
import os
import posixpath
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

import mlflow
from mlflow import pyfunc
from mlflow.environment_variables import MLFLOW_DFS_TMP
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import ModelInputExample, _save_example
from mlflow.spark import (
    _HadoopFileSystem,
    _maybe_save_model,
    _mlflowdbfs_path,
    _should_use_mlflowdbfs,
)
from mlflow.store.artifact.databricks_artifact_repo import DatabricksArtifactRepository
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import (
    _download_artifact_from_uri,
    _get_root_uri_and_artifact_path,
)
from mlflow.tracking.fluent import _initialize_logged_model
from mlflow.utils import databricks_utils
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
    _CONDA_ENV_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _mlflow_conda_env,
    _process_conda_env,
    _process_pip_requirements,
    _PythonEnv,
)
from mlflow.utils.file_utils import (
    TempDir,
    get_total_file_size,
    shutil_copytree_without_file_permissions,
    write_to,
)
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path,
    _get_flavor_configuration_from_uri,
    _validate_and_copy_code_paths,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement
from mlflow.utils.uri import (
    append_to_uri_path,
    dbfs_hdfs_uri_to_fuse_path,
    generate_tmp_dfs_path,
    get_databricks_profile_uri_from_artifact_uri,
    is_local_uri,
    is_valid_dbfs_uri,
)

FLAVOR_NAME = "johnsnowlabs"
_JOHNSNOWLABS_ENV_JSON_LICENSE_KEY = "JOHNSNOWLABS_LICENSE_JSON"
_JOHNSNOWLABS_ENV_HEALTHCARE_SECRET = "HEALTHCARE_SECRET"
_JOHNSNOWLABS_ENV_VISUAL_SECRET = "VISUAL_SECRET"
_JOHNSNOWLABS_MODEL_PATH_SUB = "jsl-model"
_logger = logging.getLogger(__name__)


def _validate_env_vars():
    if _JOHNSNOWLABS_ENV_JSON_LICENSE_KEY not in os.environ:
        raise Exception(
            f"Please set the {_JOHNSNOWLABS_ENV_JSON_LICENSE_KEY}"
            f" environment variable as the raw license.json string from John Snow Labs"
        )
    _set_env_vars()


def _set_env_vars():
    # if json license is detected, we parse it and set the env vars
    loaded_license = json.loads(os.environ[_JOHNSNOWLABS_ENV_JSON_LICENSE_KEY])
    os.environ.update({k: str(v) for k, v in loaded_license.items() if v is not None})


def get_default_pip_requirements():
    """
    Returns:
        A list of default pip requirements for MLflow Models produced by this flavor.
        Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
        that, at minimum, contains these requirements.
    """
    from johnsnowlabs import settings

    if (
        _JOHNSNOWLABS_ENV_HEALTHCARE_SECRET not in os.environ
        and _JOHNSNOWLABS_ENV_VISUAL_SECRET not in os.environ
    ):
        raise Exception(
            f"You need to set either {_JOHNSNOWLABS_ENV_HEALTHCARE_SECRET} "
            f"or {_JOHNSNOWLABS_ENV_VISUAL_SECRET} environment variable. "
            f"Please contact John Snow Labs to get one."
        )

    _SPARK_NLP_JSL_WHEEL_URI = (
        "https://pypi.johnsnowlabs.com/{secret}/spark-nlp-jsl/spark_nlp_jsl-"
        + f"{settings.raw_version_medical}-py3-none-any.whl"
    )

    _SPARK_NLP_VISUAL_WHEEL_URI = (
        "https://pypi.johnsnowlabs.com/{secret}/spark-ocr/"
        f"spark_ocr-{settings.raw_version_ocr}-py3-none-any.whl"
    )

    deps = [
        f"johnsnowlabs_for_databricks=={settings.raw_version_jsl_lib}",
        _get_pinned_requirement("pyspark"),
    ]

    if _JOHNSNOWLABS_ENV_HEALTHCARE_SECRET in os.environ:
        _SPARK_NLP_JSL_WHEEL_URI = _SPARK_NLP_JSL_WHEEL_URI.format(
            secret=os.environ[_JOHNSNOWLABS_ENV_HEALTHCARE_SECRET]
        )
        deps.append(_SPARK_NLP_JSL_WHEEL_URI)

    if _JOHNSNOWLABS_ENV_VISUAL_SECRET in os.environ:
        _SPARK_NLP_VISUAL_WHEEL_URI = _SPARK_NLP_VISUAL_WHEEL_URI.format(
            secret=os.environ[_JOHNSNOWLABS_ENV_VISUAL_SECRET]
        )
        deps.append(_SPARK_NLP_VISUAL_WHEEL_URI)

    return deps


def get_default_conda_env():
    """
    Returns:
        The default Conda environment for MLflow Models produced by calls to
        :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name="johnsnowlabs"))
def log_model(
    spark_model,
    artifact_path: str | None = None,
    conda_env=None,
    code_paths=None,
    dfs_tmpdir=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
    store_license=False,
    name: str | None = None,
    params: dict[str, Any] | None = None,
    tags: dict[str, Any] | None = None,
    model_type: str | None = None,
    step: int = 0,
    model_id: str | None = None,
):
    """
    Log a ``Johnsnowlabs NLUPipeline`` created via `nlp.load()
    <https://nlp.johnsnowlabs.com/docs/en/jsl/load_api>`_, as an MLflow artifact for the current
    run. This uses the MLlib persistence format and produces an MLflow Model with the
    ``johnsnowlabs`` flavor.

    Note: If no run is active, it will instantiate a run to obtain a run_id.

    Args:
        spark_model: NLUPipeline obtained via `nlp.load()
            <https://nlp.johnsnowlabs.com/docs/en/jsl/load_api>`_
        artifact_path: Deprecated. Use `name` instead.
        conda_env: Either a dictionary representation of a Conda environment or the path to a
            Conda environment yaml file. If provided, this describes the environment
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
                        'johnsnowlabs'
                    ]
                }
        code_paths: {{ code_paths }}
        dfs_tmpdir: Temporary directory path on Distributed (Hadoop) File System (DFS) or local
            filesystem if running in local mode. The model is written in this
            destination and then copied into the model's artifact directory. This is
            necessary as Spark ML models read from and write to DFS if running on a
            cluster. If this operation completes successfully, all temporary files
            created on the DFS are removed. Defaults to ``/tmp/mlflow``.
        registered_model_name: If given, create a model version under
            ``registered_model_name``, also creating a registered model if one
            with the given name does not exist.
        signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
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

        input_example: {{ input_example }}
        await_registration_for: Number of seconds to wait for the model version to finish
            being created and is in ``READY`` status. By default, the function
            waits for five minutes. Specify 0 or None to skip waiting.
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata:  {{ metadata }}
        store_license: If True, the license will be stored with the model and used and re-loading
            it.
        name: {{ name }}
        params: {{ params }}
        tags: {{ tags }}
        model_type: {{ model_type }}
        step: {{ step }}
        model_id: {{ model_id }}

    Returns:
        A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
        metadata of the logged model.

    .. code-block:: python
        :caption: Example

        import os
        import json
        import pandas as pd
        import mlflow
        from johnsnowlabs import nlp

        # Write your raw license.json string into the 'JOHNSNOWLABS_LICENSE_JSON' env variable
        creds = {
            "AWS_ACCESS_KEY_ID": "...",
            "AWS_SECRET_ACCESS_KEY": "...",
            "SPARK_NLP_LICENSE": "...",
            "SECRET": "...",
        }
        os.environ["JOHNSNOWLABS_LICENSE_JSON"] = json.dumps(creds)

        # Download & Install Jars/Wheels if missing and Start a spark Session
        nlp.start()

        # For more details on trainable models and parameterization like embedding choice see
        # https://nlp.johnsnowlabs.com/docs/en/jsl/training
        trainable_classifier = nlp.load("train.classifier")

        # Create a sample training dataset
        data = pd.DataFrame(
            {"text": ["I hate covid ", "I love covid"], "y": ["negative", "positive"]}
        )

        # Fit and get a trained classifier
        trained_classifier = trainable_classifier.fit(data)
        trained_classifier.predict("He hates covid")

        # Log it
        mlflow.johnsnowlabs.log_model(trained_classifier, name="my_trained_model")
    """

    _validate_env_vars()
    run_id = mlflow.tracking.fluent._get_or_start_run().info.run_id
    run_root_artifact_uri = mlflow.get_artifact_uri()
    remote_model_path = None

    # If the artifact URI is a local filesystem path, defer to Model.log() to persist the model,
    # since Spark may not be able to write directly to the driver's filesystem. For example,
    # writing to `file:/uri` will write to the local filesystem from each executor, which will
    # be incorrect on multi-node clusters.
    # If the artifact URI is not a local filesystem path we attempt to write directly to the
    # artifact repo via Spark. If this fails, we defer to Model.log().
    if is_local_uri(run_root_artifact_uri) or not _maybe_save_model(
        spark_model,
        append_to_uri_path(run_root_artifact_uri, name),
    ):
        return Model.log(
            artifact_path=artifact_path,
            name=name,
            flavor=mlflow.johnsnowlabs,
            spark_model=spark_model,
            conda_env=conda_env,
            code_paths=code_paths,
            dfs_tmpdir=dfs_tmpdir,
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=input_example,
            await_registration_for=await_registration_for,
            pip_requirements=pip_requirements,
            extra_pip_requirements=extra_pip_requirements,
            metadata=metadata,
            params=params,
            tags=tags,
            model_type=model_type,
            step=step,
            model_id=model_id,
        )
    # Otherwise, override the default model log behavior and save model directly to artifact repo
    logged_model = _initialize_logged_model(
        name=name,
        source_run_id=run.info.run_id if (run := mlflow.active_run()) else None,
        model_type=model_type,
        params=params,
        tags=tags,
        flavor=FLAVOR_NAME,
    )
    mlflow_model = Model(artifact_path=logged_model.artifact_location, run_id=run_id)
    with TempDir() as tmp:
        tmp_model_metadata_dir = tmp.path()
        _save_model_metadata(
            tmp_model_metadata_dir,
            spark_model,
            mlflow_model,
            conda_env,
            code_paths,
            signature=signature,
            input_example=input_example,
            pip_requirements=pip_requirements,
            extra_pip_requirements=extra_pip_requirements,
            remote_model_path=remote_model_path,
            store_license=store_license,
        )
        mlflow.tracking.fluent.log_artifacts(tmp_model_metadata_dir, name)
        mlflow.tracking.fluent._record_logged_model(mlflow_model)
        if registered_model_name is not None:
            mlflow.register_model(
                f"runs:/{logged_model.model_id}",
                registered_model_name,
                await_registration_for,
            )
        return mlflow_model.get_model_info()


def _save_model_metadata(
    dst_dir,
    spark_model,
    mlflow_model,
    conda_env,
    code_paths,
    signature=None,
    input_example=None,
    pip_requirements=None,
    extra_pip_requirements=None,
    remote_model_path=None,
    store_license=False,
):
    """
    Saves model metadata into the passed-in directory.
    If mlflowdbfs is not used, the persisted metadata assumes that a model can be
    loaded from a relative path to the metadata file (currently hard-coded to "jsl-model").
    If mlflowdbfs is used, remote_model_path should be provided, and the model needs to
    be loaded from the remote_model_path.
    """

    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, dst_dir)

    code_dir_subpath = _validate_and_copy_code_paths(code_paths, dst_dir)

    # add the johnsnowlabs flavor
    import pyspark

    mlflow_model.add_flavor(
        FLAVOR_NAME,
        pyspark_version=pyspark.__version__,
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
    if size := get_total_file_size(dst_dir):
        mlflow_model.model_size_bytes = size
    mlflow_model.save(str(Path(dst_dir) / MLMODEL_FILE_NAME))

    if conda_env is None:
        default_reqs = get_default_pip_requirements() if pip_requirements is None else None
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(
            default_reqs,
            pip_requirements,
            extra_pip_requirements,
        )
    else:
        conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)

    with open(str(Path(dst_dir) / _CONDA_ENV_FILE_NAME), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    # Save `constraints.txt` if necessary
    if pip_constraints:
        write_to(str(Path(dst_dir) / _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))
    write_to(str(Path(dst_dir) / _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))

    _PythonEnv.current().to_yaml(str(Path(dst_dir) / _PYTHON_ENV_FILE_NAME))

    _save_jars_and_lic(dst_dir)


def _save_jars_and_lic(dst_dir, store_license=False):
    from johnsnowlabs.auto_install.jsl_home import get_install_suite_from_jsl_home
    from johnsnowlabs.py_models.jsl_secrets import JslSecrets

    deps_data_path = Path(dst_dir) / _JOHNSNOWLABS_MODEL_PATH_SUB / "jars.jsl"
    deps_data_path.mkdir(parents=True, exist_ok=True)

    suite = get_install_suite_from_jsl_home(
        False,
        visual=_JOHNSNOWLABS_ENV_VISUAL_SECRET in os.environ,
    )
    if suite.hc.get_java_path():
        shutil.copy2(suite.hc.get_java_path(), deps_data_path / "hc_jar.jar")
    if suite.nlp.get_java_path():
        shutil.copy2(suite.nlp.get_java_path(), deps_data_path / "os_jar.jar")
    if suite.ocr.get_java_path():
        shutil.copy2(suite.ocr.get_java_path(), deps_data_path / "visual_nlp.jar")

    if store_license:
        # Read the secrets from env vars and write to license.json
        secrets = JslSecrets.build_or_try_find_secrets()
        if secrets.HC_LICENSE:
            deps_data_path.joinpath("license.json").write(secrets.json())


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name="johnsnowlabs"))
def save_model(
    spark_model,
    path,
    mlflow_model=None,
    conda_env=None,
    code_paths=None,
    dfs_tmpdir=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
    store_license=False,
):
    """
    Save a Spark johnsnowlabs Model to a local path.

    By default, this function saves models using the Spark MLlib persistence mechanism.

    Args:
        spark_model: Either a pyspark.ml.pipeline.PipelineModel or nlu.NLUPipeline object to be
            saved. `Every johnsnowlabs model <https://nlp.johnsnowlabs.com/models>`_
            is a PipelineModel and loadable as nlu.NLUPipeline.
        path: Local path where the model is to be saved.
        mlflow_model: MLflow model config this flavor is being added to.
        conda_env: Either a dictionary representation of a Conda environment or the path to a
            Conda environment yaml file. If provided, this describes the environment
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
                        'johnsnowlabs'
                    ]
                }
        code_paths: {{ code_paths }}
        dfs_tmpdir: Temporary directory path on Distributed (Hadoop) File System (DFS) or local
            filesystem if running in local mode. The model is be written in this
            destination and then copied to the requested local path. This is necessary
            as Spark ML models read from and write to DFS if running on a cluster. All
            temporary files created on the DFS are removed if this operation
            completes successfully. Defaults to ``/tmp/mlflow``.
        signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
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
        input_example: {{ input_example }}
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata: {{ metadata }}
        store_license: If True, the license will be stored with the model and used and
            re-loading it.

    .. code-block:: python
        :caption: Example

        from johnsnowlabs import nlp
        import mlflow
        import os

        # Write your raw license.json string into the 'JOHNSNOWLABS_LICENSE_JSON' env variable
        creds = {
            "AWS_ACCESS_KEY_ID": "...",
            "AWS_SECRET_ACCESS_KEY": "...",
            "SPARK_NLP_LICENSE": "...",
            "SECRET": "...",
        }
        os.environ["JOHNSNOWLABS_LICENSE_JSON"] = json.dumps(creds)

        # Download & Install Jars/Wheels if missing and Start a spark Session
        nlp.start()

        # load a model
        model = nlp.load("en.classify.bert_sequence.covid_sentiment")
        model.predict(["I hate covid", "I love covid"])

        # Save model as pyfunc and johnsnowlabs format
        mlflow.johnsnowlabs.save_model(model, "saved_model")
        model = mlflow.johnsnowlabs.load_model("saved_model")
        # Predict with reloaded model,
        # supports datatypes defined in https://nlp.johnsnowlabs.com/docs/en/jsl/predict_api#supported-data-types
        model.predict(["I hate covid", "I love covid"])
    """
    _validate_env_vars()
    if mlflow_model is None:
        mlflow_model = Model()
    if metadata is not None:
        mlflow_model.metadata = metadata
    # Spark ML stores the model on DFS if running on a cluster
    # Save it to a DFS temp dir first and copy it to local path
    if dfs_tmpdir is None:
        dfs_tmpdir = MLFLOW_DFS_TMP.get()
    tmp_path = generate_tmp_dfs_path(dfs_tmpdir)

    _unpack_and_save_model(spark_model, tmp_path)
    sparkml_data_path = os.path.abspath(str(Path(path) / _JOHNSNOWLABS_MODEL_PATH_SUB))
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
        conda_env=conda_env,
        code_paths=code_paths,
        signature=signature,
        input_example=input_example,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        store_license=store_license,
    )


def _load_model_databricks(dfs_tmpdir, local_model_path):
    from johnsnowlabs import nlp

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
    from johnsnowlabs import nlp

    dfs_tmpdir = generate_tmp_dfs_path(dfs_tmpdir_base or MLFLOW_DFS_TMP.get())
    if databricks_utils.is_in_cluster() and databricks_utils.is_dbfs_fuse_available():
        return _load_model_databricks(
            dfs_tmpdir, local_model_path or _download_artifact_from_uri(model_uri)
        )
    # model_uri = _HadoopFileSystem.maybe_copy_from_uri(model_uri, dfs_tmpdir, local_model_path)
    if model_uri and not local_model_path:
        local_model_path = _download_artifact_from_uri(model_uri)
    _get_or_create_sparksession(local_model_path)

    if _JOHNSNOWLABS_MODEL_PATH_SUB not in local_model_path:
        local_model_path = str(Path(local_model_path) / _JOHNSNOWLABS_MODEL_PATH_SUB)

    return nlp.load(path=local_model_path)


def load_model(model_uri, dfs_tmpdir=None, dst_path=None):
    """
    Load the Johnsnowlabs MLflow model from the path.

    Args:
        model_uri: The location, in URI format, of the MLflow model. For example:

            - ``/Users/me/path/to/local/model``
            - ``relative/path/to/local/model``
            - ``s3://my_bucket/path/to/model``
            - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
            - ``models:/<model_name>/<model_version>``
            - ``models:/<model_name>/<stage>``

            For more information about supported URI schemes, see
            `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
            artifact-locations>`_.
        dfs_tmpdir: Temporary directory path on Distributed (Hadoop) File System (DFS) or local
            filesystem if running in local mode. The model is loaded from this
            destination. Defaults to ``/tmp/mlflow``.
        dst_path: The local filesystem path to which to download the model artifact.
            This directory must already exist. If unspecified, a local output
            path will be created.

    Returns:
        A
        `nlu.NLUPipeline <https://nlp.johnsnowlabs.com/docs/en/jsl/predict_api>`_.

    .. code-block:: python
        :caption: Example

        import mlflow
        from johnsnowlabs import nlp
        import os

        # Write your raw license.json string into the 'JOHNSNOWLABS_LICENSE_JSON' env variable
        creds = {
            "AWS_ACCESS_KEY_ID": "...",
            "AWS_SECRET_ACCESS_KEY": "...",
            "SPARK_NLP_LICENSE": "...",
            "SECRET": "...",
        }
        os.environ["JOHNSNOWLABS_LICENSE_JSON"] = json.dumps(creds)

        # start a spark session
        nlp.start()
        # Load you MLflow Model
        model = mlflow.johnsnowlabs.load_model("johnsnowlabs_model")

        # Make predictions on test documents
        # supports datatypes defined in https://nlp.johnsnowlabs.com/docs/en/jsl/predict_api#supported-data-types
        prediction = model.transform(["I love Covid", "I hate Covid"])
    """
    # This MUST be called prior to appending the model flavor to `model_uri` in order
    # for `artifact_path` to take on the correct value for model loading via mlflowdbfs.
    _validate_env_vars()
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
    local_sparkml_model_path = str(Path(local_mlflow_model_path) / flavor_conf["model_data"])
    return _load_model(
        model_uri=sparkml_model_uri,
        dfs_tmpdir_base=dfs_tmpdir,
        local_model_path=local_sparkml_model_path,
    )


def _load_pyfunc(path, spark=None):
    """Load PyFunc implementation. Called by ``pyfunc.load_model``.

    Args:
        path: Local filesystem path to the MLflow Model with the ``johnsnowlabs`` flavor.
        spark: Optionally pass spark context when using pyfunc as UDF. required, because
            we cannot fetch the Sparkcontext inside of the Workernode which executes the UDF.

    Returns:
        None.

    """
    return _PyFuncModelWrapper(
        _load_model(model_uri=path),
        spark or _get_or_create_sparksession(path),
    )


def _get_or_create_sparksession(model_path=None):
    """Check if SparkSession running and get it.

    If none exists, create a new one using jars in model_path. If model_path not defined, rely on
    nlp.start() to create a new one using johnsnowlabs Jar resolution method. See
    https://nlp.johnsnowlabs.com/docs/en/jsl/start-a-sparksession and
    https://nlp.johnsnowlabs.com/docs/en/jsl/install_advanced.

    Args:
        model_path:

    Returns:

    """
    from johnsnowlabs import nlp

    from mlflow.utils._spark_utils import _get_active_spark_session

    _validate_env_vars()

    spark = _get_active_spark_session()
    if spark is None:
        spark_conf = {}
        spark_conf["spark.python.worker.reuse"] = "true"
        os.environ["PYSPARK_PYTHON"] = sys.executable
        os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
        if model_path:
            jar_paths, license_path = _fetch_deps_from_path(model_path)
            if license_path:
                with open(license_path) as f:
                    loaded_license = json.load(f)
                    os.environ.update(
                        {k: str(v) for k, v in loaded_license.items() if v is not None}
                    )
                    os.environ["JSL_NLP_LICENSE"] = loaded_license["HC_LICENSE"]
            _logger.info("Starting a new Session with Jars: %s", jar_paths)
            spark = nlp.start(
                nlp=False,
                spark_nlp=False,
                jar_paths=jar_paths,
                json_license_path=license_path,
                create_jsl_home_if_missing=False,
                spark_conf=spark_conf,
            )
        else:
            spark = nlp.start()
    return spark


def _fetch_deps_from_path(local_model_path):
    if _JOHNSNOWLABS_MODEL_PATH_SUB not in local_model_path:
        local_model_path = Path(local_model_path) / _JOHNSNOWLABS_MODEL_PATH_SUB / "jars.jsl"
    else:
        local_model_path = Path(local_model_path) / "jars.jsl"

    jar_paths = [
        str(local_model_path / file) for file in local_model_path.iterdir() if file.suffix == ".jar"
    ]
    license_path = [
        str(local_model_path / file)
        for file in local_model_path.iterdir()
        if file.name == "license.json"
    ]

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
        spark_model.predict("Init")
        try:
            spark_model.vanilla_transformer_pipe.write().overwrite().save(dst)
        except Exception:
            # for mlflowdbfs_path we cannot use overwrite, gives
            # org.apache.hadoop.fs.UnsupportedFileSystemException: No FileSystem for scheme
            # "mlflowdbfs"
            spark_model.save(dst)


class _PyFuncModelWrapper:
    """
    Wrapper around NLUPipeline providing interface for scoring pandas DataFrame.
    """

    def __init__(
        self,
        spark_model,
        spark=None,
    ):
        # we have this `or`, so we support _PyFuncModelWrapper(nlu_ref)
        self.spark = spark or _get_or_create_sparksession()
        self.spark_model = spark_model

    def get_raw_model(self):
        """
        Returns the underlying model.
        """
        return self.spark_model

    def predict(self, text, params: dict[str, Any] | None = None):
        """Generate predictions given input data in a pandas DataFrame.

        Args:
            text: pandas DataFrame containing input data.
            params: Additional parameters to pass to the model for inference.

        Returns:
            List with model predictions.

        """
        output_level = params.get("output_level", "") if params else ""
        return self.spark_model.predict(text, output_level=output_level).reset_index().to_json()
