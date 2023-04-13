"""
The ``mlflow.openai`` module provides an API for logging and loading OpenAI models.

Credential management for OpenAI on Databricks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When this flavor logs a model on Databricks, it saves a JSON file with the following contents as
``openai.json`` if the ``MLFLOW_OPENAI_SECRET_SCOPE`` environment variable is set.

.. code-block:: json

    {
        "OPENAI_API_TYPE": "{scope}:openai_api_type",
        "OPENAI_API_BASE": "{scope}:openai_api_base",
        "OPENAI_API_KEY": "{scope}:openai_api_key",
        "OPENAI_API_KEY_PATH": "{scope}:openai_api_key_path",
        "OPENAI_ORGANIZATION": "{scope}:openai_organization"
    }

- ``{scope}`` is the value of the ``MLFLOW_OPENAI_SECRET_SCOPE`` environment variable.
- The keys are the environment variables that the ``openai-python`` package uses to
  configure the API client.
- The values are the references to the secrets that store the values of the environment
  variables.

When the logged model is served on Databricks, each secret will be resolved and set as the
corresponding environment variable. See https://docs.databricks.com/security/secrets/index.html
for how to set up secrets on Databricks.
"""
import os
import json
import yaml
import asyncio
import logging
from enum import Enum

import mlflow
from mlflow import pyfunc
from mlflow.models import Model, ModelInputExample
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import _save_example
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
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
from mlflow.utils.file_utils import write_to
from mlflow.utils.model_utils import (
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _add_code_from_conf_to_system_path,
    _validate_and_prepare_target_save_path,
)
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.docstring_utils import format_docstring, LOG_MODEL_PARAM_DOCS
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.types import Schema, ColSpec
from mlflow.environment_variables import _MLFLOW_OPENAI_TESTING, MLFLOW_OPENAI_SECRET_SCOPE
from mlflow.utils.annotations import experimental
from mlflow.utils.databricks_utils import is_in_databricks_runtime

FLAVOR_NAME = "openai"


_logger = logging.getLogger(__name__)


@experimental
def get_default_pip_requirements():
    """
    :return: A list of default pip requirements for MLflow Models produced by this flavor.
             Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
             that, at minimum, contains these requirements.
    """
    return list(map(_get_pinned_requirement, ["openai", "aiohttp", "tiktoken", "tenacity"]))


@experimental
def get_default_conda_env():
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


def _get_class_to_task_mapping():
    from openai.api_resources import (
        Audio,
        ChatCompletion,
        Completion,
        Edit,
        Deployment,
        Embedding,
        Engine,
        FineTune,
        File,
        Image,
        Model as OpenAIModel,
        Moderation,
    )

    return {
        Audio: Audio.OBJECT_NAME,
        ChatCompletion: ChatCompletion.OBJECT_NAME,
        Completion: Completion.OBJECT_NAME,
        Edit: Edit.OBJECT_NAME,
        Deployment: Deployment.OBJECT_NAME,
        Embedding: Embedding.OBJECT_NAME,
        Engine: Engine.OBJECT_NAME,
        File: File.OBJECT_NAME,
        Image: Image.OBJECT_NAME,
        FineTune: FineTune.OBJECT_NAME,
        OpenAIModel: OpenAIModel.OBJECT_NAME,
        Moderation: "moderations",
    }


def _class_to_task(cls):
    task = _get_class_to_task_mapping().get(cls)
    if task is None:
        raise mlflow.MlflowException(
            f"Unsupported class: {cls}", error_code=INVALID_PARAMETER_VALUE
        )
    return task


def _get_model_name(model):
    import openai

    if isinstance(model, str):
        return model
    elif isinstance(model, openai.Model):
        return model.id
    else:
        raise mlflow.MlflowException(
            f"Unsupported model type: {type(model)}", error_code=INVALID_PARAMETER_VALUE
        )


def _get_task_name(task):
    if isinstance(task, str):
        return task
    elif isinstance(task, type):
        return _class_to_task(task)
    else:
        raise mlflow.MlflowException(
            f"Unsupported task type: {type(task)}", error_code=INVALID_PARAMETER_VALUE
        )


def get_openai_package_version():
    import openai

    try:
        return openai.__version__
    except AttributeError:
        # openai < 0.27.5 doesn't have a __version__ attribute
        return openai.version.VERSION


# See https://github.com/openai/openai-python/blob/cf03fe16a92cd01f2a8867537399c12e183ba58e/openai/__init__.py#L30-L38
# for the list of environment variables that openai-python uses
class OpenAIEnvVar(str, Enum):
    OPENAI_API_TYPE = "OPENAI_API_TYPE"
    OPENAI_API_BASE = "OPENAI_API_BASE"
    OPENAI_API_KEY = "OPENAI_API_KEY"
    OPENAI_API_KEY_PATH = "OPENAI_API_KEY_PATH"
    OPENAI_ORGANIZATION = "OPENAI_ORGANIZATION"

    @property
    def secret_key(self):
        return self.value.lower()

    @classmethod
    def read_environ(cls):
        env_vars = {}
        for e in OpenAIEnvVar:
            if value := os.getenv(e.value):
                env_vars[e.value] = value
        return env_vars


def _log_credentials_json(local_model_dir, scope):
    with open(os.path.join(local_model_dir, "openai.json"), "w") as f:
        json.dump({e.value: f"{scope}:{e.secret_key}" for e in OpenAIEnvVar}, f)


@experimental
@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    model,
    path,
    task,
    conda_env=None,
    code_paths=None,
    mlflow_model=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
    **kwargs,
):
    """
    Save an OpenAI model to a path on the local file system.

    :param model: The OpenAI model name or reference instance, e.g.,
                  ``openai.Model.retrieve("gpt-3.5-turbo")``.
    :param path: Local path where the model is to be saved.
    :param task: The task the model is performing, e.g., ``openai.ChatCompletion`` or
                 ``'chat.completions'``.
    :param conda_env: {{ conda_env }}
    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories
                       containing file dependencies). These files are *prepended* to the system
                       path when the model is loaded.
    :param mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.

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
    :param  kwargs: Keyword arguments specific to the OpenAI task, such as the ``temperature`` or
                    or ``top_p`` value to use for chat completion.
    """
    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)
    path = os.path.abspath(path)
    _validate_and_prepare_target_save_path(path)
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)
    task = _get_task_name(task)

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    elif task == "chat.completions":
        mlflow_model.signature = ModelSignature(
            inputs=Schema(
                [ColSpec(type="string", name="role"), ColSpec(type="string", name="content")],
            ),
            outputs=Schema([ColSpec(type="string", name=None)]),
        )
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)
    if metadata is not None:
        mlflow_model.metadata = metadata
    model_data_subpath = "model.json"
    model_data_path = os.path.join(path, model_data_subpath)
    model_dict = {
        "model": _get_model_name(model),
        "task": task,
        **kwargs,
    }
    with open(model_data_path, "w") as f:
        json.dump(model_dict, f)

    if task == "chat.completions":
        pyfunc.add_to_model(
            mlflow_model,
            loader_module="mlflow.openai",
            data=model_data_subpath,
            conda_env=_CONDA_ENV_FILE_NAME,
            python_env=_PYTHON_ENV_FILE_NAME,
            code=code_dir_subpath,
        )
    mlflow_model.add_flavor(
        FLAVOR_NAME,
        openai_version=get_openai_package_version(),
        data=model_data_subpath,
        code=code_dir_subpath,
    )
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if is_in_databricks_runtime():
        if scope := MLFLOW_OPENAI_SECRET_SCOPE.get():
            _log_credentials_json(path, scope)
        else:
            _logger.info(
                "No secret scope specified, skipping logging of secrets for OpenAI credentials. "
                "See https://mlflow.org/docs/latest/python_api/mlflow.openai.html#credential-management-for-openai-on-databricks "
                "for more information."
            )

    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements()
            inferred_reqs = mlflow.models.infer_pip_requirements(
                path, FLAVOR_NAME, fallback=default_reqs
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

    with open(os.path.join(path, _CONDA_ENV_FILE_NAME), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    # Save `constraints.txt` if necessary
    if pip_constraints:
        write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))

    # Save `requirements.txt`
    write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))

    _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))


@experimental
@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    model,
    artifact_path,
    task,
    conda_env=None,
    code_paths=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
    **kwargs,
):
    """
    Log an OpenAI model as an MLflow artifact for the current run.

    :param model: The OpenAI model name or reference instance, e.g.,
                  ``openai.Model.retrieve("gpt-3.5-turbo")``.
    :param artifact_path: Run-relative artifact path.
    :param task: The task the model is performing, e.g., ``openai.ChatCompletion`` or
                 ``'chat.completions'``.
    :param conda_env: {{ conda_env }}
    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories
                       containing file dependencies). These files are *prepended* to the system
                       path when the model is loaded.
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
    :param model_format: File format in which the model is to be saved.
    :param metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.

                     .. Note:: Experimental: This parameter may change or be removed in a future
                                             release without warning.
    :param  kwargs: Keyword arguments specific to the OpenAI task, such as the ``temperature`` or
                    or ``top_p`` value to use for chat completion.
    :return: A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
             metadata of the logged model.
    """
    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.openai,
        registered_model_name=registered_model_name,
        model=model,
        task=task,
        conda_env=conda_env,
        code_paths=code_paths,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        metadata=metadata,
        **kwargs,
    )


def _load_model(path):
    with open(path) as f:
        return json.load(f)


def _has_content_and_role(d):
    return "content" in d and "role" in d


class _OpenAIWrapper:
    def __init__(self, model):
        if model["task"] != "chat.completions":
            raise mlflow.MlflowException.invalid_parameter_value(
                "Currently, only the 'chat.completions' task is supported for pyfunc",
            )
        self.model = model

    def predict(self, data):
        import pandas as pd
        from mlflow.openai.api_request_parallel_processor import process_api_requests

        if isinstance(data, pd.DataFrame):
            if "content" not in data or "role" not in data:
                raise mlflow.MlflowException.invalid_parameter_value(
                    ("The input dataframe must contain the columns 'content' and 'role'"),
                )
            messages = data.to_dict(orient="records")
        elif isinstance(data, list) and all(isinstance(d, dict) for d in data):
            if not all(map(_has_content_and_role, data)):
                raise mlflow.MlflowException.invalid_parameter_value(
                    "The input list of dictionaries must contain the keys 'content' and 'role'",
                )
            messages = data
        elif isinstance(data, list) and all(isinstance(d, str) for d in data):
            messages = [{"role": "user", "content": x} for x in data]
        elif isinstance(data, dict):
            if not _has_content_and_role(data):
                raise mlflow.MlflowException.invalid_parameter_value(
                    "The input dictionary must contain the keys 'content' and 'role'",
                )
            messages = [data]
        elif isinstance(data, str):
            messages = [{"role": "user", "content": data}]
        else:
            raise mlflow.MlflowException.invalid_parameter_value(
                (
                    "The input data must be a Pandas DataFrame, a list of dictionaries, a list "
                    "of strings, or a string, got {}".format(type(data).__name__)
                ),
            )

        model_dict = self.model.copy()
        model_dict.pop("task", None)
        model_dict.pop("messages", None)
        requests = [
            {
                **model_dict,
                # numpy array is not JSON serializable, so convert to list
                "messages": [message],
            }
            for message in messages
        ]
        api_key = os.getenv(OpenAIEnvVar.OPENAI_API_KEY.value)
        if api_key is None:
            raise mlflow.MlflowException(
                "OpenAI API key must be set in the OPENAI_API_KEY environment variable."
            )
        results = asyncio.run(process_api_requests(requests))
        return [r["choices"][0]["message"]["content"] for r in results]


class _TestOpenAIWrapper(_OpenAIWrapper):
    """
    A wrapper class that should be used for testing purposes only.
    """

    def predict(self, data):
        from tests.openai.test_openai_model_export import _mock_async_request

        with _mock_async_request():
            return super().predict(data)


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_model``.

    :param path: Local filesystem path to the MLflow Model with the ``openai`` flavor.
    """
    wrapper_cls = _TestOpenAIWrapper if _MLFLOW_OPENAI_TESTING.get() else _OpenAIWrapper
    return wrapper_cls(_load_model(path))


@experimental
def load_model(model_uri, dst_path=None):
    """
    Load an OpenAI model from a local file or a run.

    :param model_uri: The location, in URI format, of the MLflow model. For example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/tracking.html#
                      artifact-locations>`_.
    :param dst_path: The local filesystem path to which to download the model artifact.
                     This directory must already exist. If unspecified, a local output
                     path will be created.

    :return: A dictionary representing the OpenAI model.
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    flavor_conf = _get_flavor_configuration(local_model_path, FLAVOR_NAME)
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
    model_data_path = os.path.join(local_model_path, flavor_conf.get("data", "model.json"))
    return _load_model(model_data_path)
