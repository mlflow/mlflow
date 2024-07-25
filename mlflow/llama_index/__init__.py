import logging
import os
from typing import Any, Dict, List, Optional, Union

import yaml

import mlflow
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
from mlflow.tracing.provider import trace_disabled
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import autologging_integration
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
    _validate_env_arguments,
)
from mlflow.utils.file_utils import get_total_file_size, write_to
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path,
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

FLAVOR_NAME = "llama_index"
_INDEX_PERSIST_FOLDER = "index"
_SETTINGS_FILE = "settings.json"


_logger = logging.getLogger(__name__)


def get_default_pip_requirements():
    """
    Returns:
        A list of default pip requirements for MLflow Models produced by this flavor.
        Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
        that, at a minimum, contains these requirements.
    """
    return [_get_pinned_requirement("llama-index")]


def get_default_conda_env():
    """
    Returns:
        The default Conda environment for MLflow Models produced by calls to
        :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


def _validate_engine_type(engine_type: str):
    from mlflow.llama_index.pyfunc_wrapper import SUPPORTED_ENGINES

    if engine_type not in SUPPORTED_ENGINES:
        raise ValueError(
            f"Currently mlflow only supports the following engine types: "
            f"{SUPPORTED_ENGINES}. {engine_type} is not supported, so please "
            "use one of the above types."
        )


def _get_llama_index_version() -> str:
    try:
        import llama_index.core

        return llama_index.core.__version__
    except ImportError:
        raise MlflowException(
            "The llama_index module is not installed. "
            "Please install it via `pip install llama-index`."
        )


@experimental
@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
@trace_disabled  # Suppress traces while loading model
def save_model(
    index,
    path: str,
    engine_type: str,
    model_config: Optional[Dict[str, Any]] = None,
    code_paths=None,
    mlflow_model: Optional[Model] = None,
    signature: Optional[ModelSignature] = None,
    input_example: Optional[ModelInputExample] = None,
    pip_requirements: Optional[Union[List[str], str]] = None,
    extra_pip_requirements: Optional[Union[List[str], str]] = None,
    conda_env=None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a LlamaIndex index to a path on the local file system.

    Args:
        index: LlamaIndex index to be saved.
        path: Local path where the serialized model (as YAML) is to be saved.
        engine_type: Determine the inference interface for the index when loaded as a pyfunc
            model. The supported types are as follows:

            - ``"chat"``: load the index as an instance of the LlamaIndex
              `ChatEngine <https://docs.llamaindex.ai/en/stable/module_guides/deploying/chat_engines/>`_.
            - ``"query"``: load the index as an instance of the LlamaIndex
              `QueryEngine <https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/>`_.
            - ``"retriever"``: load the index as an instance of the LlamaIndex
              `Retriever <https://docs.llamaindex.ai/en/stable/module_guides/querying/retriever/>`_.

        model_config: Keyword arguments to be passed to the LlamaIndex engine at instantiation.
            Note that not all llama index objects have supported serialization; when an object is
            not supported, an info log message will be emitted and the unsupported object will be
            dropped.
        code_paths: {{ code_paths }}
        mlflow_model: An MLflow model object that specifies the flavor that this model is being
            added to.
        signature: A Model Signature object that describes the input and output Schema of the
            model. The model signature can be inferred using ``infer_signature`` function
            of ``mlflow.models.signature``.
        input_example: {{ input_example }}
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        conda_env: {{ conda_env }}
        metadata: {{ metadata }}
    """
    from mlflow.llama_index.pyfunc_wrapper import create_engine_wrapper
    from mlflow.llama_index.serialize_objects import serialize_settings

    _validate_engine_type(engine_type)
    _validate_index(index)
    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    path = os.path.abspath(path)
    _validate_and_prepare_target_save_path(path)
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    if mlflow_model is None:
        mlflow_model = Model()
    saved_example = _save_example(mlflow_model, input_example, path)

    if signature is None and saved_example is not None:
        wrapped_model = create_engine_wrapper(index, engine_type, model_config)
        signature = _infer_signature_from_input_example(saved_example, wrapped_model)
    elif signature is False:
        signature = None

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if metadata is not None:
        mlflow_model.metadata = metadata

    # NB: llama_index.core.Settings is a singleton that manages the storage/service context
    # for a given llama_index application. Given it holds the required objects for most of
    # the index's functionality, we look to serialize the entire object. For components of
    # the object that are not serializable, we log a warning.
    settings_path = os.path.join(path, _SETTINGS_FILE)
    serialize_settings(settings_path)

    _save_index(index, path)

    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.llama_index",
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
        model_config=model_config,
    )
    mlflow_model.add_flavor(
        FLAVOR_NAME,
        llama_index_version=_get_llama_index_version(),
        code=code_dir_subpath,
        engine_type=engine_type,
    )
    if size := get_total_file_size(path):
        mlflow_model.model_size_bytes = size
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if conda_env is None:
        default_reqs = None
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements()
            inferred_reqs = mlflow.models.infer_pip_requirements(
                str(path), FLAVOR_NAME, fallback=default_reqs
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

    if pip_constraints:
        write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))

    write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))

    _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))


@experimental
@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
@trace_disabled  # Suppress traces while loading model
def log_model(
    index,
    artifact_path: str,
    engine_type: str,
    model_config: Optional[Dict[str, Any]] = None,
    code_paths: Optional[List[str]] = None,
    registered_model_name: Optional[str] = None,
    signature: Optional[ModelSignature] = None,
    input_example: Optional[ModelInputExample] = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements: Optional[Union[List[str], str]] = None,
    extra_pip_requirements: Optional[Union[List[str], str]] = None,
    conda_env=None,
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    """
    Log a LlamaIndex index as an MLflow artifact for the current run.

    Args:
        index: LlamaIndex index to be saved.
        artifact_path: Local path where the serialized model (as YAML) is to be saved.
        engine_type: Determine the inference interface for the index when loaded as a pyfunc
            model. The supported types are as follows:

            - ``"chat"``: load the index as an instance of the LlamaIndex
              `ChatEngine <https://docs.llamaindex.ai/en/stable/module_guides/deploying/chat_engines/>`_.
            - ``"query"``: load the index as an instance of the LlamaIndex
              `QueryEngine <https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/>`_.
            - ``"retriever"``: load the index as an instance of the LlamaIndex
              `Retriever <https://docs.llamaindex.ai/en/stable/module_guides/querying/retriever/>`_.

        model_config: Keyword arguments to be passed to the LlamaIndex engine at instantiation.
            Note that not all llama index objects have supported serialization; when an object is
            not supported, an info log message will be emitted and the unsupported object will be
            dropped.
        code_paths: {{ code_paths }}
        registered_model_name: This argument may change or be removed in a
            future release without warning. If given, create a model
            version under ``registered_model_name``, also creating a
            registered model if one with the given name does not exist.
        signature: A Model Signature object that describes the input and output Schema of the
            model. The model signature can be inferred using ``infer_signature`` function
            of `mlflow.models.signature`.
        input_example: {{ input_example }}
        await_registration_for: Number of seconds to wait for the model version
            to finish being created and is in ``READY`` status.
            By default, the function waits for five minutes.
            Specify 0 or None to skip waiting.
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        conda_env: {{ conda_env }}
        metadata: {{ metadata }}
        kwargs: Additional arguments for :py:class:`mlflow.models.model.Model`
    """

    return Model.log(
        artifact_path=artifact_path,
        engine_type=engine_type,
        model_config=model_config,
        flavor=mlflow.llama_index,
        registered_model_name=registered_model_name,
        index=index,
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


def _validate_index(index):
    from llama_index.core.indices.base import BaseIndex

    if not isinstance(index, BaseIndex):
        raise MlflowException.invalid_parameter_value(
            message=f"The provided object of type {type(index).__name__} is not a valid "
            "index. MLflow llama-index flavor only supports saving LlamaIndex indices."
        )


def _save_index(index, path):
    """Serialize the index."""
    index_path = os.path.join(path, _INDEX_PERSIST_FOLDER)
    index.storage_context.persist(persist_dir=index_path)


def _load_index(path, flavor_conf):
    """Deserialize the index."""
    from llama_index.core import StorageContext, load_index_from_storage

    _add_code_from_conf_to_system_path(path, flavor_conf)

    index_path = os.path.join(path, _INDEX_PERSIST_FOLDER)
    storage_context = StorageContext.from_defaults(persist_dir=index_path)
    return load_index_from_storage(storage_context)


@experimental
@trace_disabled  # Suppress traces while loading model
def load_model(model_uri, dst_path=None):
    """
    Load a LlamaIndex index from a local file or a run.

    Args:
        model_uri: The location, in URI format, of the MLflow model. For example:

            - ``/Users/me/path/to/local/model``
            - ``relative/path/to/local/model``
            - ``s3://my_bucket/path/to/model``
            - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
            - ``mlflow-artifacts:/path/to/model``

            For more information about supported URI schemes, see
            `Referencing Artifacts <https://www.mlflow.org/docs/latest/tracking.html#
            artifact-locations>`_.
        dst_path: The local filesystem path to utilize for downloading the model artifact.
            This directory must already exist if provided. If unspecified, a local output
            path will be created.

    Returns:
        A LlamaIndex index object.
    """
    from mlflow.llama_index.serialize_objects import deserialize_settings

    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)

    settings_path = os.path.join(local_model_path, _SETTINGS_FILE)
    # NB: Settings is a singleton and can be loaded via llama_index.core.Settings
    deserialize_settings(settings_path)
    return _load_index(local_model_path, flavor_conf)


def _load_pyfunc(path, model_config: Optional[Dict[str, Any]] = None):
    from mlflow.llama_index.pyfunc_wrapper import create_engine_wrapper

    index = load_model(path)
    flavor_conf = _get_flavor_configuration(model_path=path, flavor_name=FLAVOR_NAME)
    engine_type = flavor_conf.pop("engine_type")
    return create_engine_wrapper(index, engine_type, model_config)


@experimental
def autolog(
    log_traces: bool = True,
    disable: bool = False,
    silent: bool = False,
):
    """
    Enables (or disables) and configures autologging from LlamaIndex to MLflow. Currently, MLflow
    only supports autologging for tracing.

    Args:
        log_traces: If ``True``, traces are logged for Langchain models by using
            MlflowLangchainTracer as a callback during inference. If ``False``, no traces are
            collected during inference. Default to ``True``.
        disable: If ``True``, disables the LlamaIndex autologging integration. If ``False``,
            enables the LlamaIndex autologging integration.
        silent: If ``True``, suppress all event logs and warnings from MLflow during LlamaIndex
            autologging. If ``False``, show all events and warnings.
    """
    from mlflow.llama_index.tracer import remove_llama_index_tracer, set_llama_index_tracer

    # NB: The @autologging_integration annotation is used for adding shared logic. However, one
    # caveat is that the wrapped function is NOT executed when disable=True is passed. This prevents
    # us from running cleaning up logging when autologging is turned off. To workaround this, we
    # annotate _autolog() instead of this entrypoint, and define the cleanup logic outside it.
    if log_traces and not disable:
        set_llama_index_tracer()
    else:
        remove_llama_index_tracer()

    _autolog(disable=disable, silent=silent)


@autologging_integration(FLAVOR_NAME)
def _autolog(
    disable: bool = False,
    silent: bool = False,
):
    """
    TODO: Implement patching logic for autologging models and artifacts.
    """
