import logging
import os
import tempfile
from typing import Any, Optional, Union

import yaml

import mlflow
from mlflow import pyfunc
from mlflow.entities.model_registry.prompt import Prompt
from mlflow.exceptions import MlflowException
from mlflow.llama_index.constant import FLAVOR_NAME
from mlflow.llama_index.pyfunc_wrapper import create_pyfunc_wrapper
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import (
    MLMODEL_FILE_NAME,
    MODEL_CODE_PATH,
    MODEL_CONFIG,
    _update_active_model_id_based_on_mlflow_model,
)
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import (
    _load_model_code_path,
    _save_example,
    _validate_and_get_model_code_path,
)
from mlflow.tracing.provider import trace_disabled
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.annotations import experimental
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
    _validate_and_copy_file_to_directory,
    _validate_and_get_model_config_from_file,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

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


def _supported_classes():
    from llama_index.core.base.base_query_engine import BaseQueryEngine
    from llama_index.core.chat_engine.types import BaseChatEngine
    from llama_index.core.indices.base import BaseIndex
    from llama_index.core.retrievers import BaseRetriever

    supported = (BaseIndex, BaseChatEngine, BaseQueryEngine, BaseRetriever)

    try:
        from llama_index.core.workflow import Workflow

        supported += (Workflow,)
    except ImportError:
        pass

    return supported


@experimental
@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
@trace_disabled  # Suppress traces while loading model
def save_model(
    llama_index_model,
    path: str,
    engine_type: Optional[str] = None,
    model_config: Optional[Union[str, dict[str, Any]]] = None,
    code_paths=None,
    mlflow_model: Optional[Model] = None,
    signature: Optional[ModelSignature] = None,
    input_example: Optional[ModelInputExample] = None,
    pip_requirements: Optional[Union[list[str], str]] = None,
    extra_pip_requirements: Optional[Union[list[str], str]] = None,
    conda_env=None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """
    Save a LlamaIndex model to a path on the local file system.

    .. attention::

        Saving a non-index object is only supported in the 'Model-from-Code' saving mode.
        Please refer to the `Models From Code Guide <https://www.mlflow.org/docs/latest/model/models-from-code.html>`_
        for more information.

    .. note::

        When logging a model, MLflow will automatically save the state of the ``Settings``
        object so that you can use the same settings at inference time. However, please
        note that some information in the ``Settings`` object will not be saved, including:

            - API keys for avoiding key leakage.
            - Function objects which are not serializable.

    Args:
        llama_index_model: A LlamaIndex object to be saved. Supported model types are:

            1. An Index object.
            2. An Engine object e.g. ChatEngine, QueryEngine, Retriever.
            3. A `Workflow <https://docs.llamaindex.ai/en/stable/module_guides/workflow/>`_ object.
            4. A string representing the path to a script contains LlamaIndex model definition
                of the one of the above types.

        path: Local path where the serialized model (as YAML) is to be saved.
        engine_type: Required when saving an Index object to determine the inference interface
            for the index when loaded as a pyfunc model. This field is **not** required when
            saving other LlamaIndex objects. The supported values are as follows:

            - ``"chat"``: load the index as an instance of the LlamaIndex
              `ChatEngine <https://docs.llamaindex.ai/en/stable/module_guides/deploying/chat_engines/>`_.
            - ``"query"``: load the index as an instance of the LlamaIndex
              `QueryEngine <https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/>`_.
            - ``"retriever"``: load the index as an instance of the LlamaIndex
              `Retriever <https://docs.llamaindex.ai/en/stable/module_guides/querying/retriever/>`_.

        model_config: The model configuration to apply when loading the model back with
            ``mlflow.pyfunc.load_model()``. It will be applied in a different way depending on the
            model type and saving method. See the docstring of :func:`log_model` for more details
            and usage examples.

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
    from llama_index.core.indices.base import BaseIndex

    from mlflow.llama_index.serialize_objects import serialize_settings

    # TODO: make this logic cleaner and maybe a util
    with tempfile.TemporaryDirectory() as temp_dir:
        model_or_code_path = _validate_and_prepare_llama_index_model_or_path(
            llama_index_model, temp_dir
        )

        _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

        path = os.path.abspath(path)
        _validate_and_prepare_target_save_path(path)

        if isinstance(model_config, str):
            model_config = _validate_and_get_model_config_from_file(model_config)

        model_code_path = None
        if isinstance(model_or_code_path, str):
            model_code_path = model_or_code_path
            llama_index_model = _load_model_code_path(model_code_path, model_config)
            _validate_and_copy_file_to_directory(model_code_path, path, "code")

            # Warn when user provides `engine_type` argument while saving an engine directly
            if not isinstance(llama_index_model, BaseIndex) and engine_type is not None:
                _logger.warning(
                    "The `engine_type` argument is ignored when saving a non-index object."
                )

        elif isinstance(model_or_code_path, BaseIndex):
            _validate_engine_type(engine_type)
            llama_index_model = model_or_code_path

        elif isinstance(model_or_code_path, _supported_classes()):
            raise MlflowException.invalid_parameter_value(
                "Saving a non-index object is only supported in the 'Model-from-Code' saving mode. "
                "The legacy serialization method is exclusively for saving index objects. Please "
                "pass the path to the script containing the model definition to save a non-index "
                "object. For more information, see "
                "https://www.mlflow.org/docs/latest/model/models-from-code.html",
            )

    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    if mlflow_model is None:
        mlflow_model = Model()
    saved_example = _save_example(mlflow_model, input_example, path)

    if signature is None and saved_example is not None:
        wrapped_model = create_pyfunc_wrapper(llama_index_model, engine_type, model_config)
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

    # Do not save the index/engine object in model-from-code saving mode
    if not isinstance(model_code_path, str) and isinstance(llama_index_model, BaseIndex):
        _save_index(llama_index_model, path)

    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.llama_index",
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
        model_code_path=model_code_path,
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
    llama_index_model,
    artifact_path: Optional[str] = None,
    engine_type: Optional[str] = None,
    model_config: Optional[dict[str, Any]] = None,
    code_paths: Optional[list[str]] = None,
    registered_model_name: Optional[str] = None,
    signature: Optional[ModelSignature] = None,
    input_example: Optional[ModelInputExample] = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements: Optional[Union[list[str], str]] = None,
    extra_pip_requirements: Optional[Union[list[str], str]] = None,
    conda_env=None,
    metadata: Optional[dict[str, Any]] = None,
    prompts: Optional[list[Union[str, Prompt]]] = None,
    name: Optional[str] = None,
    params: Optional[dict[str, Any]] = None,
    tags: Optional[dict[str, Any]] = None,
    model_type: Optional[str] = None,
    step: int = 0,
    model_id: Optional[str] = None,
    **kwargs,
):
    """
    Log a LlamaIndex model as an MLflow artifact for the current run.

    .. attention::

        Saving a non-index object is only supported in the 'Model-from-Code' saving mode.
        Please refer to the `Models From Code Guide <https://www.mlflow.org/docs/latest/model/models-from-code.html>`_
        for more information.

    .. note::

        When logging a model, MLflow will automatically save the state of the ``Settings``
        object so that you can use the same settings at inference time. However, please
        note that some information in the ``Settings`` object will not be saved, including:

            - API keys for avoiding key leakage.
            - Function objects which are not serializable.

    Args:
        llama_index_model: A LlamaIndex object to be saved. Supported model types are:

            1. An Index object.
            2. An Engine object e.g. ChatEngine, QueryEngine, Retriever.
            3. A `Workflow <https://docs.llamaindex.ai/en/stable/module_guides/workflow/>`_ object.
            4. A string representing the path to a script contains LlamaIndex model definition
                of the one of the above types.

        artifact_path: Deprecated. Use `name` instead.
        engine_type: Required when saving an Index object to determine the inference interface
            for the index when loaded as a pyfunc model. This field is **not** required when
            saving other LlamaIndex objects. The supported values are as follows:

            - ``"chat"``: load the index as an instance of the LlamaIndex
              `ChatEngine <https://docs.llamaindex.ai/en/stable/module_guides/deploying/chat_engines/>`_.
            - ``"query"``: load the index as an instance of the LlamaIndex
              `QueryEngine <https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/>`_.
            - ``"retriever"``: load the index as an instance of the LlamaIndex
              `Retriever <https://docs.llamaindex.ai/en/stable/module_guides/querying/retriever/>`_.

        model_config: The model configuration to apply when loading the model back with
            ``mlflow.pyfunc.load_model()``. It will be applied in a different way depending on the
            model type and saving method:

            For in-memory Index objects saved directly, it will be passed as keyword arguments to
            instantiate the LlamaIndex engine with the specified engine type at logging.

            .. code-block:: python

                with mlflow.start_run() as run:
                    model_info = mlflow.llama_index.log_model(
                        index,
                        name="index",
                        engine_type="chat",
                        model_config={"top_k": 10},
                    )

                # When loading back, MLflow will call ``index.as_chat_engine(top_k=10)``
                engine = mlflow.pyfunc.load_model(model_info.model_uri)

            For other model types saved with the `Model-from-Code <https://www.mlflow.org/docs/latest/model/models-from-code.html>`
            method, the config will be accessed via the :py:class`~mlflow.models.ModelConfig`
            object within your model code.

            .. code-block:: python

                with mlflow.start_run() as run:
                    model_info = mlflow.llama_index.log_model(
                        "model.py",
                        name="model",
                        model_config={"qdrant_host": "localhost", "qdrant_port": 6333},
                    )

            model.py:

            .. code-block:: python

                import mlflow
                from llama_index.vector_stores.qdrant import QdrantVectorStore
                import qdrant_client


                # The model configuration is accessible via the ModelConfig singleton
                model_config = mlflow.models.ModelConfig()
                qdrant_host = model_config.get("top_k", 5)
                qdrant_port = model_config.get("qdrant_port", 6333)

                client = qdrant_client.Client(host=qdrant_host, port=qdrant_port)
                vectorstore = QdrantVectorStore(client)

                # the rest of the model definition...

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
        prompts: {{ prompts }}
        name: {{ name }}
        params: {{ params }}
        tags: {{ tags }}
        model_type: {{ model_type }}
        step: {{ step }}
        model_id: {{ model_id }}
        kwargs: Additional arguments for :py:class:`mlflow.models.model.Model`
    """
    return Model.log(
        artifact_path=artifact_path,
        name=name,
        engine_type=engine_type,
        model_config=model_config,
        flavor=mlflow.llama_index,
        registered_model_name=registered_model_name,
        llama_index_model=llama_index_model,
        conda_env=conda_env,
        code_paths=code_paths,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        metadata=metadata,
        prompts=prompts,
        params=params,
        tags=tags,
        model_type=model_type,
        step=step,
        model_id=model_id,
        **kwargs,
    )


def _validate_and_prepare_llama_index_model_or_path(llama_index_model, temp_dir=None):
    if isinstance(llama_index_model, str):
        return _validate_and_get_model_code_path(llama_index_model, temp_dir)

    if not isinstance(llama_index_model, _supported_classes()):
        supported_cls_names = [cls.__name__ for cls in _supported_classes()]
        raise MlflowException.invalid_parameter_value(
            message=f"The provided object of type {type(llama_index_model).__name__} is not "
            "supported. MLflow llama-index flavor only supports saving LlamaIndex objects "
            f"subclassed from one of the following classes: {supported_cls_names}.",
        )

    return llama_index_model


def _save_index(index, path):
    """Serialize the index."""
    index_path = os.path.join(path, _INDEX_PERSIST_FOLDER)
    index.storage_context.persist(persist_dir=index_path)


def _load_llama_model(path, flavor_conf):
    """Load the LlamaIndex index/engine/workflow from either model code or serialized index."""
    from llama_index.core import StorageContext, load_index_from_storage

    _add_code_from_conf_to_system_path(path, flavor_conf)

    # Handle model-from-code
    pyfunc_flavor_conf = _get_flavor_configuration(model_path=path, flavor_name=pyfunc.FLAVOR_NAME)
    if model_code_path := pyfunc_flavor_conf.get(MODEL_CODE_PATH):
        # TODO: The code path saved in the MLModel file is the local absolute path to the code
        # file when it is saved. We should update the relative path in artifact directory.
        model_code_path = os.path.join(path, os.path.basename(model_code_path))

        model_config = pyfunc_flavor_conf.get(MODEL_CONFIG) or flavor_conf.get(MODEL_CONFIG, {})
        if isinstance(model_config, str):
            config_path = os.path.join(path, os.path.basename(model_config))
            model_config = _validate_and_get_model_config_from_file(config_path)

        return _load_model_code_path(model_code_path, model_config)
    else:
        # Use default vector store when loading from the serialized index
        index_path = os.path.join(path, _INDEX_PERSIST_FOLDER)
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        return load_index_from_storage(storage_context)


@experimental
@trace_disabled  # Suppress traces while loading model
def load_model(model_uri, dst_path=None):
    """
    Load a LlamaIndex index/engine/workflow from a local file or a run.

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
    mlflow_model = Model.load(local_model_path)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)

    settings_path = os.path.join(local_model_path, _SETTINGS_FILE)
    # NB: Settings is a singleton and can be loaded via llama_index.core.Settings
    deserialize_settings(settings_path)
    model = _load_llama_model(local_model_path, flavor_conf)
    _update_active_model_id_based_on_mlflow_model(mlflow_model)
    return model


def _load_pyfunc(path, model_config: Optional[dict[str, Any]] = None):
    from mlflow.llama_index.pyfunc_wrapper import create_pyfunc_wrapper

    index = load_model(path)
    flavor_conf = _get_flavor_configuration(model_path=path, flavor_name=FLAVOR_NAME)
    engine_type = flavor_conf.pop(
        "engine_type", None
    )  # Not present when saving an non-index object
    return create_pyfunc_wrapper(index, engine_type, model_config)
