import logging
import os
from typing import Any, Dict, Optional

import yaml
from llama_index.core import StorageContext, load_index_from_storage

import mlflow
import mlflow.exceptions
from mlflow import pyfunc
from mlflow.llama_index.serialize_objects import (
    deserialize_json_to_engine_kwargs,
    deserialize_json_to_settings,
    serialize_engine_kwargs_to_json,
    serialize_settings_to_json,
)
from mlflow.llama_index.signature import (
    infer_signature_from_input_example,
    validate_and_resolve_signature,
)
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
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

# TODO: remove and put in another PR
_CHAT_ENGINE_NAME = "chat"
_QUERY_ENGINE_NAME = "query"
_RETRIEVER_ENGINE_NAME = "retriever"

_SUPPORTED_ENGINES = {_CHAT_ENGINE_NAME, _QUERY_ENGINE_NAME, _RETRIEVER_ENGINE_NAME}

FLAVOR_NAME = "llama_index"
_INDEX_PERSIST_FOLDER = "index"
_SETTINGS_DIRECTORY = "settings"
_ENGINE_KWARGS_DIRECTORY = "engine_kwargs"


_logger = logging.getLogger(__name__)


def get_default_pip_requirements():
    """
    Returns:
        A list of default pip requirements for MLflow Models produced by this flavor.
        Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
        that, at a minimum, contains these requirements.
    """
    return [_get_pinned_requirement("llama_index")]


def get_default_conda_env():
    """
    Returns:
        The default Conda environment for MLflow Models produced by calls to
        :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


def _should_add_pyfunc_to_model(index):
    """
    PLACEHOLDER: validate whether the desired index being logged should be supported by pyfunc
    """
    return True


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    index,
    path: str,
    engine_type: str,
    engine_kwargs: Optional[Dict[str, any]] = None,
    conda_env=None,
    code_paths=None,
    mlflow_model=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    model_config=None,
    metadata=None,
):
    """ """
    import llama_index

    if engine_type not in _SUPPORTED_ENGINES:
        raise ValueError(
            f"Currently mlflow only supports the following engine types: "
            f"{_SUPPORTED_ENGINES}. {engine_type} is not supported, so please"
            "use one of the above types"
        )

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    path = os.path.abspath(path)
    _validate_and_prepare_target_save_path(path)
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    if signature is None and input_example is not None:
        wrapped_model = _LlamaIndexModelWrapper(index)
        signature = infer_signature_from_input_example(input_example, wrapped_model)
    elif signature is False:
        signature = None

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        signature = validate_and_resolve_signature(index, signature)
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)
    if metadata is not None:
        mlflow_model.metadata = metadata

    engine_kwargs_path = os.path.join(path, _ENGINE_KWARGS_DIRECTORY)
    serialize_engine_kwargs_to_json(engine_kwargs, engine_kwargs_path)

    from llama_index.core import Settings

    settings_path = os.path.join(path, _SETTINGS_DIRECTORY)
    serialize_settings_to_json(Settings, settings_path)

    model_data_path = path
    _save_model(index, model_data_path)

    flavor_conf = {"engine_type": engine_type}
    if _should_add_pyfunc_to_model(index):
        if mlflow_model.signature is None:
            mlflow_model.signature = infer_signature_from_input_example(
                index=index,
                example=input_example,
                flavor_config=flavor_conf,
            )

    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.llama_index",
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
    )
    mlflow_model.add_flavor(
        FLAVOR_NAME,
        llama_index_version=llama_index.core.__version__,
        code=code_dir_subpath,
        **flavor_conf,
    )
    if size := get_total_file_size(path):
        mlflow_model.model_size_bytes = size
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if conda_env is None:
        default_reqs = None
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements()
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


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    index,
    artifact_path,
    engine_type: str,
    engine_kwargs: Optional[Dict[str, any]] = None,
    conda_env=None,
    code_paths=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
):
    """ """

    return Model.log(
        artifact_path=artifact_path,
        engine_type=engine_type,
        engine_kwargs=engine_kwargs,
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
    )


def _save_model(index, path):
    """Serialize settings, additional llama index objects, and the index."""
    index_path = os.path.join(path, _INDEX_PERSIST_FOLDER)
    index.storage_context.persist(persist_dir=index_path)


def _load_model(path, flavor_conf):
    """Deserialize settings, additional llama index objects, and the index."""
    _add_code_from_conf_to_system_path(path, flavor_conf)

    index_path = os.path.join(path, _INDEX_PERSIST_FOLDER)
    storage_context = StorageContext.from_defaults(persist_dir=index_path)
    return load_index_from_storage(storage_context)


def load_model(model_uri, dst_path=None):
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)

    settings_path = os.path.join(local_model_path, _SETTINGS_DIRECTORY)
    # Settings is a singleton state is returned on import
    _ = deserialize_json_to_settings(settings_path)

    return _load_model(local_model_path, flavor_conf)


def _load_pyfunc(path):
    engine_kwargs_path = os.path.join(path, _ENGINE_KWARGS_DIRECTORY)
    engine_kwargs = deserialize_json_to_engine_kwargs(engine_kwargs_path)

    flavor_conf = _get_flavor_configuration(model_path=path, flavor_name=FLAVOR_NAME)
    engine_type = flavor_conf["engine_type"]
    return _LlamaIndexModelWrapper(_load_model(path, flavor_conf), engine_type, engine_kwargs)


class _LlamaIndexModelWrapper:
    def __init__(self, index, engine_type, engine_kwargs):
        """
        PLACEHOLDER
        - Convert to a builder class
        """
        _engine_to_instantiation_method = {
            _CHAT_ENGINE_NAME: "as_chat_engine",
            _QUERY_ENGINE_NAME: "as_query_engine",
            _RETRIEVER_ENGINE_NAME: "as_retriever",
        }
        _engine_to_interaction_method = {
            _CHAT_ENGINE_NAME: "chat",
            _QUERY_ENGINE_NAME: "query",
            _RETRIEVER_ENGINE_NAME: "retrieve",
        }

        engine_method_name = _engine_to_instantiation_method[engine_type]
        engine_method = getattr(index, engine_method_name)
        engine = engine_method(**engine_kwargs)

        engine_interaction_method = _engine_to_interaction_method[engine_type]
        self.predict_method = getattr(engine, engine_interaction_method)

    def predict(self, data, params: Optional[Dict[str, Any]] = None):
        """
        PLACEHOLDER
        """
        data = "asdf"
        return self.predict_method(data)
