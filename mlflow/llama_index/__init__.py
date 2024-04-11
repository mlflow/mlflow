import logging
import os
from typing import Any, Dict, Optional, Union

import pandas as pd
import yaml
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.schema import QueryBundle, QueryType

import mlflow
from mlflow import pyfunc
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

_SUPPORTED_ENGINES = {"chat", "query", "retriever"}

FLAVOR_NAME = "llama_index"
_INDEX_PERSIST_FOLDER = "index"


# model_data_artifact_paths = [_MODEL_BINARY_FILE_NAME]

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

    model_data_path = path  # os.path.join(path, _INDEX_PERSIST_FOLDER)
    _save_model(index, model_data_path)

    flavor_conf = {}
    if _should_add_pyfunc_to_model(index):
        if mlflow_model.signature is None:
            mlflow_model.signature = infer_signature_from_input_example(
                index=index,
                example=input_example,
                model_config=model_config,
                flavor_config=flavor_conf,
            )

    # model_bin_kwargs = {_MODEL_BINARY_KEY: _MODEL_BINARY_FILE_NAME}
    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.llama_index",
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
        # **model_bin_kwargs,
    )
    mlflow_model.add_flavor(
        FLAVOR_NAME,
        llama_index_version=llama_index.core.__version__,
        code=code_dir_subpath,
        # **flavor_conf,
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
    index_path = os.path.join(path, _INDEX_PERSIST_FOLDER)
    index.storage_context.persist(persist_dir=index_path)


def _load_model(path):
    index_path = os.path.join(path, _INDEX_PERSIST_FOLDER)
    storage_context = StorageContext.from_defaults(persist_dir=index_path)
    return load_index_from_storage(storage_context)


def _load_pyfunc(path):
    return _LlamaIndexModelWrapper(_load_model(path))


def load_model(model_uri, dst_path=None):
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)

    return _load_model(local_model_path)


class _LlamaIndexModelWrapper:
    def __init__(self, index):
        self.index = index

    def _unwrap_query_or_retriever_data(self, data) -> Union[QueryType]:
        if isinstance(data, QueryBundle):
            return data

        elif isinstance(data, pd.DataFrame):
            return str(data.iloc[0, 0])

    def _unwrap_chat_data(self, data) -> str:
        """
        The chat() method takes the following parameters:
        - message: str
        - chat_history: Optional[List[ChatMessage]]
        """
        # TODO: figure out good way to map from all data input types to message and chat_history
        if isinstance(data, pd.DataFrame):
            if "message" in data.colums:
                message = data["message"].iloc[0]
            else:
                raise ValueError("DataFrame must have a column named 'message'")

            if "chat_history" in data.columns:
                chat_history = data["chat_history"].iloc[0]  # assumes a list in this column
            else:
                raise ValueError("DataFrame must have a column named 'chat_history'")

            return message, chat_history

    def _is_supported_engine(self, engine: str):
        return engine in _SUPPORTED_ENGINES

    def predict(self, data: Union[QueryType], params: Optional[Dict[str, Any]] = None):
        engine = params.pop("engine")
        if not self._is_supported_engine(engine):
            raise ValueError(
                f"Engine {engine} is not supported. Supported engines are: {_SUPPORTED_ENGINES}"
            )

        if engine in ("query", "retrieve"):
            query = self._unwrap_query_or_retriever_data(data)
            return self.index.as_query_engine(**params).query(query)
        elif engine == "chat":
            message, chat_history = self._unwrap_chat_data(data)
            return self.index.as_chat_engine(**params).chat(message, chat_history)
