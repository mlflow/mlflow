"""
The ``mlflow.pyfunc.model`` module defines logic for saving and loading custom "python_function"
models with a user-defined ``PythonModel`` subclass.
"""

import inspect
import logging
import os
import shutil
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Generator, Optional, Union

import cloudpickle
import pandas as pd
import yaml

import mlflow.pyfunc
import mlflow.utils
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME, MODEL_CODE_PATH
from mlflow.models.rag_signatures import ChatCompletionRequest, SplitChatMessagesRequest
from mlflow.models.signature import (
    _extract_type_hints,
    _is_context_in_predict_function_signature,
    _TypeHints,
)
from mlflow.models.utils import _load_model_code_path
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.pyfunc.utils import pyfunc
from mlflow.pyfunc.utils.data_validation import (
    _check_func_signature,
    _get_func_info_if_type_hint_supported,
    _wrap_predict_with_pyfunc,
)
from mlflow.pyfunc.utils.input_converter import _hydrate_dataclass
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types.agent import ChatAgentMessage, ChatAgentParams, ChatAgentResponse
from mlflow.types.llm import (
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatMessage,
    ChatParams,
)
from mlflow.types.utils import _is_list_dict_str, _is_list_str
from mlflow.utils.annotations import deprecated, experimental
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
from mlflow.utils.file_utils import TempDir, get_total_file_size, write_to
from mlflow.utils.model_utils import _get_flavor_configuration, _validate_infer_and_copy_code_paths
from mlflow.utils.requirements_utils import _get_pinned_requirement

CONFIG_KEY_ARTIFACTS = "artifacts"
CONFIG_KEY_ARTIFACT_RELATIVE_PATH = "path"
CONFIG_KEY_ARTIFACT_URI = "uri"
CONFIG_KEY_PYTHON_MODEL = "python_model"
CONFIG_KEY_CLOUDPICKLE_VERSION = "cloudpickle_version"
_SAVED_PYTHON_MODEL_SUBPATH = "python_model.pkl"
_DEFAULT_CHAT_MODEL_METADATA_TASK = "agent/v1/chat"
_DEFAULT_CHAT_AGENT_METADATA_TASK = "agent/v2/chat"

_logger = logging.getLogger(__name__)


def get_default_pip_requirements():
    """
    Returns:
        A list of default pip requirements for MLflow Models produced by this flavor. Calls to
        :func:`save_model()` and :func:`log_model()` produce a pip environment that, at minimum,
        contains these requirements.
    """
    return [_get_pinned_requirement("cloudpickle")]


def get_default_conda_env():
    """
    Returns:
        The default Conda environment for MLflow Models produced by calls to
        :func:`save_model() <mlflow.pyfunc.save_model>`
        and :func:`log_model() <mlflow.pyfunc.log_model>` when a user-defined subclass of
        :class:`PythonModel` is provided.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


def _log_warning_if_params_not_in_predict_signature(logger, params):
    if params:
        logger.warning(
            "The underlying model does not support passing additional parameters to the predict"
            f" function. `params` {params} will be ignored."
        )


class PythonModel:
    """
    Represents a generic Python model that evaluates inputs and produces API-compatible outputs.
    By subclassing :class:`~PythonModel`, users can create customized MLflow models with the
    "python_function" ("pyfunc") flavor, leveraging custom inference logic and artifact
    dependencies.
    """

    __metaclass__ = ABCMeta

    def load_context(self, context):
        """
        Loads artifacts from the specified :class:`~PythonModelContext` that can be used by
        :func:`~PythonModel.predict` when evaluating inputs. When loading an MLflow model with
        :func:`~load_model`, this method is called as soon as the :class:`~PythonModel` is
        constructed.

        The same :class:`~PythonModelContext` will also be available during calls to
        :func:`~PythonModel.predict`, but it may be more efficient to override this method
        and load artifacts from the context at model load time.

        Args:
            context: A :class:`~PythonModelContext` instance containing artifacts that the model
                     can use to perform inference.
        """

    @deprecated("predict_type_hints", "2.20.0")
    def _get_type_hints(self):
        return self.predict_type_hints

    @property
    def predict_type_hints(self) -> _TypeHints:
        """
        Internal method to get type hints from the predict function signature.
        """
        if hasattr(self, "_predict_type_hints"):
            return self._predict_type_hints
        if _is_context_in_predict_function_signature(func=self.predict):
            self._predict_type_hints = _extract_type_hints(self.predict, input_arg_index=1)
        else:
            self._predict_type_hints = _extract_type_hints(self.predict, input_arg_index=0)
        return self._predict_type_hints

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        # automatically wrap the predict method with pyfunc to ensure data validation
        # NB: skip wrapping for built-in classes defined in MLflow e.g, ChatModel
        if not cls.__module__.startswith("mlflow."):
            predict_attr = cls.__dict__.get("predict")
            if predict_attr is not None and callable(predict_attr):
                func_info = _get_func_info_if_type_hint_supported(predict_attr)
                setattr(cls, "predict", _wrap_predict_with_pyfunc(predict_attr, func_info))
            predict_stream_attr = cls.__dict__.get("predict_stream")
            if predict_stream_attr is not None and callable(predict_stream_attr):
                _check_func_signature(predict_stream_attr, "predict_stream")

    @abstractmethod
    def predict(self, context, model_input, params: Optional[dict[str, Any]] = None):
        """
        Evaluates a pyfunc-compatible input and produces a pyfunc-compatible output.
        For more information about the pyfunc input/output API, see the :ref:`pyfunc-inference-api`.

        Args:
            context: A :class:`~PythonModelContext` instance containing artifacts that the model
                     can use to perform inference.
            model_input: A pyfunc-compatible input for the model to evaluate.
            params: Additional parameters to pass to the model for inference.

        .. tip::
            Since MLflow 2.20.0, `context` parameter can be removed from `predict` function
            signature if it's not used. `def predict(self, model_input, params=None)` is valid.
        """

    def predict_stream(self, context, model_input, params: Optional[dict[str, Any]] = None):
        """
        Evaluates a pyfunc-compatible input and produces an iterator of output.
        For more information about the pyfunc input API, see the :ref:`pyfunc-inference-api`.

        Args:
            context: A :class:`~PythonModelContext` instance containing artifacts that the model
                     can use to perform inference.
            model_input: A pyfunc-compatible input for the model to evaluate.
            params: Additional parameters to pass to the model for inference.

        .. tip::
            Since MLflow 2.20.0, `context` parameter can be removed from `predict_stream` function
            signature if it's not used.
            `def predict_stream(self, model_input, params=None)` is valid.
        """
        raise NotImplementedError()


class _FunctionPythonModel(PythonModel):
    """
    When a user specifies a ``python_model`` argument that is a function, we wrap the function
    in an instance of this class.
    """

    def __init__(self, func, signature=None):
        self.signature = signature
        # only wrap `func` if @pyfunc is not already applied
        if not getattr(func, "_is_pyfunc", False):
            self.func = pyfunc(func)
        else:
            self.func = func

    @property
    def predict_type_hints(self):
        if hasattr(self, "_predict_type_hints"):
            return self._predict_type_hints
        self._predict_type_hints = _extract_type_hints(self.func, input_arg_index=0)
        return self._predict_type_hints

    def predict(
        self,
        model_input,
        params: Optional[dict[str, Any]] = None,
    ):
        """
        Args:
            model_input: A pyfunc-compatible input for the model to evaluate.
            params: Additional parameters to pass to the model for inference.

        Returns:
            Model predictions.
        """
        # callable only supports one input argument for now
        return self.func(model_input)


class PythonModelContext:
    """
    A collection of artifacts that a :class:`~PythonModel` can use when performing inference.
    :class:`~PythonModelContext` objects are created *implicitly* by the
    :func:`save_model() <mlflow.pyfunc.save_model>` and
    :func:`log_model() <mlflow.pyfunc.log_model>` persistence methods, using the contents specified
    by the ``artifacts`` parameter of these methods.
    """

    def __init__(self, artifacts, model_config):
        """
        Args:
            artifacts: A dictionary of ``<name, artifact_path>`` entries, where ``artifact_path``
                is an absolute filesystem path to a given artifact.
            model_config: The model configuration to make available to the model at
                loading time.
        """
        self._artifacts = artifacts
        self._model_config = model_config

    @property
    def artifacts(self):
        """
        A dictionary containing ``<name, artifact_path>`` entries, where ``artifact_path`` is an
        absolute filesystem path to the artifact.
        """
        return self._artifacts

    @experimental
    @property
    def model_config(self):
        """
        A dictionary containing ``<config, value>`` entries, where ``config`` is the name
        of the model configuration keys and ``value`` is the value of the given configuration.
        """

        return self._model_config


@experimental
class ChatModel(PythonModel, metaclass=ABCMeta):
    """
    A subclass of :class:`~PythonModel` that makes it more convenient to implement models
    that are compatible with popular LLM chat APIs. By subclassing :class:`~ChatModel`,
    users can create MLflow models with a ``predict()`` method that is more convenient
    for chat tasks than the generic :class:`~PythonModel` API. ChatModels automatically
    define input/output signatures and an input example, so manually specifying these values
    when calling :func:`mlflow.pyfunc.save_model() <mlflow.pyfunc.save_model>` is not necessary.

    See the documentation of the ``predict()`` method below for details on that parameters and
    outputs that are expected by the ``ChatModel`` API.
    """

    @abstractmethod
    def predict(
        self, context, messages: list[ChatMessage], params: ChatParams
    ) -> ChatCompletionResponse:
        """
        Evaluates a chat input and produces a chat output.

        Args:
            context: A :class:`~PythonModelContext` instance containing artifacts that the model
                can use to perform inference.
            messages (List[:py:class:`ChatMessage <mlflow.types.llm.ChatMessage>`]):
                A list of :py:class:`ChatMessage <mlflow.types.llm.ChatMessage>`
                objects representing chat history.
            params (:py:class:`ChatParams <mlflow.types.llm.ChatParams>`):
                A :py:class:`ChatParams <mlflow.types.llm.ChatParams>` object
                containing various parameters used to modify model behavior during
                inference.

        .. tip::
            Since MLflow 2.20.0, `context` parameter can be removed from `predict` function
            signature if it's not used.
            `def predict(self, messages: list[ChatMessage], params: ChatParams)` is valid.

        Returns:
            A :py:class:`ChatCompletionResponse <mlflow.types.llm.ChatCompletionResponse>`
            object containing the model's response(s), as well as other metadata.
        """

    def predict_stream(
        self, context, messages: list[ChatMessage], params: ChatParams
    ) -> Generator[ChatCompletionChunk, None, None]:
        """
        Evaluates a chat input and produces a chat output.
        Override this function to implement a real stream prediction.

        Args:
            context: A :class:`~PythonModelContext` instance containing artifacts that the model
                can use to perform inference.
            messages (List[:py:class:`ChatMessage <mlflow.types.llm.ChatMessage>`]):
                A list of :py:class:`ChatMessage <mlflow.types.llm.ChatMessage>`
                objects representing chat history.
            params (:py:class:`ChatParams <mlflow.types.llm.ChatParams>`):
                A :py:class:`ChatParams <mlflow.types.llm.ChatParams>` object
                containing various parameters used to modify model behavior during
                inference.

        .. tip::
            Since MLflow 2.20.0, `context` parameter can be removed from `predict_stream` function
            signature if it's not used.
            `def predict_stream(self, messages: list[ChatMessage], params: ChatParams)` is valid.

        Returns:
            A generator over :py:class:`ChatCompletionChunk <mlflow.types.llm.ChatCompletionChunk>`
            object containing the model's response(s), as well as other metadata.
        """
        raise NotImplementedError(
            "Streaming implementation not provided. Please override the "
            "`predict_stream` method on your model to generate streaming "
            "predictions"
        )


@experimental
class ChatAgent(PythonModel, metaclass=ABCMeta):
    """
    A subclass of :class:`~PythonModel` that makes it more convenient to implement agents
    that are compatible with popular LLM chat APIs. By subclassing :class:`~ChatAgent`,
    users can create MLflow models with a ``predict()`` method that is more convenient
    for chat tasks than the generic :class:`~PythonModel` API. ChatAgents automatically
    define input/output signatures and an input example, so manually specifying these values
    when calling :func:`mlflow.pyfunc.save_model() <mlflow.pyfunc.save_model>` is not necessary.

    See the documentation of the ``predict()`` and ``predict_stream()`` method below for details on
    that parameters and outputs that are expected by the ``ChatAgent`` API.

    Before logging, ``predict()`` and ``predict_stream()`` methods only take in Pydantic models.
    After logging, you can pass in a single dictionary that conforms to a ChatAgentRequest schema.
    Look at CHAT_AGENT_INPUT_EXAMPLE in mlflow.types.agent for an example.
    """

    def _convert_messages_to_dict(self, messages: list[ChatAgentMessage]):
        return [m.model_dump_compat(exclude_none=True) for m in messages]

    # nb: we use `messages` instead of `model_input` so that the trace generated by default is
    # compatible with mlflow evaluate
    @abstractmethod
    def predict(
        self, messages: list[ChatAgentMessage], params: ChatAgentParams
    ) -> ChatAgentResponse:
        """
        Evaluates a ChatAgent input and produces a ChatAgent output.

        Args:
            messages (List[:py:class:`ChatAgentMessage <mlflow.types.llm.ChatAgentMessage>`]):
                A list of :py:class:`ChatAgentMessage <mlflow.types.llm.ChatAgentMessage>`
                objects representing the chat history.
            params (:py:class:`ChatAgentParams <mlflow.types.llm.ChatAgentParams>`):
                A :py:class:`ChatAgentParams <mlflow.types.llm.ChatAgentParams>` object
                containing various parameters used to modify model behavior during
                inference.

        Returns:
            A :py:class:`ChatAgentResponse <mlflow.types.llm.ChatAgentResponse>` object containing
            the model's response(s), as well as other metadata.
        """

    def predict_stream(
        self, messages: list[ChatAgentMessage], params: ChatAgentParams
    ) -> Generator[ChatAgentResponse, None, None]:
        """
        Evaluates a ChatAgent input and produces a ChatAgent output.
        Override this function to implement a real stream prediction.

        Args:
            messages (List[:py:class:`ChatAgentMessage <mlflow.types.llm.ChatAgentMessage>`]):
                A list of :py:class:`ChatAgentMessage <mlflow.types.llm.ChatAgentMessage>`
                objects representing the chat history.
            params (:py:class:`ChatAgentParams <mlflow.types.llm.ChatAgentParams>`):
                A :py:class:`ChatAgentParams <mlflow.types.llm.ChatAgentParams>` object
                containing various parameters used to modify model behavior during
                inference.
        """
        raise NotImplementedError(
            "Streaming implementation not provided. Please override the "
            "`predict_stream` method on your model to generate streaming predictions"
        )


def _save_model_with_class_artifacts_params(  # noqa: D417
    path,
    python_model,
    signature=None,
    artifacts=None,
    conda_env=None,
    code_paths=None,
    mlflow_model=None,
    pip_requirements=None,
    extra_pip_requirements=None,
    model_config=None,
    streamable=None,
    model_code_path=None,
    infer_code_paths=False,
):
    """
    Args:
        path: The path to which to save the Python model.
        python_model: An instance of a subclass of :class:`~PythonModel`. ``python_model``
            defines how the model loads artifacts and how it performs inference.
        artifacts: A dictionary containing ``<name, artifact_uri>`` entries. Remote artifact URIs
            are resolved to absolute filesystem paths, producing a dictionary of
            ``<name, absolute_path>`` entries, (e.g. {"file": "absolute_path"}).
            ``python_model`` can reference these resolved entries as the ``artifacts`` property
            of the ``context`` attribute. If ``<artifact_name, 'hf:/repo_id'>``(e.g.
            {"bert-tiny-model": "hf:/prajjwal1/bert-tiny"}) is provided, then the model can be
            fetched from huggingface hub using repo_id `prajjwal1/bert-tiny` directly. If ``None``,
            no artifacts are added to the model.
        conda_env: Either a dictionary representation of a Conda environment or the path to a Conda
            environment yaml file. If provided, this decsribes the environment this model should be
            run in. At minimum, it should specify the dependencies contained in
            :func:`get_default_conda_env()`. If ``None``, the default
            :func:`get_default_conda_env()` environment is added to the model.
        code_paths: A list of local filesystem paths to Python file dependencies (or directories
            containing file dependencies). These files are *prepended* to the system path before the
            model is loaded.
        mlflow_model: The model to which to add the ``mlflow.pyfunc`` flavor.
        model_config: The model configuration for the flavor. Model configuration is available
            during model loading time.

            .. Note:: Experimental: This parameter may change or be removed in a future release
                without warning.

        model_code_path: The path to the code that is being logged as a PyFunc model. Can be used
            to load python_model when python_model is None.

            .. Note:: Experimental: This parameter may change or be removed in a future release
                without warning.

        streamable: A boolean value indicating if the model supports streaming prediction,
                    If None, MLflow will try to inspect if the model supports streaming
                    by checking if `predict_stream` method exists. Default None.
    """
    if mlflow_model is None:
        mlflow_model = Model()

    custom_model_config_kwargs = {
        CONFIG_KEY_CLOUDPICKLE_VERSION: cloudpickle.__version__,
    }
    if callable(python_model):
        python_model = _FunctionPythonModel(func=python_model, signature=signature)
    saved_python_model_subpath = _SAVED_PYTHON_MODEL_SUBPATH

    # If model_code_path is defined, we load the model into python_model, but we don't want to
    # pickle/save the python_model since the module won't be able to be imported.
    if not model_code_path:
        try:
            with open(os.path.join(path, saved_python_model_subpath), "wb") as out:
                cloudpickle.dump(python_model, out)
        except Exception as e:
            raise MlflowException(
                "Failed to serialize Python model. Please save the model into a python file "
                "and use code-based logging method instead. See"
                "https://mlflow.org/docs/latest/models.html#models-from-code for more information."
            ) from e

        custom_model_config_kwargs[CONFIG_KEY_PYTHON_MODEL] = saved_python_model_subpath

    if artifacts:
        saved_artifacts_config = {}
        with TempDir() as tmp_artifacts_dir:
            saved_artifacts_dir_subpath = "artifacts"
            hf_prefix = "hf:/"
            for artifact_name, artifact_uri in artifacts.items():
                if artifact_uri.startswith(hf_prefix):
                    try:
                        from huggingface_hub import snapshot_download
                    except ImportError as e:
                        raise MlflowException(
                            "Failed to import huggingface_hub. Please install huggingface_hub "
                            f"to log the model with artifact_uri {artifact_uri}. Error: {e}"
                        )

                    repo_id = artifact_uri[len(hf_prefix) :]
                    try:
                        snapshot_location = snapshot_download(
                            repo_id=repo_id,
                            local_dir=os.path.join(
                                path, saved_artifacts_dir_subpath, artifact_name
                            ),
                            local_dir_use_symlinks=False,
                        )
                    except Exception as e:
                        raise MlflowException.invalid_parameter_value(
                            "Failed to download snapshot from Hugging Face Hub with artifact_uri: "
                            f"{artifact_uri}. Error: {e}"
                        )
                    saved_artifact_subpath = (
                        Path(snapshot_location).relative_to(Path(os.path.realpath(path))).as_posix()
                    )
                else:
                    tmp_artifact_path = _download_artifact_from_uri(
                        artifact_uri=artifact_uri, output_path=tmp_artifacts_dir.path()
                    )

                    relative_path = (
                        Path(tmp_artifact_path)
                        .relative_to(Path(tmp_artifacts_dir.path()))
                        .as_posix()
                    )

                    saved_artifact_subpath = os.path.join(
                        saved_artifacts_dir_subpath, relative_path
                    )

                saved_artifacts_config[artifact_name] = {
                    CONFIG_KEY_ARTIFACT_RELATIVE_PATH: saved_artifact_subpath,
                    CONFIG_KEY_ARTIFACT_URI: artifact_uri,
                }

            shutil.move(tmp_artifacts_dir.path(), os.path.join(path, saved_artifacts_dir_subpath))
        custom_model_config_kwargs[CONFIG_KEY_ARTIFACTS] = saved_artifacts_config

    if streamable is None:
        streamable = python_model.__class__.predict_stream != PythonModel.predict_stream

    if model_code_path:
        loader_module = mlflow.pyfunc.loaders.code_model.__name__
    elif python_model:
        loader_module = _get_pyfunc_loader_module(python_model)
    else:
        raise MlflowException(
            "Either `python_model` or `model_code_path` must be provided to save the model.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    mlflow.pyfunc.add_to_model(
        model=mlflow_model,
        loader_module=loader_module,
        code=None,
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        model_config=model_config,
        streamable=streamable,
        model_code_path=model_code_path,
        **custom_model_config_kwargs,
    )
    if size := get_total_file_size(path):
        mlflow_model.model_size_bytes = size
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    saved_code_subpath = _validate_infer_and_copy_code_paths(
        code_paths,
        path,
        infer_code_paths,
        mlflow.pyfunc.FLAVOR_NAME,
    )
    mlflow_model.flavors[mlflow.pyfunc.FLAVOR_NAME][mlflow.pyfunc.CODE] = saved_code_subpath

    # `mlflow_model.code` is updated, re-generate `MLmodel` file.
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements()
            # To ensure `_load_pyfunc` can successfully load the model during the dependency
            # inference, `mlflow_model.save` must be called beforehand to save an MLmodel file.
            inferred_reqs = mlflow.models.infer_pip_requirements(
                path,
                mlflow.pyfunc.FLAVOR_NAME,
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

    with open(os.path.join(path, _CONDA_ENV_FILE_NAME), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    # Save `constraints.txt` if necessary
    if pip_constraints:
        write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))

    # Save `requirements.txt`
    write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))

    _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))


def _load_context_model_and_signature(
    model_path: str, model_config: Optional[dict[str, Any]] = None
):
    pyfunc_config = _get_flavor_configuration(
        model_path=model_path, flavor_name=mlflow.pyfunc.FLAVOR_NAME
    )
    signature = mlflow.models.Model.load(model_path).signature

    if MODEL_CODE_PATH in pyfunc_config:
        conf_model_code_path = pyfunc_config.get(MODEL_CODE_PATH)
        model_code_path = os.path.join(model_path, os.path.basename(conf_model_code_path))
        python_model = _load_model_code_path(model_code_path, model_config)

        if callable(python_model):
            python_model = _FunctionPythonModel(python_model, signature=signature)
    else:
        python_model_cloudpickle_version = pyfunc_config.get(CONFIG_KEY_CLOUDPICKLE_VERSION, None)
        if python_model_cloudpickle_version is None:
            mlflow.pyfunc._logger.warning(
                "The version of CloudPickle used to save the model could not be found in the "
                "MLmodel configuration"
            )
        elif python_model_cloudpickle_version != cloudpickle.__version__:
            # CloudPickle does not have a well-defined cross-version compatibility policy. Micro
            # version releases have been known to cause incompatibilities. Therefore, we match on
            # the full library version
            mlflow.pyfunc._logger.warning(
                "The version of CloudPickle that was used to save the model, `CloudPickle %s`, "
                "differs from the version of CloudPickle that is currently running, `CloudPickle "
                "%s`, and may be incompatible",
                python_model_cloudpickle_version,
                cloudpickle.__version__,
            )

        python_model_subpath = pyfunc_config.get(CONFIG_KEY_PYTHON_MODEL, None)
        if python_model_subpath is None:
            raise MlflowException("Python model path was not specified in the model configuration")
        with open(os.path.join(model_path, python_model_subpath), "rb") as f:
            python_model = cloudpickle.load(f)

    artifacts = {}
    for saved_artifact_name, saved_artifact_info in pyfunc_config.get(
        CONFIG_KEY_ARTIFACTS, {}
    ).items():
        artifacts[saved_artifact_name] = os.path.join(
            model_path, saved_artifact_info[CONFIG_KEY_ARTIFACT_RELATIVE_PATH]
        )

    context = PythonModelContext(artifacts=artifacts, model_config=model_config)
    python_model.load_context(context=context)

    return context, python_model, signature


def _load_pyfunc(model_path: str, model_config: Optional[dict[str, Any]] = None):
    context, python_model, signature = _load_context_model_and_signature(model_path, model_config)
    return _PythonModelPyfuncWrapper(
        python_model=python_model,
        context=context,
        signature=signature,
    )


def _get_first_string_column(pdf):
    iter_string_columns = (col for col, val in pdf.iloc[0].items() if isinstance(val, str))
    return next(iter_string_columns, None)


class _PythonModelPyfuncWrapper:
    """
    Wrapper class that creates a predict function such that
    predict(model_input: pd.DataFrame) -> model's output as pd.DataFrame (pandas DataFrame)
    """

    def __init__(self, python_model: PythonModel, context, signature):
        """
        Args:
            python_model: An instance of a subclass of :class:`~PythonModel`.
            context: A :class:`~PythonModelContext` instance containing artifacts that
                     ``python_model`` may use when performing inference.
            signature: :class:`~ModelSignature` instance describing model input and output.
        """
        self.python_model = python_model
        self.context = context
        self.signature = signature

    def _convert_input(self, model_input):
        import pandas as pd

        hints = self.python_model.predict_type_hints
        # we still need this for backwards compatibility
        if isinstance(model_input, pd.DataFrame):
            if _is_list_str(hints.input):
                first_string_column = _get_first_string_column(model_input)
                if first_string_column is None:
                    raise MlflowException.invalid_parameter_value(
                        "Expected model input to contain at least one string column"
                    )
                return model_input[first_string_column].tolist()
            elif _is_list_dict_str(hints.input):
                if (
                    len(self.signature.inputs) == 1
                    and next(iter(self.signature.inputs)).name is None
                ):
                    first_string_column = _get_first_string_column(model_input)
                    return model_input[[first_string_column]].to_dict(orient="records")
                return model_input.to_dict(orient="records")
            elif isinstance(hints.input, type) and (
                issubclass(hints.input, ChatCompletionRequest)
                or issubclass(hints.input, SplitChatMessagesRequest)
            ):
                # If the type hint is a RAG dataclass, we hydrate it
                # If there are multiple rows, we should throw
                if len(model_input) > 1:
                    raise MlflowException(
                        "Expected a single input for dataclass type hint, but got multiple rows"
                    )
                # Since single input is expected, we take the first row
                return _hydrate_dataclass(hints.input, model_input.iloc[0])
        return model_input

    def predict(self, model_input, params: Optional[dict[str, Any]] = None):
        """
        Args:
            model_input: Model input data as one of dict, str, bool, bytes, float, int, str type.
            params: Additional parameters to pass to the model for inference.

        Returns:
            Model predictions as an iterator of chunks. The chunks in the iterator must be type of
            dict or string. Chunk dict fields are determined by the model implementation.
        """
        parameters = inspect.signature(self.python_model.predict).parameters
        kwargs = {}
        if "params" in parameters:
            kwargs["params"] = params
        else:
            _log_warning_if_params_not_in_predict_signature(_logger, params)
        if _is_context_in_predict_function_signature(parameters=parameters):
            return self.python_model.predict(
                self.context, self._convert_input(model_input), **kwargs
            )
        else:
            return self.python_model.predict(self._convert_input(model_input), **kwargs)

    def predict_stream(self, model_input, params: Optional[dict[str, Any]] = None):
        """
        Args:
            model_input: LLM Model single input.
            params: Additional parameters to pass to the model for inference.

        Returns:
            Streaming predictions.
        """
        parameters = inspect.signature(self.python_model.predict_stream).parameters
        kwargs = {}
        if "params" in parameters:
            kwargs["params"] = params
        else:
            _log_warning_if_params_not_in_predict_signature(_logger, params)
        if _is_context_in_predict_function_signature(parameters=parameters):
            return self.python_model.predict_stream(
                self.context, self._convert_input(model_input), **kwargs
            )
        else:
            return self.python_model.predict_stream(self._convert_input(model_input), **kwargs)


def _get_pyfunc_loader_module(python_model):
    if isinstance(python_model, ChatModel):
        return mlflow.pyfunc.loaders.chat_model.__name__
    elif isinstance(python_model, ChatAgent):
        return mlflow.pyfunc.loaders.chat_agent.__name__
    return __name__


class ModelFromDeploymentEndpoint(PythonModel):
    """
    A PythonModel wrapper for invoking an MLflow Deployments endpoint.
    This class is particularly used for running evaluation against an MLflow Deployments endpoint.
    """

    def __init__(self, endpoint, params):
        self.endpoint = endpoint
        self.params = params

    def predict(
        self, context, model_input: Union[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]
    ):
        """
        Run prediction on the input data.

        Args:
            context: A :class:`~PythonModelContext` instance containing artifacts that the model
                can use to perform inference.
            model_input: The input data for prediction, either of the following:
                - Pandas DataFrame: If the default evaluator is used, input is a DF
                    that contains the multiple request payloads in a single column.
                - A dictionary: If the model_type is "databricks-agents" and the
                    Databricks RAG evaluator is used, this PythonModel can be invoked
                    with a single dict corresponding to the ChatCompletionsRequest schema.
                - A list of dictionaries: Currently we don't have any evaluator that
                    gives this input format, but we keep this for future use cases and
                    compatibility with normal pyfunc models.

        Return:
            The prediction result. The return type will be consistent with the model input type,
            e.g., if the input is a Pandas DataFrame, the return will be a Pandas Series.
        """
        if isinstance(model_input, dict):
            return self._predict_single(model_input)
        elif isinstance(model_input, list) and all(isinstance(data, dict) for data in model_input):
            return [self._predict_single(data) for data in model_input]
        elif isinstance(model_input, pd.DataFrame):
            if len(model_input.columns) != 1:
                raise MlflowException(
                    f"The number of input columns must be 1, but got {model_input.columns}. "
                    "Multi-column input is not supported for evaluating an MLflow Deployments "
                    "endpoint. Please include the input text or payload in a single column.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            input_column = model_input.columns[0]

            predictions = [self._predict_single(data) for data in model_input[input_column]]
            return pd.Series(predictions)
        else:
            raise MlflowException(
                f"Invalid input data type: {type(model_input)}. The input data must be either "
                "a Pandas DataFrame, a dictionary, or a list of dictionaries containing the "
                "request payloads for evaluating an MLflow Deployments endpoint.",
                error_code=INVALID_PARAMETER_VALUE,
            )

    def _predict_single(self, data: Union[str, dict[str, Any]]) -> dict[str, Any]:
        """
        Send a single prediction request to the MLflow Deployments endpoint.

        Args:
            data: The single input data for prediction. If the input data is a string, we will
                construct the request payload from it. If the input data is a dictionary, we
                will directly use it as the request payload.

        Returns:
            The prediction result from the MLflow Deployments endpoint as a dictionary.
        """
        from mlflow.metrics.genai.model_utils import call_deployments_api, get_endpoint_type

        endpoint_type = get_endpoint_type(f"endpoints:/{self.endpoint}")

        if isinstance(data, str):
            # If the input payload is string, MLflow needs to construct the JSON
            # payload based on the endpoint type. If the endpoint type is not
            # set on the endpoint, we will default to chat format.
            endpoint_type = endpoint_type or "llm/v1/chat"
            prediction = call_deployments_api(self.endpoint, data, self.params, endpoint_type)
        elif isinstance(data, dict):
            # If the input is dictionary, we assume the input is already in the
            # compatible format for the endpoint.
            prediction = call_deployments_api(self.endpoint, data, self.params, endpoint_type)
        else:
            raise MlflowException(
                f"Invalid input data type: {type(data)}. The feature column of the evaluation "
                "dataset must contain only strings or dictionaries containing the request "
                "payload for evaluating an MLflow Deployments endpoint.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        return prediction
