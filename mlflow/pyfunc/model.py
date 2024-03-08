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
from typing import Any, Dict, List, Optional

import cloudpickle
import yaml

import mlflow.pyfunc
import mlflow.utils
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _extract_type_hints
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types.llm import ChatMessage, ChatParams, ChatResponse
from mlflow.utils.annotations import experimental
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
from mlflow.utils.model_utils import _get_flavor_configuration, _validate_and_copy_code_paths
from mlflow.utils.requirements_utils import _get_pinned_requirement

CONFIG_KEY_ARTIFACTS = "artifacts"
CONFIG_KEY_ARTIFACT_RELATIVE_PATH = "path"
CONFIG_KEY_ARTIFACT_URI = "uri"
CONFIG_KEY_PYTHON_MODEL = "python_model"
CONFIG_KEY_CLOUDPICKLE_VERSION = "cloudpickle_version"
_SAVED_PYTHON_MODEL_SUBPATH = "python_model.pkl"


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

    def _get_type_hints(self):
        return _extract_type_hints(self.predict, input_arg_index=1)

    @abstractmethod
    def predict(self, context, model_input, params: Optional[Dict[str, Any]] = None):
        """
        Evaluates a pyfunc-compatible input and produces a pyfunc-compatible output.
        For more information about the pyfunc input/output API, see the :ref:`pyfunc-inference-api`.

        Args:
            context: A :class:`~PythonModelContext` instance containing artifacts that the model
                     can use to perform inference.
            model_input: A pyfunc-compatible input for the model to evaluate.
            params: Additional parameters to pass to the model for inference.

                     .. Note:: Experimental: This parameter may change or be removed in a future
                                 release without warning.
        """


class _FunctionPythonModel(PythonModel):
    """
    When a user specifies a ``python_model`` argument that is a function, we wrap the function
    in an instance of this class.
    """

    def __init__(self, func, hints=None, signature=None):
        self.func = func
        self.hints = hints
        self.signature = signature

    def _get_type_hints(self):
        return _extract_type_hints(self.func, input_arg_index=0)

    def predict(
        self,
        context,
        model_input,
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            context: A instance containing artifacts that the model
                can use to perform inference.
            model_input: A pyfunc-compatible input for the model to evaluate.
            params: Additional parameters to pass to the model for inference.

                .. Note:: Experimental: This parameter may change or be removed in a future
                    release without warning.

        Returns:
            Model predictions.
        """
        if inspect.signature(self.func).parameters.get("params"):
            return self.func(model_input, params=params)
        _log_warning_if_params_not_in_predict_signature(_logger, params)
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
    def predict(self, context, messages: List[ChatMessage], params: ChatParams) -> ChatResponse:
        """
        Evaluates a chat input and produces a chat output.

        Args:
            messages (List[:py:class:`ChatMessage <mlflow.types.llm.ChatMessage>`]):
                A list of :py:class:`ChatMessage <mlflow.types.llm.ChatMessage>`
                objects representing chat history.
            params (:py:class:`ChatParams <mlflow.types.llm.ChatParams>`):
                A :py:class:`ChatParams <mlflow.types.llm.ChatParams>` object
                containing various parameters used to modify model behavior during
                inference.

        Returns:
            A :py:class:`ChatResponse <mlflow.types.llm.ChatResponse>` object containing
            the model's response(s), as well as other metadata.
        """


def _save_model_with_class_artifacts_params(
    path,
    python_model,
    signature=None,
    hints=None,
    artifacts=None,
    conda_env=None,
    code_paths=None,
    mlflow_model=None,
    pip_requirements=None,
    extra_pip_requirements=None,
    model_config=None,
):
    """
    Args:
        path: The path to which to save the Python model.
        python_model: An instance of a subclass of :class:`~PythonModel`. ``python_model``
            defines how the model loads artifacts and how it performs inference.
        artifacts: A dictionary containing ``<name, artifact_uri>`` entries. Remote artifact URIs
            are resolved to absolute filesystem paths, producing a dictionary of
            ``<name, absolute_path>`` entries, (e.g. {"file": "aboslute_path"}).
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
    """
    if mlflow_model is None:
        mlflow_model = Model()

    custom_model_config_kwargs = {
        CONFIG_KEY_CLOUDPICKLE_VERSION: cloudpickle.__version__,
    }
    if callable(python_model):
        python_model = _FunctionPythonModel(python_model, hints, signature)
    saved_python_model_subpath = _SAVED_PYTHON_MODEL_SUBPATH

    try:
        with open(os.path.join(path, saved_python_model_subpath), "wb") as out:
            cloudpickle.dump(python_model, out)
    except Exception as e:
        # cloudpickle sometimes raises TypeError instead of PicklingError.
        # catching generic Exception and checking message to handle both cases.
        if "cannot pickle" in str(e).lower():
            raise MlflowException(
                "Failed to serialize Python model. Please audit your "
                "class variables (e.g. in `__init__()`) for any "
                "unpicklable objects. If you're trying to save an external model "
                "in your custom pyfunc, Please use the `artifacts` parameter "
                "in `mlflow.pyfunc.save_model()`, and load your external model "
                "in the `load_context()` method instead. For example:\n\n"
                "class MyModel(mlflow.pyfunc.PythonModel):\n"
                "    def load_context(self, context):\n"
                "        model_path = context.artifacts['my_model_path']\n"
                "        // custom load logic here\n"
                "        self.model = load_model(model_path)\n\n"
                "For more information, see our full tutorial at: "
                "https://mlflow.org/docs/latest/traditional-ml/creating-custom-pyfunc/index.html"
                f"\n\nFull serialization error: {e}"
            ) from None
        else:
            raise e

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

    saved_code_subpath = _validate_and_copy_code_paths(code_paths, path)

    mlflow.pyfunc.add_to_model(
        model=mlflow_model,
        loader_module=_get_pyfunc_loader_module(python_model),
        code=saved_code_subpath,
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        model_config=model_config,
        **custom_model_config_kwargs,
    )
    if size := get_total_file_size(path):
        mlflow_model.model_size_bytes = size
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
    model_path: str, model_config: Optional[Dict[str, Any]] = None
):
    pyfunc_config = _get_flavor_configuration(
        model_path=model_path, flavor_name=mlflow.pyfunc.FLAVOR_NAME
    )

    python_model_cloudpickle_version = pyfunc_config.get(CONFIG_KEY_CLOUDPICKLE_VERSION, None)
    if python_model_cloudpickle_version is None:
        mlflow.pyfunc._logger.warning(
            "The version of CloudPickle used to save the model could not be found in the MLmodel"
            " configuration"
        )
    elif python_model_cloudpickle_version != cloudpickle.__version__:
        # CloudPickle does not have a well-defined cross-version compatibility policy. Micro version
        # releases have been known to cause incompatibilities. Therefore, we match on the full
        # library version
        mlflow.pyfunc._logger.warning(
            "The version of CloudPickle that was used to save the model, `CloudPickle %s`, differs"
            " from the version of CloudPickle that is currently running, `CloudPickle %s`, and may"
            " be incompatible",
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
    signature = mlflow.models.Model.load(model_path).signature

    return context, python_model, signature


def _load_pyfunc(model_path: str, model_config: Optional[Dict[str, Any]] = None):
    context, python_model, signature = _load_context_model_and_signature(model_path, model_config)
    return _PythonModelPyfuncWrapper(
        python_model=python_model, context=context, signature=signature
    )


def _get_first_string_column(pdf):
    iter_string_columns = (col for col, val in pdf.iloc[0].items() if isinstance(val, str))
    return next(iter_string_columns, None)


class _PythonModelPyfuncWrapper:
    """
    Wrapper class that creates a predict function such that
    predict(model_input: pd.DataFrame) -> model's output as pd.DataFrame (pandas DataFrame)
    """

    def __init__(self, python_model, context, signature):
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

        hints = self.python_model._get_type_hints()
        if hints.input == List[str]:
            if isinstance(model_input, pd.DataFrame):
                first_string_column = _get_first_string_column(model_input)
                if first_string_column is None:
                    raise MlflowException.invalid_parameter_value(
                        "Expected model input to contain at least one string column"
                    )
                return model_input[first_string_column].tolist()
            elif isinstance(model_input, list):
                if all(isinstance(x, dict) for x in model_input):
                    return [next(iter(d.values())) for d in model_input]
                elif all(isinstance(x, str) for x in model_input):
                    return model_input
        elif hints.input == List[Dict[str, str]]:
            if isinstance(model_input, pd.DataFrame):
                if (
                    len(self.signature.inputs) == 1
                    and next(iter(self.signature.inputs)).name is None
                ):
                    first_string_column = _get_first_string_column(model_input)
                    return model_input[[first_string_column]].to_dict(orient="records")
                columns = [x.name for x in self.signature.inputs]
                return model_input[columns].to_dict(orient="records")
            elif isinstance(model_input, list) and all(isinstance(x, dict) for x in model_input):
                keys = [x.name for x in self.signature.inputs]
                return [{k: d[k] for k in keys} for d in model_input]

        return model_input

    def predict(self, model_input, params: Optional[Dict[str, Any]] = None):
        """
        Args:
            model_input: Model input data.
            params: Additional parameters to pass to the model for inference.

                .. Note:: Experimental: This parameter may change or be removed in a future
                    release without warning.

        Returns:
            Model predictions.
        """
        if inspect.signature(self.python_model.predict).parameters.get("params"):
            return self.python_model.predict(
                self.context, self._convert_input(model_input), params=params
            )
        _log_warning_if_params_not_in_predict_signature(_logger, params)
        return self.python_model.predict(self.context, self._convert_input(model_input))


def _get_pyfunc_loader_module(python_model):
    if isinstance(python_model, ChatModel):
        return mlflow.pyfunc.loaders.chat_model.__name__
    return __name__
