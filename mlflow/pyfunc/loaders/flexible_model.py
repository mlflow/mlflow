
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
from mlflow.models.model import MLMODEL_FILE_NAME, MODEL_CODE_PATH
from mlflow.models.signature import _extract_type_hints
from mlflow.models.utils import _load_model_code_path
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
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
from mlflow.utils.model_utils import _get_flavor_configuration, _validate_infer_and_copy_code_paths
from mlflow.utils.requirements_utils import _get_pinned_requirement

CONFIG_KEY_ARTIFACTS = "artifacts"
CONFIG_KEY_ARTIFACT_RELATIVE_PATH = "path"
CONFIG_KEY_ARTIFACT_URI = "uri"
CONFIG_KEY_PYTHON_MODEL = "python_model"
CONFIG_KEY_CLOUDPICKLE_VERSION = "cloudpickle_version"
_SAVED_PYTHON_MODEL_SUBPATH = "python_model.pkl"


_logger = logging.getLogger(__name__)

class _FlexibleModelPyfuncWrapper:
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
            model_input: Model input data as one of dict, str, bool, bytes, float, int, str type.
            params: Additional parameters to pass to the model for inference.

        Returns:
            Model predictions as an iterator of chunks. The chunks in the iterator must be type of
            dict or string. Chunk dict fields are determined by the model implementation.
        """
        if inspect.signature(self.python_model.predict).parameters.get("params"):
            return self.python_model.predict(
                self.context, self._convert_input(model_input), params=params
            )
        _log_warning_if_params_not_in_predict_signature(_logger, params)
        return self.python_model.predict(self.context, self._convert_input(model_input))

    def predict_stream(self, model_input, params: Optional[Dict[str, Any]] = None):
        """
        Args:
            model_input: LLM Model single input.
            params: Additional parameters to pass to the model for inference.

        Returns:
            Streaming predictions.
        """
        if inspect.signature(self.python_model.predict_stream).parameters.get("params"):
            return self.python_model.predict_stream(
                self.context, self._convert_input(model_input), params=params
            )
        _log_warning_if_params_not_in_predict_signature(_logger, params)
        return self.python_model.predict_stream(self.context, self._convert_input(model_input))