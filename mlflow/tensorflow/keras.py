import importlib
import os
import re
import yaml
import tempfile
import shutil
import warnings

import pandas as pd

from packaging.version import Version
from mlflow import pyfunc
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
import mlflow.tracking
from mlflow.exceptions import MlflowException
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import ModelInputExample, _save_example
from mlflow.utils.environment import (
    _validate_env_arguments,
    _process_pip_requirements,
    _process_conda_env,
    _CONDA_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _PythonEnv,
)
from mlflow.utils.file_utils import write_to
from mlflow.utils.model_utils import (
    _validate_and_copy_code_paths,
    _add_code_from_conf_to_system_path,
    _validate_and_prepare_target_save_path,
)
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS

# File name to which custom objects cloudpickle is saved - used during save and load
_CUSTOM_OBJECTS_SAVE_PATH = "custom_objects.cloudpickle"
_KERAS_MODULE_SPEC_PATH = "keras_module.txt"
_KERAS_SAVE_FORMAT_PATH = "save_format.txt"
# File name to which keras model is saved
_MODEL_SAVE_PATH = "model"
_PIP_ENV_SUBPATH = "requirements.txt"






def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_model``.

    :param path: Local filesystem path to the MLflow Model with the ``keras`` flavor.
    """
    if os.path.isfile(os.path.join(path, _KERAS_MODULE_SPEC_PATH)):
        with open(os.path.join(path, _KERAS_MODULE_SPEC_PATH), "r") as f:
            keras_module = importlib.import_module(f.read())
    else:
        import tensorflow.keras

        keras_module = tensorflow.keras

    # By default, we assume the save_format is h5 for backwards compatibility
    save_format = "h5"
    save_format_path = os.path.join(path, _KERAS_SAVE_FORMAT_PATH)
    if os.path.isfile(save_format_path):
        with open(save_format_path, "r") as f:
            save_format = f.read()

    # In SavedModel format, if we don't compile the model
    should_compile = save_format == "tf"
    K = importlib.import_module(keras_module.__name__ + ".backend")
    if K.backend() == "tensorflow":
        K.set_learning_phase(0)
        m = _load_keras_model(
            path, keras_module=keras_module, save_format=save_format, compile=should_compile
        )
        return _KerasModelWrapper(m, None, None)
    else:
        raise MlflowException("Unsupported backend '%s'" % K._BACKEND)
