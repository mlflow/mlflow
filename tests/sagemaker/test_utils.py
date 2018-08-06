import json
import os
import pickle

from mlflow.utils import PYTHON_VERSION

RESPONSE_KEY_CONDA_ENV_NAME = "CONDA_ENV_NAME"
RESPONSE_KEY_PYTHON_VERSION = "PYTHON_VERSION"
DEFAULT_ENV_NAME = "DEFAULT_ENV"


class DockerTestPredictor:

    def __init__(self, func):
        self.func = func
    
    def predict(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def load_pyfunc(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_env_name(input_df):
    if "CONDA_DEFAULT_ENV" in os.environ:
        # A non-default Anaconda environment is active
        conda_env_name = os.environ["CONDA_DEFAULT_ENV"]
    else:
        conda_env_name = DEFAULT_ENV_NAME

    response = json.dumps({ RESPONSE_KEY_CONDA_ENV_NAME : conda_env_name })
    return response


def get_py_version(input_df):
    response = json.dumps({ RESPONSE_KEY_PYTHON_VERSION : PYTHON_VERSION })
    return response
