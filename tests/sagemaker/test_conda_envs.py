from __future__ import print_function

import os
import json
import pickle
import shutil
import tempfile
import pytest
import yaml

import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.linear_model
import sklearn.neighbors

from tests.helper_functions import score_model_in_sagemaker_docker_container

from mlflow import pyfunc
import mlflow.pyfunc.cli

from mlflow import sklearn as mlflow_sklearn 

from mlflow import tracking
from mlflow.models import Model
from mlflow.utils import PYTHON_VERSION
from mlflow.utils.file_utils import TempDir

from collections import namedtuple

SerializationConfig = namedtuple("SerializationConfig", 
        ["model_path", "config_path", "data_path", "model_file", "mlflow_model"])

KEY_CONDA_ENV_NAME = "CONDA_ENV_NAME"
DEFAULT_ENV_NAME = "DEFAULT_ENV"


class DockerTestPredictor:

    def __init__(self, func):
        self.func = func
    
    def predict(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def save_model(serialization_config, predictor, conda_env, remove_py_version=False):
    with open(serialization_config.model_file, "wb") as out:
        pickle.dump(predictor, out)
    
    pyfunc.add_to_model(serialization_config.mlflow_model, 
                        loader_module="tests.sagemaker.test_conda_envs",
                        data=serialization_config.data_path,
                        env=conda_env)

    if remove_py_version:
        flavors = serialization_config.mlflow_model.flavors
        pyfunc_config = flavors[pyfunc.FLAVOR_NAME]
        del pyfunc_config[pyfunc.PY_VERSION]

    serialization_config.mlflow_model.save(serialization_config.config_path)
    return serialization_config.model_path


def pred_fn(input_df):
    if "CONDA_DEFAULT_ENV" in os.environ:
        # A non-default Anaconda environment is active
        conda_env_name = os.environ["CONDA_DEFAULT_ENV"]
    else:
        conda_env_name = DEFAULT_ENV_NAME

    response = json.dumps({ KEY_CONDA_ENV_NAME : conda_env_name })
    return response


def load_pyfunc(path):
    with open(path, "rb") as f:
        return pickle.load(f)


@pytest.fixture
def serialization_config():
    model_path = tempfile.mkdtemp(dir="/tmp")
    config_path = os.path.join(model_path, "MLmodel")
    data_path = "model.pkl"
    model_file = os.path.join(model_path, data_path)
    model = Model()
    return SerializationConfig(model_path=model_path,
                               config_path=config_path,
                               data_path=data_path, 
                               model_file=model_file, 
                               mlflow_model=model)

def test_sagemaker_container_respects_custom_conda_environment(serialization_config):
    custom_env_name = "custom_env"
    conda_env = { "name" : custom_env_name, 
                  "dependencies" : ["pytest", "python={pyv}".format(pyv=PYTHON_VERSION)] 
                }
    conda_env = yaml.dump(conda_env, default_flow_style=False)

    conda_env_subpath = "env"
    conda_env_path = os.path.join(serialization_config.model_path, conda_env_subpath)
    with open(conda_env_path, "wb") as out:
        out.write(conda_env)

    print(serialization_config)

    predictor = DockerTestPredictor(func=pred_fn)
    model_path = save_model(serialization_config, predictor, conda_env_subpath)

    data = pd.DataFrame()
    response = score_model_in_sagemaker_docker_container(model_path, data)
    response = json.loads(response)

    assert KEY_CONDA_ENV_NAME in response
    docker_env_name = response[KEY_CONDA_ENV_NAME]
    assert docker_env_name == custom_env_name

def test_sagemaker_container_uses_default_environment_for_absent_py_version_and_custom_env(
        serialization_config):
    predictor = DockerTestPredictor(func=pred_fn)
    model_path = save_model(serialization_config=serialization_config, 
                            predictor=predictor, 
                            conda_env=None, 
                            remove_py_version=True)

    data = pd.DataFrame()
    response = score_model_in_sagemaker_docker_container(model_path, data)

    assert KEY_CONDA_ENV_NAME in response
    docker_env_name = response[KEY_CONDA_ENV_NAME]
    assert docker_env_name == DEFAULT_ENV_NAME 
