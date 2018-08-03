from __future__ import print_function

import os
import pickle
import shutil
import tempfile
import pytest

from click.testing import CliRunner
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


class DockerTestPredictor:

    def __init__(self, func):
        self.func = func
    
    def predict(self, *args, **kwargs):
        return self.func(*args, **kwargs)

def save_model(serialization_config, predictor, conda_env):
    with open(serialization_config.model_file, "wb") as out:
        pickle.dump(predictor, out)
    
    pyfunc.add_to_model(serialization_config.mlflow_model, 
                        loader_module="tests.sagemaker.test_conda_envs",
                        data=serialization_config.data_path,
                        env=conda_env)
    
    serialization_config.mlflow_model.save(serialization_config.config_path)
    return serialization_config.model_path


def pred_fn(input_df):
    import os
    import json

    conda_env_name = os.environ["CONDA_DEFAULT_ENV"]
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
    conda_env = 'name="custom_env"\n'

    conda_env_subpath = "env"
    conda_env_path = os.path.join(serialization_config.model_path, conda_env_subpath)
    with open(conda_env_path, "wb") as out:
        out.write(conda_env)

    print(serialization_config)

    predictor = DockerTestPredictor(func=pred_fn)
    model_path = save_model(serialization_config, predictor, conda_env_subpath)

    data = pd.DataFrame()
    score_model_in_sagemaker_docker_container(model_path, data)
