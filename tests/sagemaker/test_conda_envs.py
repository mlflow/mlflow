from __future__ import print_function

import os
import json
import pickle
import tempfile
import pytest
import yaml

import pandas as pd
import sklearn.datasets as _sklearn_datasets
import sklearn.linear_model as glm

from tests.helper_functions import score_model_in_sagemaker_docker_container

from mlflow import pyfunc

from mlflow import sklearn as mlflow_sklearn 

from mlflow.models import Model
from mlflow.utils import PYTHON_VERSION

from collections import namedtuple

SerializationConfig = namedtuple("SerializationConfig", 
        ["model_path", "config_path", "data_path", "model_file", "mlflow_model"])

DataSet = namedtuple("DataSet", ["samples", "sample_schema", "labels"])

RESPONSE_KEY_CONDA_ENV_NAME = "CONDA_ENV_NAME"
RESPONSE_KEY_PYTHON_VERSION = "PYTHON_VERSION"
DEFAULT_ENV_NAME = "DEFAULT_ENV"


class _DockerTestPredictor:

    def __init__(self, func):
        self.func = func
    
    def predict(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def _get_env_name(input_df):
    if "CONDA_DEFAULT_ENV" in os.environ:
        # A non-default Anaconda environment is active
        conda_env_name = os.environ["CONDA_DEFAULT_ENV"]
    else:
        conda_env_name = DEFAULT_ENV_NAME

    response = json.dumps({ RESPONSE_KEY_CONDA_ENV_NAME : conda_env_name })
    return response


def _get_py_version(input_df):
    response = json.dumps({ RESPONSE_KEY_PYTHON_VERSION : PYTHON_VERSION })
    return response


def _create_conda_env_yaml(env_name, dependencies=[]):
    conda_env = { "name" : env_name, 
                  "dependencies" : dependencies 
                }
    conda_env = yaml.dump(conda_env, default_flow_style=False)
    return conda_env


def _save_model(serialization_config, predictor, conda_env, py_version=PYTHON_VERSION):
    with open(serialization_config.model_file, "wb") as out:
        pickle.dump(predictor, out)
    
    pyfunc.add_to_model(serialization_config.mlflow_model, 
                        loader_module="tests.sagemaker.test_conda_envs",
                        data=serialization_config.data_path,
                        env=conda_env)

    flavors = serialization_config.mlflow_model.flavors
    pyfunc_config = flavors[pyfunc.FLAVOR_NAME]
    if py_version is not None:
        pyfunc_config[pyfunc.PY_VERSION] = py_version
    else:
        del pyfunc_config[pyfunc.PY_VERSION]

    serialization_config.mlflow_model.save(serialization_config.config_path)
    return serialization_config.model_path


def load_pyfunc(path):
    with open(path, "rb") as f:
        return pickle.load(f)


@pytest.fixture
def _serialization_config():
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


@pytest.fixture
def _sklearn_data():
    iris = _sklearn_datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target
    return DataSet(samples=X, sample_schema=iris.feature_names[:2], labels=y)


@pytest.fixture
def _sklearn_model(_sklearn_data):
    linear_lr = glm.LogisticRegression()
    linear_lr.fit(_sklearn_data.samples, _sklearn_data.labels)
    return linear_lr


# @pytest.mark.large
def test_sagemaker_container_activates_custom_conda_env(_serialization_config):
    custom_env_name = "custom_env"
    conda_env = _create_conda_env_yaml(env_name=custom_env_name,
                                      dependencies=["pytest", 
                                                    "python={pyv}".format(pyv=PYTHON_VERSION)])

    conda_env_subpath = "env"
    conda_env_path = os.path.join(_serialization_config.model_path, conda_env_subpath)
    with open(conda_env_path, "wb") as out:
        out.write(conda_env)

    predictor = _DockerTestPredictor(func=_get_env_name)
    model_path = _save_model(serialization_config=_serialization_config, 
                            predictor=predictor, 
                            conda_env=conda_env_subpath, 
                            py_version=PYTHON_VERSION)

    data = pd.DataFrame()
    response = score_model_in_sagemaker_docker_container(model_path, data)
    response = json.loads(response)

    assert RESPONSE_KEY_CONDA_ENV_NAME in response
    docker_env_name = response[RESPONSE_KEY_CONDA_ENV_NAME]
    assert docker_env_name == custom_env_name


# @pytest.mark.large
def test_sagemaker_container_uses_py_version_specified_by_custom_conda_env(_serialization_config):
    custom_env_name = "custom_env"
    custom_env_py_version = "3.6.4"
    conda_env = _create_conda_env_yaml(env_name=custom_env_name, dependencies=[
        "pytest", "python={pyv}".format(pyv=custom_env_py_version)])

    conda_env_subpath = "env"
    conda_env_path = os.path.join(_serialization_config.model_path, conda_env_subpath)
    with open(conda_env_path, "wb") as out:
        out.write(conda_env)

    predictor = _DockerTestPredictor(func=_get_py_version)
    model_path = _save_model(serialization_config=_serialization_config, 
                            predictor=predictor, 
                            conda_env=conda_env_subpath, 
                            py_version=PYTHON_VERSION)

    data = pd.DataFrame()
    response = score_model_in_sagemaker_docker_container(model_path, data)
    response = json.loads(response)

    assert RESPONSE_KEY_PYTHON_VERSION in response
    docker_py_version = response[RESPONSE_KEY_PYTHON_VERSION]
    assert docker_py_version == custom_env_py_version 


# @pytest.mark.large
def test_sagemaker_container_adds_model_py_version_to_custom_env_when_py_version_is_unspecified(
        _serialization_config):
    custom_env_name = "custom_env"
    conda_env = _create_conda_env_yaml(env_name=custom_env_name,
                                      dependencies=["pytest"]) 

    conda_env_subpath = "env"
    conda_env_path = os.path.join(_serialization_config.model_path, conda_env_subpath)
    with open(conda_env_path, "wb") as out:
        out.write(conda_env)

    predictor = _DockerTestPredictor(func=_get_py_version)
    model_path = _save_model(serialization_config=_serialization_config, 
                            predictor=predictor, 
                            conda_env=conda_env_subpath, 
                            py_version=PYTHON_VERSION)

    data = pd.DataFrame()
    response = score_model_in_sagemaker_docker_container(model_path, data)
    response = json.loads(response)

    assert RESPONSE_KEY_PYTHON_VERSION in response
    docker_py_version = response[RESPONSE_KEY_PYTHON_VERSION]
    assert docker_py_version == PYTHON_VERSION


# @pytest.mark.large
def test_sagemaker_container_uses_default_env_for_absent_py_version_and_custom_env(
        _serialization_config):
    predictor = _DockerTestPredictor(func=_get_env_name)
    model_path = _save_model(serialization_config=_serialization_config, 
                            predictor=predictor, 
                            conda_env=None, 
                            py_version=None)

    data = pd.DataFrame()
    response = score_model_in_sagemaker_docker_container(model_path, data)
    response = json.loads(response)

    assert RESPONSE_KEY_CONDA_ENV_NAME in response
    docker_env_name = response[RESPONSE_KEY_CONDA_ENV_NAME]
    assert docker_env_name == DEFAULT_ENV_NAME


# @pytest.mark.large
def test_sagemaker_container_uses_conda_supported_python_version_for_absent_custom_env(
        _serialization_config):
    predictor = _DockerTestPredictor(func=_get_py_version)
    model_path = _save_model(serialization_config=_serialization_config, 
                            predictor=predictor, 
                            conda_env=None,
                            py_version=PYTHON_VERSION)

    data = pd.DataFrame()
    response = score_model_in_sagemaker_docker_container(model_path, data, timeout_seconds=600)
    response = json.loads(response)

    assert RESPONSE_KEY_PYTHON_VERSION in response
    docker_py_version = response[RESPONSE_KEY_PYTHON_VERSION]
    assert docker_py_version == PYTHON_VERSION


# @pytest.mark.large
def test_sagemaker_container_uses_default_env_for_unsupported_py_version_and_absent_custom_env(
        _serialization_config):
    unsupported_py_version = "3.3"
    predictor = _DockerTestPredictor(func=_get_py_version)
    model_path = _save_model(serialization_config=_serialization_config, 
                            predictor=predictor, 
                            conda_env=None,
                            py_version=unsupported_py_version)

    data = pd.DataFrame()
    response = score_model_in_sagemaker_docker_container(model_path, data)
    response = json.loads(response)

    assert RESPONSE_KEY_PYTHON_VERSION in response
    docker_py_version = response[RESPONSE_KEY_PYTHON_VERSION]
    assert docker_py_version != PYTHON_VERSION
    docker_minor_py_version = int(docker_py_version.split(".")[1])
    assert docker_minor_py_version >= 6


# @pytest.mark.large
def test_sagemaker_container_serves__sklearn_model_with_compatible_py_version(
        _sklearn_model, _sklearn_data):
    model_path = tempfile.mktemp(dir="/tmp")
    mlflow_sklearn.save_model(sk_model=_sklearn_model, 
                              path=model_path)
    sample = pd.DataFrame(_sklearn_data[0])
    response = score_model_in_sagemaker_docker_container(model_path, sample, timeout_seconds=600)
    response = pd.read_json(response)
    assert response.empty == False
    
