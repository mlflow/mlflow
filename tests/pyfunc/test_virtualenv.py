import os
from unittest import mock
from collections import namedtuple

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

import mlflow
from mlflow.pyfunc.scoring_server import CONTENT_TYPE_JSON_SPLIT_ORIENTED
from mlflow.utils.virtualenv import (
    _MLFLOW_ENV_ROOT_ENV_VAR,
    _is_pyenv_available,
    _is_virtualenv_available,
)
from tests.helper_functions import pyfunc_serve_and_score_model

requires_pyenv_and_virtualenv = pytest.mark.skipif(
    not (_is_pyenv_available() and _is_virtualenv_available()),
    reason="requires pyenv and virtualenv",
)


@pytest.fixture(scope="module")
def sklearn_model():
    X, y = load_iris(return_X_y=True, as_frame=True)
    model = LogisticRegression().fit(X, y)
    X_pred = X.sample(frac=0.1, random_state=0)
    y_pred = model.predict(X_pred)
    return namedtuple("Model", ["model", "X_pred", "y_pred"])(model, X_pred, y_pred)


def serve_and_score(model_uri, data, extra_args=None):
    resp = pyfunc_serve_and_score_model(
        model_uri,
        data=data,
        content_type=CONTENT_TYPE_JSON_SPLIT_ORIENTED,
        extra_args=["--env-manager=virtualenv"] + (extra_args or []),
    )
    return pd.read_json(resp.content, orient="records").values.squeeze()


@pytest.fixture
def temp_mlflow_env_root(tmp_path, monkeypatch):
    env_root = tmp_path / "envs"
    env_root.mkdir(exist_ok=True)
    monkeypatch.setenv(_MLFLOW_ENV_ROOT_ENV_VAR, str(env_root))
    return env_root


use_temp_mlflow_env_root = pytest.mark.usefixtures(temp_mlflow_env_root.__name__)


@requires_pyenv_and_virtualenv
@use_temp_mlflow_env_root
def test_serve_and_score(sklearn_model):
    with mlflow.start_run():
        model_info = mlflow.sklearn.log_model(sklearn_model.model, artifact_path="model")

    scores = serve_and_score(model_info.model_uri, sklearn_model.X_pred)
    np.testing.assert_array_almost_equal(scores, sklearn_model.y_pred)


@requires_pyenv_and_virtualenv
@use_temp_mlflow_env_root
def test_reuse_environment(temp_mlflow_env_root, sklearn_model):
    with mlflow.start_run():
        model_info = mlflow.sklearn.log_model(sklearn_model.model, artifact_path="model")

    scores = serve_and_score(model_info.model_uri, sklearn_model.X_pred)
    np.testing.assert_array_almost_equal(scores, sklearn_model.y_pred)
    # This call should reuse the environment created in the previous call
    scores = serve_and_score(model_info.model_uri, sklearn_model.X_pred)
    np.testing.assert_array_almost_equal(scores, sklearn_model.y_pred)
    assert len(os.listdir(temp_mlflow_env_root)) == 1


@requires_pyenv_and_virtualenv
@use_temp_mlflow_env_root
def test_python_env_does_not_exist(sklearn_model):
    with mlflow.start_run():
        with mock.patch("mlflow.utils.environment.PythonEnv.to_yaml") as mock_to_yaml:
            model_info = mlflow.sklearn.log_model(sklearn_model.model, artifact_path="model")
            mock_to_yaml.assert_called_once()

    scores = serve_and_score(model_info.model_uri, sklearn_model.X_pred)
    np.testing.assert_array_almost_equal(scores, sklearn_model.y_pred)


@requires_pyenv_and_virtualenv
def test_pip_install_fails(temp_mlflow_env_root, sklearn_model):
    with mlflow.start_run():
        model_info = mlflow.sklearn.log_model(
            sklearn_model.model,
            artifact_path="model",
            # Enforce pip install to fail using a non-existent package version
            pip_requirements=["mlflow==999.999.999"],
        )
    with pytest.raises(AssertionError, match="scoring process died"):
        serve_and_score(model_info.model_uri, sklearn_model.X_pred)
    assert len(list(temp_mlflow_env_root.iterdir())) == 0


@requires_pyenv_and_virtualenv
@use_temp_mlflow_env_root
def test_model_contains_conda_packages(sklearn_model):
    conda_env = {
        "name": "mlflow-env",
        "channels": ["conda-forge"],
        "dependencies": [
            "python=3.7.9",
            "conda-package=1.2.3",  # <- conda package
            "pip<=21.3.1",
            {
                "pip": [
                    "mlflow",
                    "scikit-learn==1.0.2",
                ]
            },
        ],
    }

    with mlflow.start_run():
        with mock.patch("mlflow.utils.environment.PythonEnv.to_yaml") as mock_to_yaml:
            model_info = mlflow.sklearn.log_model(
                sklearn_model.model,
                artifact_path="model",
                conda_env=conda_env,
            )
            mock_to_yaml.assert_called_once()

    with pytest.raises(AssertionError, match="scoring process died"):
        serve_and_score(model_info.model_uri, sklearn_model.X_pred)
