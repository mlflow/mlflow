import os
from pathlib import Path
from collections import namedtuple

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

import mlflow
from mlflow.pyfunc.scoring_server import CONTENT_TYPE_JSON_SPLIT_ORIENTED
from mlflow.utils.environment import _PYTHON_ENV_FILE_NAME, _REQUIREMENTS_FILE_NAME
from mlflow.utils.virtualenv import (
    _MLFLOW_ENV_ROOT_ENV_VAR,
    _is_pyenv_available,
    _is_virtualenv_available,
)
from tests.helper_functions import pyfunc_serve_and_score_model

pytestmark = pytest.mark.skipif(
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


@use_temp_mlflow_env_root
def test_serve_and_score(sklearn_model):
    with mlflow.start_run():
        model_info = mlflow.sklearn.log_model(sklearn_model.model, artifact_path="model")

    scores = serve_and_score(model_info.model_uri, sklearn_model.X_pred)
    np.testing.assert_array_almost_equal(scores, sklearn_model.y_pred)


@use_temp_mlflow_env_root
def test_reuse_environment(temp_mlflow_env_root, sklearn_model):
    with mlflow.start_run():
        model_info = mlflow.sklearn.log_model(sklearn_model.model, artifact_path="model")

    # Serve the model
    scores = serve_and_score(model_info.model_uri, sklearn_model.X_pred)
    np.testing.assert_array_almost_equal(scores, sklearn_model.y_pred)
    # Serve the model again (environment created in the previous serving should be reused)
    scores = serve_and_score(model_info.model_uri, sklearn_model.X_pred)
    np.testing.assert_array_almost_equal(scores, sklearn_model.y_pred)
    assert len(os.listdir(temp_mlflow_env_root)) == 1


@use_temp_mlflow_env_root
def test_python_env_file_is_missing(sklearn_model):
    with mlflow.start_run():
        model_info = mlflow.sklearn.log_model(sklearn_model.model, artifact_path="model")
        model_artifact_path = Path(mlflow.get_artifact_uri("model").replace("file://", ""))

    model_artifact_path.joinpath(_PYTHON_ENV_FILE_NAME).unlink()
    scores = serve_and_score(model_info.model_uri, sklearn_model.X_pred)
    np.testing.assert_array_almost_equal(scores, sklearn_model.y_pred)


@use_temp_mlflow_env_root
def test_python_env_file_and_requirements_file_do_not_exist(sklearn_model):
    with mlflow.start_run():
        model_info = mlflow.sklearn.log_model(sklearn_model.model, artifact_path="model")
        model_artifact_path = Path(mlflow.get_artifact_uri("model").replace("file://", ""))

    model_artifact_path.joinpath(_PYTHON_ENV_FILE_NAME).unlink()
    model_artifact_path.joinpath(_REQUIREMENTS_FILE_NAME).unlink()
    scores = serve_and_score(model_info.model_uri, sklearn_model.X_pred)
    np.testing.assert_array_almost_equal(scores, sklearn_model.y_pred)


def test_pip_installation_failure(temp_mlflow_env_root, sklearn_model):
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
        model_info = mlflow.sklearn.log_model(
            sklearn_model.model,
            artifact_path="model",
            conda_env=conda_env,
        )
        model_artifact_path = Path(mlflow.get_artifact_uri("model").replace("file://", ""))

    model_artifact_path.joinpath(_PYTHON_ENV_FILE_NAME).unlink()
    with pytest.raises(AssertionError, match="scoring process died"):
        serve_and_score(model_info.model_uri, sklearn_model.X_pred)
