import os
import sys
from collections import namedtuple
from io import BytesIO
from pathlib import Path
from stat import S_IRGRP, S_IROTH, S_IRUSR, S_IXGRP, S_IXOTH, S_IXUSR

import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

import mlflow
from mlflow.environment_variables import MLFLOW_ENV_ROOT
from mlflow.pyfunc.scoring_server import CONTENT_TYPE_JSON
from mlflow.utils.environment import _PYTHON_ENV_FILE_NAME, _REQUIREMENTS_FILE_NAME
from mlflow.utils.virtualenv import (
    _is_pyenv_available,
    _is_virtualenv_available,
)

from tests.helper_functions import pyfunc_serve_and_score_model

pytestmark = pytest.mark.skipif(
    not (_is_pyenv_available() and _is_virtualenv_available()),
    reason="requires pyenv and virtualenv",
)

TEST_DIR = "tests"
TEST_MLFLOW_1X_MODEL_DIR = os.path.join(TEST_DIR, "resources", "example_mlflow_1x_sklearn_model")


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
        content_type=CONTENT_TYPE_JSON,
        extra_args=["--env-manager=virtualenv"] + (extra_args or []),
    )
    return pd.read_json(BytesIO(resp.content), orient="records").values.squeeze()


@pytest.fixture
def temp_mlflow_env_root(tmp_path, monkeypatch):
    env_root = tmp_path / "envs"
    env_root.mkdir(exist_ok=True)
    monkeypatch.setenv(MLFLOW_ENV_ROOT.name, str(env_root))
    return env_root


use_temp_mlflow_env_root = pytest.mark.usefixtures(temp_mlflow_env_root.__name__)


@use_temp_mlflow_env_root
def test_restore_environment_with_virtualenv(sklearn_model):
    with mlflow.start_run():
        model_info = mlflow.sklearn.log_model(sklearn_model.model, artifact_path="model")

    scores = serve_and_score(model_info.model_uri, sklearn_model.X_pred)
    np.testing.assert_array_almost_equal(scores, sklearn_model.y_pred)


@use_temp_mlflow_env_root
def test_serve_and_score_read_only_model_directory(sklearn_model, tmp_path):
    model_path = str(tmp_path / "model")
    mlflow.sklearn.save_model(sklearn_model.model, path=model_path)
    os.chmod(
        model_path,
        S_IRUSR | S_IRGRP | S_IROTH | S_IXUSR | S_IXGRP | S_IXOTH,
    )

    scores = serve_and_score(model_path, sklearn_model.X_pred)
    np.testing.assert_array_almost_equal(scores, sklearn_model.y_pred)


@use_temp_mlflow_env_root
def test_serve_and_score_1x_models():
    X, _ = load_iris(return_X_y=True, as_frame=True)
    X_pred = X.sample(frac=0.1, random_state=0)
    loaded_model = mlflow.pyfunc.load_model(TEST_MLFLOW_1X_MODEL_DIR)
    y_pred = loaded_model.predict(X_pred)

    scores = serve_and_score(TEST_MLFLOW_1X_MODEL_DIR, X_pred)
    np.testing.assert_array_almost_equal(scores, y_pred)


@use_temp_mlflow_env_root
def test_reuse_environment(temp_mlflow_env_root, sklearn_model):
    with mlflow.start_run():
        model_info = mlflow.sklearn.log_model(sklearn_model.model, artifact_path="model")

    # Serve the model
    scores = serve_and_score(model_info.model_uri, sklearn_model.X_pred)
    np.testing.assert_array_almost_equal(scores, sklearn_model.y_pred)
    # Serve the model again. The environment created in the previous serving should be reused.
    scores = serve_and_score(model_info.model_uri, sklearn_model.X_pred)
    np.testing.assert_array_almost_equal(scores, sklearn_model.y_pred)
    assert len(list(temp_mlflow_env_root.iterdir())) == 1


@use_temp_mlflow_env_root
def test_differenet_requirements_create_different_environments(temp_mlflow_env_root, sklearn_model):
    sklearn_req = f"scikit-learn=={sklearn.__version__}"
    with mlflow.start_run():
        model_info1 = mlflow.sklearn.log_model(
            sklearn_model.model,
            artifact_path="model",
            pip_requirements=[sklearn_req],
        )
    scores = serve_and_score(model_info1.model_uri, sklearn_model.X_pred)
    np.testing.assert_array_almost_equal(scores, sklearn_model.y_pred)

    # Log the same model with different requirements
    with mlflow.start_run():
        model_info2 = mlflow.sklearn.log_model(
            sklearn_model.model,
            artifact_path="model",
            pip_requirements=[sklearn_req, "numpy"],
        )
    scores = serve_and_score(model_info2.model_uri, sklearn_model.X_pred)
    np.testing.assert_array_almost_equal(scores, sklearn_model.y_pred)
    # Two environments should exist now because the first and second models have different
    # requirements
    assert len(list(temp_mlflow_env_root.iterdir())) == 2


@use_temp_mlflow_env_root
def test_environment_directory_is_cleaned_up_when_unexpected_error_occurs(
    temp_mlflow_env_root, sklearn_model
):
    sklearn_req = "scikit-learn==999.999.999"
    with mlflow.start_run():
        model_info1 = mlflow.sklearn.log_model(
            sklearn_model.model,
            artifact_path="model",
            pip_requirements=[sklearn_req],
        )

    try:
        serve_and_score(model_info1.model_uri, sklearn_model.X_pred)
    except Exception:
        pass
    else:
        assert False, "Should have raised an exception"
    assert len(list(temp_mlflow_env_root.iterdir())) == 0


@use_temp_mlflow_env_root
def test_python_env_file_does_not_exist(sklearn_model):
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


def test_environment_is_removed_when_package_installation_fails(
    temp_mlflow_env_root, sklearn_model
):
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
def test_restore_environment_from_conda_yaml_containing_conda_packages(sklearn_model):
    conda_env = {
        "name": "mlflow-env",
        "channels": ["conda-forge"],
        "dependencies": [
            "python=" + ".".join(map(str, sys.version_info[:3])),
            "conda-package=1.2.3",  # conda package
            "pip",
            {
                "pip": [
                    "mlflow",
                    f"scikit-learn=={sklearn.__version__}",
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
    serve_and_score(model_info.model_uri, sklearn_model.X_pred)
