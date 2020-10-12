from collections import namedtuple
import os


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston, load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pandas as pd
import pytest

import mlflow.shap


ModelWithData = namedtuple("ModelWithData", ["model", "X"])


def yield_artifacts(run_id, path=None):
    """
    Yields all artifacts in the specified run.
    """
    client = mlflow.tracking.MlflowClient()
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            yield from yield_artifacts(run_id, item.path)
        else:
            yield item.path


def get_iris():
    data = load_iris()
    return (
        pd.DataFrame(data.data, columns=data.feature_names),
        pd.Series(data.target, name="target"),
    )


def get_boston():
    data = load_boston()
    return (
        pd.DataFrame(data.data[:, :4], columns=data.feature_names[:4]),
        pd.Series(data.target, name="target"),
    )


@pytest.fixture(scope="module")
def regression_model():
    X, y = get_boston()
    model = RandomForestRegressor()
    model.fit(X, y)
    return ModelWithData(model, X)


@pytest.fixture(scope="module")
def classification_model():
    X, y = get_iris()
    model = RandomForestClassifier()
    model.fit(X, y)
    return ModelWithData(model, X)


@pytest.mark.large
@pytest.mark.parametrize("np_obj", [np.float(0.0), np.array([0.0])])
def test_log_numpy(np_obj):

    with mlflow.start_run() as run:
        mlflow.shap._log_numpy(np_obj, "test.npy")
        mlflow.shap._log_numpy(np_obj, "test.npy", artifact_path="dir")

    artifacts = set(yield_artifacts(run.info.run_id))
    assert artifacts == {"test.npy", "dir/test.npy"}


@pytest.mark.large
def test_log_matplotlib_figure():

    fig, ax = plt.subplots()
    ax.plot([0, 1], [2, 3])

    with mlflow.start_run() as run:
        mlflow.shap._log_matplotlib_figure(fig, "test.png")
        mlflow.shap._log_matplotlib_figure(fig, "test.png", artifact_path="dir")

    artifacts = set(yield_artifacts(run.info.run_id))
    assert artifacts == {"test.png", "dir/test.png"}


@pytest.mark.parametrize("artifact_path", [None, "dir"])
def test_log_explanation_regression_model(regression_model, artifact_path):
    model = regression_model.model
    X = regression_model.X

    with mlflow.start_run() as run:
        mlflow.shap.log_explanation(model.predict, X, artifact_path)

    artifacts = set(yield_artifacts(run.info.run_id))

    if artifact_path is None:
        assert artifacts == {
            "shap/base_values.npy",
            "shap/shap_values.npy",
            "shap/summary_bar_plot.png",
        }
    else:
        assert artifacts == {
            os.path.join(artifact_path, "base_values.npy"),
            os.path.join(artifact_path, "shap_values.npy"),
            os.path.join(artifact_path, "summary_bar_plot.png"),
        }


@pytest.mark.parametrize("artifact_path", [None, "dir"])
def test_log_explanation_classification_model(classification_model, artifact_path):
    model = classification_model.model
    X = classification_model.X

    with mlflow.start_run() as run:
        mlflow.shap.log_explanation(model.predict, X, artifact_path)

    artifacts = set(yield_artifacts(run.info.run_id))

    if artifact_path is None:
        assert artifacts == {
            "shap/base_values.npy",
            "shap/shap_values.npy",
            "shap/summary_bar_plot.png",
        }
    else:
        assert artifacts == {
            os.path.join(artifact_path, "base_values.npy"),
            os.path.join(artifact_path, "shap_values.npy"),
            os.path.join(artifact_path, "summary_bar_plot.png"),
        }
