import os
from typing import Any, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import shap
from sklearn.datasets import load_diabetes, load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import mlflow
from mlflow import MlflowClient


class ModelWithExplanation(NamedTuple):
    model: Any
    X: Any
    shap_values: Any
    base_values: Any


def yield_artifacts(run_id, path=None):
    """
    Yields all artifacts in the specified run.
    """
    client = MlflowClient()
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


def get_diabetes():
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    return X.iloc[:100, :4], y.iloc[:100]


@pytest.fixture(scope="module")
def regressor():
    X, y = get_diabetes()
    model = RandomForestRegressor()
    model.fit(X, y)

    explainer = shap.KernelExplainer(model.predict, shap.kmeans(X, 100))
    shap_values = explainer.shap_values(X)

    return ModelWithExplanation(model, X, shap_values, explainer.expected_value)


@pytest.fixture(scope="module")
def classifier():
    X, y = get_iris()
    model = RandomForestClassifier()
    model.fit(X, y)

    explainer = shap.KernelExplainer(model.predict_proba, shap.kmeans(X, 100))
    shap_values = explainer.shap_values(X)

    return ModelWithExplanation(model, X, shap_values, explainer.expected_value)


@pytest.mark.parametrize("np_obj", [float(0.0), np.array([0.0])])
def test_log_numpy(np_obj):
    with mlflow.start_run() as run:
        mlflow.shap._log_numpy(np_obj, "test.npy")
        mlflow.shap._log_numpy(np_obj, "test.npy", artifact_path="dir")

    artifacts = set(yield_artifacts(run.info.run_id))
    assert artifacts == {"test.npy", "dir/test.npy"}


def test_log_matplotlib_figure():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [2, 3])

    with mlflow.start_run() as run:
        mlflow.shap._log_matplotlib_figure(fig, "test.png")
        mlflow.shap._log_matplotlib_figure(fig, "test.png", artifact_path="dir")

    artifacts = set(yield_artifacts(run.info.run_id))
    assert artifacts == {"test.png", "dir/test.png"}


def test_log_explanation_with_regressor(regressor):
    model = regressor.model
    X = regressor.X

    with mlflow.start_run() as run:
        explanation_path = mlflow.shap.log_explanation(model.predict, X)

    # Assert no figure is open
    assert len(plt.get_fignums()) == 0

    artifact_path = "model_explanations_shap"
    artifacts = set(yield_artifacts(run.info.run_id))

    assert explanation_path == os.path.join(run.info.artifact_uri, artifact_path)
    assert artifacts == {
        os.path.join(artifact_path, "base_values.npy"),
        os.path.join(artifact_path, "shap_values.npy"),
        os.path.join(artifact_path, "summary_bar_plot.png"),
    }

    shap_values = np.load(os.path.join(explanation_path, "shap_values.npy"))
    base_values = np.load(os.path.join(explanation_path, "base_values.npy"))
    np.testing.assert_array_equal(shap_values, regressor.shap_values)
    np.testing.assert_array_equal(base_values, regressor.base_values)


def test_log_explanation_with_classifier(classifier):
    model = classifier.model
    X = classifier.X

    with mlflow.start_run() as run:
        explanation_uri = mlflow.shap.log_explanation(model.predict_proba, X)

    # Assert no figure is open
    assert len(plt.get_fignums()) == 0

    artifact_path = "model_explanations_shap"
    artifacts = set(yield_artifacts(run.info.run_id))

    assert explanation_uri == os.path.join(run.info.artifact_uri, artifact_path)
    assert artifacts == {
        os.path.join(artifact_path, "base_values.npy"),
        os.path.join(artifact_path, "shap_values.npy"),
        os.path.join(artifact_path, "summary_bar_plot.png"),
    }

    shap_values = np.load(os.path.join(explanation_uri, "shap_values.npy"))
    base_values = np.load(os.path.join(explanation_uri, "base_values.npy"))
    np.testing.assert_array_equal(shap_values, classifier.shap_values)
    np.testing.assert_array_equal(base_values, classifier.base_values)


@pytest.mark.parametrize("artifact_path", ["dir", "dir1/dir2"])
def test_log_explanation_with_artifact_path(regressor, artifact_path):
    model = regressor.model
    X = regressor.X

    with mlflow.start_run() as run:
        explanation_path = mlflow.shap.log_explanation(model.predict, X, artifact_path)

    # Assert no figure is open
    assert len(plt.get_fignums()) == 0

    artifacts = set(yield_artifacts(run.info.run_id))

    assert explanation_path == os.path.join(run.info.artifact_uri, artifact_path)
    assert artifacts == {
        os.path.join(artifact_path, "base_values.npy"),
        os.path.join(artifact_path, "shap_values.npy"),
        os.path.join(artifact_path, "summary_bar_plot.png"),
    }

    shap_values = np.load(os.path.join(explanation_path, "shap_values.npy"))
    base_values = np.load(os.path.join(explanation_path, "base_values.npy"))
    np.testing.assert_array_equal(shap_values, regressor.shap_values)
    np.testing.assert_array_equal(base_values, regressor.base_values)


def test_log_explanation_without_active_run(regressor):
    model = regressor.model
    X = regressor.X.values

    with mlflow.start_run() as run:
        explanation_uri = mlflow.shap.log_explanation(model.predict, X)

        # Assert no figure is open
        assert len(plt.get_fignums()) == 0

        artifact_path = "model_explanations_shap"
        artifacts = set(yield_artifacts(run.info.run_id))

        assert explanation_uri == os.path.join(run.info.artifact_uri, artifact_path)
        assert artifacts == {
            os.path.join(artifact_path, "base_values.npy"),
            os.path.join(artifact_path, "shap_values.npy"),
            os.path.join(artifact_path, "summary_bar_plot.png"),
        }

        shap_values = np.load(os.path.join(explanation_uri, "shap_values.npy"))
        base_values = np.load(os.path.join(explanation_uri, "base_values.npy"))
        np.testing.assert_array_equal(shap_values, regressor.shap_values)
        np.testing.assert_array_equal(base_values, regressor.base_values)


def test_log_explanation_with_numpy_array(regressor):
    model = regressor.model
    X = regressor.X.values

    with mlflow.start_run() as run:
        explanation_uri = mlflow.shap.log_explanation(model.predict, X)

    # Assert no figure is open
    assert len(plt.get_fignums()) == 0

    artifact_path = "model_explanations_shap"
    artifacts = set(yield_artifacts(run.info.run_id))

    assert explanation_uri == os.path.join(run.info.artifact_uri, artifact_path)
    assert artifacts == {
        os.path.join(artifact_path, "base_values.npy"),
        os.path.join(artifact_path, "shap_values.npy"),
        os.path.join(artifact_path, "summary_bar_plot.png"),
    }

    shap_values = np.load(os.path.join(explanation_uri, "shap_values.npy"))
    base_values = np.load(os.path.join(explanation_uri, "base_values.npy"))
    np.testing.assert_array_equal(shap_values, regressor.shap_values)
    np.testing.assert_array_equal(base_values, regressor.base_values)


def test_log_explanation_with_small_features():
    """
    Verifies that `log_explanation` does not fail even when `features` has less records than
    `_MAXIMUM_BACKGROUND_DATA_SIZE`.
    """
    num_rows = 50
    assert num_rows < mlflow.shap._MAXIMUM_BACKGROUND_DATA_SIZE

    X, y = get_diabetes()
    X = X.iloc[:num_rows]
    y = y[:num_rows]
    model = RandomForestRegressor()
    model.fit(X, y)

    with mlflow.start_run() as run:
        explanation_uri = mlflow.shap.log_explanation(model.predict, X)

    artifact_path = "model_explanations_shap"
    artifacts = set(yield_artifacts(run.info.run_id))

    assert explanation_uri == os.path.join(run.info.artifact_uri, artifact_path)
    assert artifacts == {
        os.path.join(artifact_path, "base_values.npy"),
        os.path.join(artifact_path, "shap_values.npy"),
        os.path.join(artifact_path, "summary_bar_plot.png"),
    }

    explainer = shap.KernelExplainer(model.predict, shap.kmeans(X, num_rows))
    shap_values_expected = explainer.shap_values(X)

    base_values = np.load(os.path.join(explanation_uri, "base_values.npy"))
    shap_values = np.load(os.path.join(explanation_uri, "shap_values.npy"))
    np.testing.assert_array_equal(base_values, explainer.expected_value)
    np.testing.assert_array_equal(shap_values, shap_values_expected)
