from collections import namedtuple
import os


import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.datasets import load_boston, load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pandas as pd
import pytest

import mlflow


ModelWithExplanation = namedtuple(
    "ModelWithExplanation", ["model", "X", "shap_values", "base_values"]
)


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
        pd.DataFrame(data.data[:100, :4], columns=data.feature_names[:4]),
        pd.Series(data.target[:100], name="target"),
    )


@pytest.fixture(scope="module")
def regressor():
    X, y = get_boston()
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


@pytest.mark.large
@pytest.mark.parametrize("artifact_path", [None, "dir"])
def test_log_explanation_with_regressor(regressor, artifact_path):
    model = regressor.model
    X = regressor.X

    with mlflow.start_run() as run:
        explanation_path = mlflow.shap.log_explanation(model.predict, X, artifact_path)

    # Assert no figure is open
    assert len(plt.get_fignums()) == 0

    artifact_path_expected = "model_explanations_shap" if artifact_path is None else artifact_path
    artifacts = set(yield_artifacts(run.info.run_id))

    assert explanation_path == os.path.join(run.info.artifact_uri, artifact_path_expected)
    assert artifacts == {
        os.path.join(artifact_path_expected, "base_values.npy"),
        os.path.join(artifact_path_expected, "shap_values.npy"),
        os.path.join(artifact_path_expected, "summary_bar_plot.png"),
    }

    shap_values = np.load(os.path.join(explanation_path, "shap_values.npy"))
    base_values = np.load(os.path.join(explanation_path, "base_values.npy"))
    np.testing.assert_array_equal(shap_values, regressor.shap_values)
    np.testing.assert_array_equal(base_values, regressor.base_values)


@pytest.mark.large
@pytest.mark.parametrize("artifact_path", [None, "dir"])
def test_log_explanation_with_classifier(classifier, artifact_path):
    model = classifier.model
    X = classifier.X

    with mlflow.start_run() as run:
        explanation_uri = mlflow.shap.log_explanation(model.predict_proba, X, artifact_path)

    # Assert no figure is open
    assert len(plt.get_fignums()) == 0

    artifact_path_expected = "model_explanations_shap" if artifact_path is None else artifact_path
    artifacts = set(yield_artifacts(run.info.run_id))

    assert explanation_uri == os.path.join(run.info.artifact_uri, artifact_path_expected)
    assert artifacts == {
        os.path.join(artifact_path_expected, "base_values.npy"),
        os.path.join(artifact_path_expected, "shap_values.npy"),
        os.path.join(artifact_path_expected, "summary_bar_plot.png"),
    }

    shap_values = np.load(os.path.join(explanation_uri, "shap_values.npy"))
    base_values = np.load(os.path.join(explanation_uri, "base_values.npy"))
    np.testing.assert_array_equal(shap_values, classifier.shap_values)
    np.testing.assert_array_equal(base_values, classifier.base_values)


@pytest.mark.large
def test_log_explanation_with_numpy_array(regressor):
    model = regressor.model
    X = regressor.X.values

    with mlflow.start_run() as run:
        explanation_uri = mlflow.shap.log_explanation(model.predict, X)

    # Assert no figure is open
    assert len(plt.get_fignums()) == 0

    artifact_path_expected = "model_explanations_shap"
    artifacts = set(yield_artifacts(run.info.run_id))

    assert explanation_uri == os.path.join(run.info.artifact_uri, artifact_path_expected)
    assert artifacts == {
        os.path.join(artifact_path_expected, "base_values.npy"),
        os.path.join(artifact_path_expected, "shap_values.npy"),
        os.path.join(artifact_path_expected, "summary_bar_plot.png"),
    }

    shap_values = np.load(os.path.join(explanation_uri, "shap_values.npy"))
    base_values = np.load(os.path.join(explanation_uri, "base_values.npy"))
    np.testing.assert_array_equal(shap_values, regressor.shap_values)
    np.testing.assert_array_equal(base_values, regressor.base_values)


@pytest.mark.large
def test_log_explanation_with_small_dataset(regressor):
    """
    Test that `log_explanation` does not fail when `features` has less rows than
    `_MAXIMUM_BACKGROUND_DATA_SIZE`.
    """
    model = regressor.model
    X = regressor.X

    with mlflow.start_run():
        mlflow.shap.log_explanation(
            model.predict, X.iloc[: mlflow.shap._MAXIMUM_BACKGROUND_DATA_ROWS - 1]
        )
