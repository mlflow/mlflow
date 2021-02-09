import mlflow
import shap
import numpy as np
import sklearn
from mlflow.utils import PYTHON_VERSION


def test_sklearn_log_explainer():
    """
    Tests mlflow.shap log_explainer with mlflow serialization of the underlying model
    """

    with mlflow.start_run() as run:

        run_id = run.info.run_id

        X, y = shap.datasets.boston()
        model = sklearn.ensemble.RandomForestRegressor(n_estimators=100)
        model.fit(X, y)

        explainer_original = shap.Explainer(model.predict, X, algorithm="permutation")
        shap_values_original = explainer_original(X[:5])

        mlflow.shap.log_explainer(explainer_original, "test_explainer")

        explainer_loaded = mlflow.shap.load_explainer("runs:/" + run_id + "/test_explainer")
        shap_values_new = explainer_loaded(X[:5])

        np.testing.assert_array_equal(shap_values_original.base_values, shap_values_new.base_values)
        np.testing.assert_allclose(shap_values_original.values, shap_values_new.values, rtol=100, atol=100)


def test_sklearn_log_explainer_self_serialization():
    """
    Tests mlflow.shap log_explainer with SHAP internal serialization of the underlying model
    """

    with mlflow.start_run() as run:

        run_id = run.info.run_id

        X, y = shap.datasets.boston()
        model = sklearn.ensemble.RandomForestRegressor(n_estimators=100)
        model.fit(X, y)

        explainer_original = shap.Explainer(model.predict, X, algorithm="permutation")
        shap_values_original = explainer_original(X[:5])

        mlflow.shap.log_explainer(explainer_original, "test_explainer", False)

        explainer_loaded = mlflow.shap.load_explainer("runs:/" + run_id + "/test_explainer")
        shap_values_new = explainer_loaded(X[:5])

        np.testing.assert_array_equal(shap_values_original.base_values, shap_values_new.base_values)
        np.testing.assert_allclose(shap_values_original.values, shap_values_new.values, rtol=100, atol=100)


def test_sklearn_log_explainer_pyfunc():
    """
    Tests mlflow.shap log_explainer with mlflow
    serialization of the underlying model using pyfunc flavor
    """

    with mlflow.start_run() as run:

        run_id = run.info.run_id

        X, y = shap.datasets.boston()
        model = sklearn.ensemble.RandomForestRegressor(n_estimators=100)
        model.fit(X, y)

        explainer_original = shap.Explainer(model.predict, X, algorithm="permutation")
        shap_values_original = explainer_original(X[:2])

        mlflow.shap.log_explainer(explainer_original, "test_explainer")

        explainer_pyfunc = mlflow.pyfunc.load_model("runs:/" + run_id + "/test_explainer")
        shap_values_new = explainer_pyfunc.predict(X[:2])

        np.testing.assert_allclose(shap_values_original.values, shap_values_new, rtol=100, atol=100)


def test_load_pyfunc():

    X, y = shap.datasets.boston()
    model = sklearn.ensemble.RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    explainer_original = shap.Explainer(model.predict, X, algorithm="permutation")
    shap_values_original = explainer_original(X[:2])
    mlflow.shap.save_model(explainer_original, "pyfunc_test")

    explainer_pyfunc = mlflow.shap._load_pyfunc("pyfunc_test")
    shap_values_new = explainer_pyfunc.predict(X[:2])

    np.testing.assert_allclose(shap_values_original.values, shap_values_new, rtol=100, atol=100)


def test_merge_environment():

    test_shap_env = {
        "channels": ["default", "conda-forge"],
        "dependencies": ["python=3.8.5", "pip", {"pip": ["mlflow", "shap==0.38.0"]}],
    }

    test_model_env = {
        "channels": ["default", "conda-forge"],
        "dependencies": [
            "python=3.8.5",
            "pip",
            {"pip": ["mlflow", "scikit-learn==0.24.0", "cloudpickle==1.6.0"]},
        ],
    }

    expected_merged_env = {
        "name": "mlflow-env",
        "channels": ["defaults", "conda-forge"],
        "dependencies": [
            "python={}".format(PYTHON_VERSION),
            "pip",
            {"pip": ["mlflow", "scikit-learn==0.24.0", "cloudpickle==1.6.0", "shap==0.38.0"]},
        ],
    }

    actual_merged_env = mlflow.shap._merge_environments(test_shap_env, test_model_env)

    assert sorted(expected_merged_env["channels"]) == sorted(actual_merged_env["channels"])

    expected_pip_deps = []
    actual_pip_deps = []

    expected_conda_deps = []
    actual_conda_deps = []

    for dep in expected_merged_env["dependencies"]:
        if isinstance(dep, dict):
            expected_conda_deps = dep["pip"]
        else:
            expected_pip_deps.append(dep)

    for dep in actual_merged_env["dependencies"]:
        if isinstance(dep, dict):
            actual_conda_deps = dep["pip"]
        else:
            actual_pip_deps.append(dep)

    assert sorted(expected_pip_deps) == sorted(actual_pip_deps)
    assert sorted(expected_conda_deps) == sorted(actual_conda_deps)
