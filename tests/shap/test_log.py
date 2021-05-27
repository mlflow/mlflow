import mlflow
import shap
import numpy as np
import sklearn
from mlflow.utils import PYTHON_VERSION
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.model_utils import _get_flavor_configuration


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

        explainer_uri = "runs:/" + run_id + "/test_explainer"

        explainer_loaded = mlflow.shap.load_explainer(explainer_uri)
        shap_values_new = explainer_loaded(X[:5])

        explainer_path = _download_artifact_from_uri(artifact_uri=explainer_uri)
        flavor_conf = _get_flavor_configuration(
            model_path=explainer_path, flavor_name=mlflow.shap.FLAVOR_NAME
        )
        underlying_model_flavor = flavor_conf["underlying_model_flavor"]

        assert underlying_model_flavor == mlflow.sklearn.FLAVOR_NAME
        np.testing.assert_array_equal(shap_values_original.base_values, shap_values_new.base_values)
        np.testing.assert_allclose(
            shap_values_original.values, shap_values_new.values, rtol=100, atol=100
        )


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

        mlflow.shap.log_explainer(
            explainer_original, "test_explainer", serialize_model_using_mlflow=False
        )

        explainer_uri = "runs:/" + run_id + "/test_explainer"

        explainer_loaded = mlflow.shap.load_explainer("runs:/" + run_id + "/test_explainer")
        shap_values_new = explainer_loaded(X[:5])

        explainer_path = _download_artifact_from_uri(artifact_uri=explainer_uri)
        flavor_conf = _get_flavor_configuration(
            model_path=explainer_path, flavor_name=mlflow.shap.FLAVOR_NAME
        )
        underlying_model_flavor = flavor_conf["underlying_model_flavor"]

        assert underlying_model_flavor is None
        np.testing.assert_array_equal(shap_values_original.base_values, shap_values_new.base_values)
        np.testing.assert_allclose(
            shap_values_original.values, shap_values_new.values, rtol=100, atol=100
        )


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


def test_load_pyfunc(tmpdir):

    X, y = shap.datasets.boston()
    model = sklearn.ensemble.RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    explainer_original = shap.Explainer(model.predict, X, algorithm="permutation")
    shap_values_original = explainer_original(X[:2])
    path = tmpdir.join("pyfunc_test").strpath
    mlflow.shap.save_explainer(explainer_original, path)

    explainer_pyfunc = mlflow.shap._load_pyfunc(path)
    shap_values_new = explainer_pyfunc.predict(X[:2])

    np.testing.assert_allclose(shap_values_original.values, shap_values_new, rtol=100, atol=100)


def test_merge_environment():

    test_shap_env = {
        "channels": ["conda-forge"],
        "dependencies": ["python=3.8.5", "pip", {"pip": ["mlflow", "shap==0.38.0"]}],
    }

    test_model_env = {
        "channels": ["conda-forge"],
        "dependencies": [
            "python=3.8.5",
            "pip",
            {"pip": ["mlflow", "scikit-learn==0.24.0", "cloudpickle==1.6.0"]},
        ],
    }

    expected_merged_env = {
        "name": "mlflow-env",
        "channels": ["conda-forge"],
        "dependencies": [
            "python={}".format(PYTHON_VERSION),
            "pip",
            {"pip": ["mlflow", "scikit-learn==0.24.0", "cloudpickle==1.6.0", "shap==0.38.0"]},
        ],
    }

    actual_merged_env = mlflow.shap._merge_environments(test_shap_env, test_model_env)

    assert sorted(expected_merged_env["channels"]) == sorted(actual_merged_env["channels"])

    expected_conda_deps, expected_pip_deps = mlflow.shap._get_conda_and_pip_dependencies(
        expected_merged_env
    )
    actual_conda_deps, actual_pip_deps = mlflow.shap._get_conda_and_pip_dependencies(
        actual_merged_env
    )

    assert sorted(expected_pip_deps) == sorted(actual_pip_deps)
    assert sorted(expected_conda_deps) == sorted(actual_conda_deps)
