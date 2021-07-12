import mlflux
import shap
import numpy as np
import pandas as pd
import sklearn
from mlflux.utils import PYTHON_VERSION
from mlflux.tracking import MlflowClient
from mlflux.tracking.artifact_utils import _download_artifact_from_uri
from mlflux.utils.model_utils import _get_flavor_configuration


def test_sklearn_log_explainer():
    """
    Tests mlflux.shap log_explainer with mlflux serialization of the underlying model
    """

    with mlflux.start_run() as run:

        run_id = run.info.run_id

        X, y = shap.datasets.boston()
        model = sklearn.ensemble.RandomForestRegressor(n_estimators=100)
        model.fit(X, y)

        explainer_original = shap.Explainer(model.predict, X, algorithm="permutation")
        shap_values_original = explainer_original(X[:5])

        mlflux.shap.log_explainer(explainer_original, "test_explainer")

        explainer_uri = "runs:/" + run_id + "/test_explainer"

        explainer_loaded = mlflux.shap.load_explainer(explainer_uri)
        shap_values_new = explainer_loaded(X[:5])

        explainer_path = _download_artifact_from_uri(artifact_uri=explainer_uri)
        flavor_conf = _get_flavor_configuration(
            model_path=explainer_path, flavor_name=mlflux.shap.FLAVOR_NAME
        )
        underlying_model_flavor = flavor_conf["underlying_model_flavor"]

        assert underlying_model_flavor == mlflux.sklearn.FLAVOR_NAME
        np.testing.assert_array_equal(shap_values_original.base_values, shap_values_new.base_values)
        np.testing.assert_allclose(
            shap_values_original.values, shap_values_new.values, rtol=100, atol=100
        )


def test_sklearn_log_explainer_self_serialization():
    """
    Tests mlflux.shap log_explainer with SHAP internal serialization of the underlying model
    """

    with mlflux.start_run() as run:

        run_id = run.info.run_id

        X, y = shap.datasets.boston()
        model = sklearn.ensemble.RandomForestRegressor(n_estimators=100)
        model.fit(X, y)

        explainer_original = shap.Explainer(model.predict, X, algorithm="permutation")
        shap_values_original = explainer_original(X[:5])

        mlflux.shap.log_explainer(
            explainer_original, "test_explainer", serialize_model_using_mlflow=False
        )

        explainer_uri = "runs:/" + run_id + "/test_explainer"

        explainer_loaded = mlflux.shap.load_explainer("runs:/" + run_id + "/test_explainer")
        shap_values_new = explainer_loaded(X[:5])

        explainer_path = _download_artifact_from_uri(artifact_uri=explainer_uri)
        flavor_conf = _get_flavor_configuration(
            model_path=explainer_path, flavor_name=mlflux.shap.FLAVOR_NAME
        )
        underlying_model_flavor = flavor_conf["underlying_model_flavor"]

        assert underlying_model_flavor is None
        np.testing.assert_array_equal(shap_values_original.base_values, shap_values_new.base_values)
        np.testing.assert_allclose(
            shap_values_original.values, shap_values_new.values, rtol=100, atol=100
        )


def test_sklearn_log_explainer_pyfunc():
    """
    Tests mlflux.shap log_explainer with mlflux
    serialization of the underlying model using pyfunc flavor
    """

    with mlflux.start_run() as run:

        run_id = run.info.run_id

        X, y = shap.datasets.boston()
        model = sklearn.ensemble.RandomForestRegressor(n_estimators=100)
        model.fit(X, y)

        explainer_original = shap.Explainer(model.predict, X, algorithm="permutation")
        shap_values_original = explainer_original(X[:2])

        mlflux.shap.log_explainer(explainer_original, "test_explainer")

        explainer_pyfunc = mlflux.pyfunc.load_model("runs:/" + run_id + "/test_explainer")
        shap_values_new = explainer_pyfunc.predict(X[:2])

        np.testing.assert_allclose(shap_values_original.values, shap_values_new, rtol=100, atol=100)


def test_log_explanation_doesnt_create_autologged_run():
    mlflux.sklearn.autolog(disable=False, exclusive=False)
    dataset = sklearn.datasets.load_boston()
    X = pd.DataFrame(dataset.data[:50, :8], columns=dataset.feature_names[:8])
    y = dataset.target[:50]
    model = sklearn.linear_model.LinearRegression()
    model.fit(X, y)

    with mlflux.start_run() as run:
        mlflux.shap.log_explanation(model.predict, X)

    run_data = MlflowClient().get_run(run.info.run_id).data
    metrics, params, tags = run_data.metrics, run_data.params, run_data.tags
    assert not metrics
    assert not params
    assert all("mlflux." in key for key in tags)
    assert "mlflux.autologging" not in tags


def test_load_pyfunc(tmpdir):

    X, y = shap.datasets.boston()
    model = sklearn.ensemble.RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    explainer_original = shap.Explainer(model.predict, X, algorithm="permutation")
    shap_values_original = explainer_original(X[:2])
    path = tmpdir.join("pyfunc_test").strpath
    mlflux.shap.save_explainer(explainer_original, path)

    explainer_pyfunc = mlflux.shap._load_pyfunc(path)
    shap_values_new = explainer_pyfunc.predict(X[:2])

    np.testing.assert_allclose(shap_values_original.values, shap_values_new, rtol=100, atol=100)


def test_merge_environment():

    test_shap_env = {
        "channels": ["conda-forge"],
        "dependencies": ["python=3.8.5", "pip", {"pip": ["mlflux", "shap==0.38.0"]}],
    }

    test_model_env = {
        "channels": ["conda-forge"],
        "dependencies": [
            "python=3.8.5",
            "pip",
            {"pip": ["mlflux", "scikit-learn==0.24.0", "cloudpickle==1.6.0"]},
        ],
    }

    expected_merged_env = {
        "name": "mlflux-env",
        "channels": ["conda-forge"],
        "dependencies": [
            "python={}".format(PYTHON_VERSION),
            "pip",
            {"pip": ["mlflux", "scikit-learn==0.24.0", "cloudpickle==1.6.0", "shap==0.38.0"]},
        ],
    }

    actual_merged_env = mlflux.shap._merge_environments(test_shap_env, test_model_env)

    assert sorted(expected_merged_env["channels"]) == sorted(actual_merged_env["channels"])

    expected_conda_deps, expected_pip_deps = mlflux.shap._get_conda_and_pip_dependencies(
        expected_merged_env
    )
    actual_conda_deps, actual_pip_deps = mlflux.shap._get_conda_and_pip_dependencies(
        actual_merged_env
    )

    assert sorted(expected_pip_deps) == sorted(actual_pip_deps)
    assert sorted(expected_conda_deps) == sorted(actual_conda_deps)
