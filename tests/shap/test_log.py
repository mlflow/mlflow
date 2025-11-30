import json
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import shap
import sklearn
from numba import njit
from packaging.version import Version
from sklearn.datasets import fetch_california_housing, load_diabetes

import mlflow
import mlflow.pyfunc.scoring_server as pyfunc_scoring_server
from mlflow import MlflowClient
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils import PYTHON_VERSION
from mlflow.utils.model_utils import _get_flavor_configuration

from tests.helper_functions import (
    _assert_pip_requirements,
    _compare_logged_code_paths,
    _mlflow_major_version_string,
    pyfunc_serve_and_score_model,
)


@pytest.fixture(scope="module")
def shap_model():
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    model = sklearn.ensemble.RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    return shap.Explainer(model.predict, X, algorithm="permutation")


def get_housing_data():
    X, y = fetch_california_housing(as_frame=True, return_X_y=True)

    return X[:1000], y[:1000]


def test_sklearn_log_explainer():
    """
    Tests mlflow.shap log_explainer with mlflow serialization of the underlying model
    """

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        X, y = get_housing_data()

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

        X, y = get_housing_data()

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

        X, y = get_housing_data()

        model = sklearn.ensemble.RandomForestRegressor(n_estimators=100)
        model.fit(X, y)

        explainer_original = shap.Explainer(model.predict, X, algorithm="permutation")
        shap_values_original = explainer_original(X[:2])

        mlflow.shap.log_explainer(explainer_original, "test_explainer")

        explainer_pyfunc = mlflow.pyfunc.load_model("runs:/" + run_id + "/test_explainer")
        shap_values_new = explainer_pyfunc.predict(X[:2])

        np.testing.assert_allclose(shap_values_original.values, shap_values_new, rtol=100, atol=100)


def test_log_explanation_doesnt_create_autologged_run():
    try:
        mlflow.sklearn.autolog(disable=False, exclusive=False)
        X, y = sklearn.datasets.load_diabetes(return_X_y=True, as_frame=True)
        X = X.iloc[:50, :4]
        y = y.iloc[:50]
        model = sklearn.linear_model.LinearRegression()
        model.fit(X, y)

        with mlflow.start_run() as run:
            mlflow.shap.log_explanation(model.predict, X)

        run_data = MlflowClient().get_run(run.info.run_id).data
        metrics = run_data.metrics
        params = run_data.params
        tags = run_data.tags
        assert not metrics
        assert not params
        assert all("mlflow." in key for key in tags)
        assert "mlflow.autologging" not in tags
    finally:
        mlflow.sklearn.autolog(disable=True)


def test_load_pyfunc(tmp_path):
    X, y = get_housing_data()

    model = sklearn.ensemble.RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    explainer_original = shap.Explainer(model.predict, X, algorithm="permutation")
    shap_values_original = explainer_original(X[:2])
    path = str(tmp_path.joinpath("pyfunc_test"))
    mlflow.shap.save_explainer(explainer_original, path)

    explainer_pyfunc = mlflow.shap._load_pyfunc(path)
    shap_values_new = explainer_pyfunc.predict(X[:2])

    np.testing.assert_allclose(shap_values_original.values, shap_values_new, rtol=100, atol=100)


def test_merge_environment():
    expected_mlflow_version = _mlflow_major_version_string()
    test_shap_env = {
        "channels": ["conda-forge"],
        "dependencies": ["python=3.8.5", "pip", {"pip": [expected_mlflow_version, "shap==0.38.0"]}],
    }

    test_model_env = {
        "channels": ["conda-forge"],
        "dependencies": [
            "python=3.8.5",
            "pip",
            {"pip": [expected_mlflow_version, "scikit-learn==0.24.0", "cloudpickle==1.6.0"]},
        ],
    }

    expected_merged_env = {
        "name": "mlflow-env",
        "channels": ["conda-forge"],
        "dependencies": [
            f"python={PYTHON_VERSION}",
            "pip",
            {
                "pip": [
                    expected_mlflow_version,
                    "scikit-learn==0.24.0",
                    "cloudpickle==1.6.0",
                    "shap==0.38.0",
                ]
            },
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

    assert sorted(expected_pip_deps) == actual_pip_deps
    assert sorted(expected_conda_deps) == actual_conda_deps


def test_merge_environment_with_duplicates():
    expected_mlflow_version = _mlflow_major_version_string()
    duplicate_dependency = "numpy==1.19.2"

    # Introduce the duplicate in both environments
    test_shap_env = {
        "channels": ["conda-forge"],
        "dependencies": [
            "python=3.8.5",
            "pip",
            {"pip": [expected_mlflow_version, "shap==0.38.0", duplicate_dependency]},
        ],
    }

    test_model_env = {
        "channels": ["conda-forge"],
        "dependencies": [
            "python=3.8.5",
            "pip",
            {
                "pip": [
                    expected_mlflow_version,
                    "scikit-learn==0.24.0",
                    "cloudpickle==1.6.0",
                    duplicate_dependency,
                ]
            },
        ],
    }

    # The expected merged environment should not have duplicates
    expected_merged_env = {
        "name": "mlflow-env",
        "channels": ["conda-forge"],
        "dependencies": [
            f"python={PYTHON_VERSION}",
            "pip",
            {
                "pip": [
                    expected_mlflow_version,
                    "scikit-learn==0.24.0",
                    "cloudpickle==1.6.0",
                    "shap==0.38.0",
                    duplicate_dependency,
                ]
            },
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

    # Check that there are no duplicates in the actual pip dependencies
    assert sorted(actual_pip_deps) == sorted(set(actual_pip_deps))
    assert sorted(expected_pip_deps) == actual_pip_deps
    assert sorted(expected_conda_deps) == actual_conda_deps


def test_log_model_with_pip_requirements(shap_model, tmp_path):
    expected_mlflow_version = _mlflow_major_version_string()
    sklearn_default_reqs = mlflow.sklearn.get_default_pip_requirements(include_cloudpickle=True)
    # Path to a requirements file
    req_file = tmp_path.joinpath("requirements.txt")
    req_file.write_text("a")
    with mlflow.start_run():
        model_info = mlflow.shap.log_explainer(shap_model, "model", pip_requirements=str(req_file))
        _assert_pip_requirements(
            model_info.model_uri,
            [expected_mlflow_version, "a", *sklearn_default_reqs],
            strict=False,
        )

    # List of requirements
    with mlflow.start_run():
        model_info = mlflow.shap.log_explainer(
            shap_model, "model", pip_requirements=[f"-r {req_file}", "b"]
        )
        _assert_pip_requirements(
            model_info.model_uri,
            [expected_mlflow_version, "a", "b", *sklearn_default_reqs],
            strict=False,
        )

    # Constraints file
    with mlflow.start_run():
        model_info = mlflow.shap.log_explainer(
            shap_model, "model", pip_requirements=[f"-c {req_file}", "b"]
        )
        _assert_pip_requirements(
            model_info.model_uri,
            [expected_mlflow_version, "b", "-c constraints.txt", *sklearn_default_reqs],
            ["a"],
            strict=False,
        )


def test_log_model_with_extra_pip_requirements(shap_model, tmp_path):
    expected_mlflow_version = _mlflow_major_version_string()
    shap_default_reqs = mlflow.shap.get_default_pip_requirements()
    sklearn_default_reqs = mlflow.sklearn.get_default_pip_requirements(include_cloudpickle=True)

    # Path to a requirements file
    req_file = tmp_path.joinpath("requirements.txt")
    req_file.write_text("a")
    with mlflow.start_run():
        log_info = mlflow.shap.log_explainer(
            shap_model, "model", extra_pip_requirements=str(req_file)
        )
        _assert_pip_requirements(
            log_info.model_uri,
            [expected_mlflow_version, *shap_default_reqs, "a", *sklearn_default_reqs],
        )

    # List of requirements
    with mlflow.start_run():
        log_info = mlflow.shap.log_explainer(
            shap_model, "model", extra_pip_requirements=[f"-r {req_file}", "b"]
        )
        _assert_pip_requirements(
            log_info.model_uri,
            [expected_mlflow_version, *shap_default_reqs, "a", "b", *sklearn_default_reqs],
        )

    # Constraints file
    with mlflow.start_run():
        log_info = mlflow.shap.log_explainer(
            shap_model, "model", extra_pip_requirements=[f"-c {req_file}", "b"]
        )
        _assert_pip_requirements(
            log_info.model_uri,
            [
                expected_mlflow_version,
                *shap_default_reqs,
                "b",
                "-c constraints.txt",
                *sklearn_default_reqs,
            ],
            ["a"],
        )


def create_identity_function():
    def identity(x):
        return x

    def _identity_inverse(x):
        return x

    identity.inverse = _identity_inverse

    return identity


@pytest.mark.skipif(Version(shap.__version__) < Version("0.42.0"), reason="numba njit compatible")
def test_pyfunc_serve_and_score_njit():
    # Create a numba-compatible identify link function due to breaking changes in shap
    # version 0.42.0. Python functions can no longer be passed to the numba jit compiler
    # with the changes introduced in that version.
    @njit
    def identity_function(x):
        return x

    X, y = get_housing_data()

    reg = sklearn.ensemble.RandomForestRegressor(n_estimators=10).fit(X, y)
    model = shap.Explainer(
        reg.predict,
        masker=X,
        algorithm="permutation",
        # `link` defaults to `shap.links.identity` which is decorated by `numba.jit` and causes
        # the following error when loading the explainer for serving:
        # ```
        # Exception: The passed link function needs to be callable and have a callable
        # .inverse property!
        # ```
        # As a workaround, use an identify function that's NOT decorated by `numba.jit`.
        link=identity_function,
    )
    artifact_path = "model"
    with mlflow.start_run():
        model_info = mlflow.shap.log_explainer(model, artifact_path)

    resp = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=pd.DataFrame(X[:3]),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
    )
    decoded_json = json.loads(resp.content.decode("utf-8"))
    scores = pd.DataFrame(data=decoded_json["predictions"]).values
    np.testing.assert_allclose(scores, model(X[:3]).values, rtol=100, atol=100)


@pytest.mark.skipif(Version(shap.__version__) > Version("0.41.0"), reason="numba jit compatible")
def test_pyfunc_serve_and_score():
    # Note: this implementation of an identify function is only compatible with versions of
    # shap <= 0.41.0. A breaking change was introduced with how numba is used with shap in version
    # 0.42.0.
    X, y = get_housing_data()

    reg = sklearn.ensemble.RandomForestRegressor(n_estimators=10).fit(X, y)
    model = shap.Explainer(
        reg.predict,
        masker=X,
        algorithm="permutation",
        # `link` defaults to `shap.links.identity` which is decorated by `numba.jit` and causes
        # the following error when loading the explainer for serving:
        # ```
        # Exception: The passed link function needs to be callable and have a callable
        # .inverse property!
        # ```
        # As a workaround, use an identify function that's NOT decorated by `numba.jit`.
        link=create_identity_function(),
    )
    artifact_path = "model"
    with mlflow.start_run():
        model_info = mlflow.shap.log_explainer(model, artifact_path)

    resp = pyfunc_serve_and_score_model(
        model_info.model_uri,
        data=pd.DataFrame(X[:3]),
        content_type=pyfunc_scoring_server.CONTENT_TYPE_JSON,
    )
    decoded_json = json.loads(resp.content.decode("utf-8"))
    scores = pd.DataFrame(data=decoded_json["predictions"]).values
    np.testing.assert_allclose(scores, model(X[:3]).values, rtol=100, atol=100)


def test_log_model_with_code_paths(shap_model):
    artifact_path = "model"
    with (
        mlflow.start_run(),
        mock.patch("mlflow.shap._add_code_from_conf_to_system_path") as add_mock,
    ):
        model_info = mlflow.shap.log_explainer(shap_model, artifact_path, code_paths=[__file__])
        _compare_logged_code_paths(__file__, model_info.model_uri, mlflow.shap.FLAVOR_NAME)
        mlflow.shap.load_explainer(model_info.model_uri)
        add_mock.assert_called()


def test_model_save_load_with_metadata(shap_model, tmp_path):
    model_path = str(tmp_path.joinpath("pyfunc_test"))
    mlflow.shap.save_explainer(
        shap_model, path=model_path, metadata={"metadata_key": "metadata_value"}
    )

    reloaded_model = mlflow.pyfunc.load_model(model_uri=model_path)
    assert reloaded_model.metadata.metadata["metadata_key"] == "metadata_value"


def test_model_log_with_metadata(shap_model):
    artifact_path = "model"

    with mlflow.start_run():
        model_info = mlflow.shap.log_explainer(
            shap_model, artifact_path=artifact_path, metadata={"metadata_key": "metadata_value"}
        )

    reloaded_model = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)
    assert reloaded_model.metadata.metadata["metadata_key"] == "metadata_value"
