import os
import sys
from unittest import mock

import pytest
import sklearn.neighbors as knn
from sklearn import datasets

import mlflow.sklearn
import mlflow.utils.model_utils as mlflow_model_utils
from mlflow.environment_variables import MLFLOW_RECORD_ENV_VARS_IN_MODEL_LOGGING
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.utils.file_utils import TempDir
from mlflow.utils.model_utils import env_var_tracker


@pytest.fixture(scope="module")
def sklearn_knn_model():
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    y = iris.target
    knn_model = knn.KNeighborsClassifier()
    knn_model.fit(X, y)
    return knn_model


@pytest.fixture
def model_path(tmp_path):
    return os.path.join(tmp_path, "model")


def test_get_flavor_configuration_throws_exception_when_requested_flavor_is_missing(
    model_path, sklearn_knn_model
):
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model, path=model_path)

    # The saved model contains the "sklearn" flavor, so this call should succeed
    sklearn_flavor_config = mlflow_model_utils._get_flavor_configuration(
        model_path=model_path, flavor_name=mlflow.sklearn.FLAVOR_NAME
    )
    assert sklearn_flavor_config is not None


def test_get_flavor_configuration_with_present_flavor_returns_expected_configuration(
    sklearn_knn_model, model_path
):
    mlflow.sklearn.save_model(sk_model=sklearn_knn_model, path=model_path)

    sklearn_flavor_config = mlflow_model_utils._get_flavor_configuration(
        model_path=model_path, flavor_name=mlflow.sklearn.FLAVOR_NAME
    )
    model_config = Model.load(os.path.join(model_path, "MLmodel"))
    assert sklearn_flavor_config == model_config.flavors[mlflow.sklearn.FLAVOR_NAME]


def test_add_code_to_system_path(sklearn_knn_model, model_path):
    mlflow.sklearn.save_model(
        sk_model=sklearn_knn_model,
        path=model_path,
        code_paths=[
            "tests/utils/test_resources/dummy_module.py",
            "tests/utils/test_resources/dummy_package",
        ],
    )

    sklearn_flavor_config = mlflow_model_utils._get_flavor_configuration(
        model_path=model_path, flavor_name=mlflow.sklearn.FLAVOR_NAME
    )
    with TempDir(chdr=True):
        # Load the model from a new directory that is not a parent of the source code path to
        # verify that source code paths and their subdirectories are resolved correctly
        with pytest.raises(ModuleNotFoundError, match="No module named 'dummy_module'"):
            import dummy_module

        mlflow_model_utils._add_code_from_conf_to_system_path(model_path, sklearn_flavor_config)
        import dummy_module  # noqa: F401

    # If this raises an exception it's because dummy_package.test imported
    # dummy_package.operator and not the built-in operator module. This only
    # happens if MLflow is misconfiguring the system path.
    from dummy_package import base  # noqa: F401

    # Ensure that the custom tests/utils/test_resources/dummy_package/pandas.py is not
    # overwriting the 3rd party `pandas` package
    assert "dummy_package" in sys.modules
    assert "pandas" in sys.modules
    assert "site-packages" in sys.modules["pandas"].__file__


@mock.patch("builtins.open", side_effect=OSError("[Errno 95] Operation not supported"))
def test_add_code_to_system_path_not_copyable_file(sklearn_knn_model, model_path):
    with pytest.raises(MlflowException, match=r"Failed to copy the specified code path"):
        mlflow.sklearn.save_model(
            sk_model=sklearn_knn_model,
            path=model_path,
            code_paths=["tests/utils/test_resources/dummy_module.py"],
        )


def test_env_var_tracker(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "host")
    assert "DATABRICKS_HOST" in os.environ
    assert "TEST_API_KEY" not in os.environ

    with env_var_tracker() as tracked_env_names:
        assert os.environ["DATABRICKS_HOST"] == "host"
        monkeypatch.setenv("TEST_API_KEY", "key")
        # accessed env var is tracked
        assert os.environ.get("TEST_API_KEY") == "key"
        # test non-existing env vars fetched by `get` are not tracked
        os.environ.get("INVALID_API_KEY", "abc")
        # test non-existing env vars are not tracked
        try:
            os.environ["ANOTHER_API_KEY"]
        except KeyError:
            pass
        assert all(x in tracked_env_names for x in ["DATABRICKS_HOST", "TEST_API_KEY"])
        assert all(x not in tracked_env_names for x in ["INVALID_API_KEY", "ANOTHER_API_KEY"])

    assert isinstance(os.environ, os._Environ)
    assert all(x in os.environ for x in ["DATABRICKS_HOST", "TEST_API_KEY"])
    assert all(x not in os.environ for x in ["INVALID_API_KEY", "ANOTHER_API_KEY"])

    monkeypatch.setenv(MLFLOW_RECORD_ENV_VARS_IN_MODEL_LOGGING.name, "false")
    with env_var_tracker() as env:
        os.environ.get("API_KEY")
        assert env == set()
