import os
import sys
from unittest import mock

import pytest
import sklearn.linear_model as logreg_module
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
    logreg_model = logreg_module.LogisticRegression()
    logreg_model.fit(X, y)
    return logreg_model


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


def test_add_code_to_system_path_not_copyable_file(sklearn_knn_model, model_path):
    with mock.patch("builtins.open", side_effect=OSError("[Errno 95] Operation not supported")):
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


@pytest.mark.parametrize(
    ("model_code", "pickle", "expected"),
    [
        (None, None, True),
        ("false", None, False),
        (None, "false", False),
        ("true", "false", False),
        ("false", "true", False),
        ("true", "true", True),
    ],
)
def test_is_pickle_loading_allowed(monkeypatch, model_code, pickle, expected):
    for name, value in [
        ("MLFLOW_ALLOW_MODEL_CODE_EXECUTION", model_code),
        ("MLFLOW_ALLOW_PICKLE_DESERIALIZATION", pickle),
    ]:
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert mlflow_model_utils._is_pickle_loading_allowed() is expected


def test_is_pickle_loading_allowed_warns_on_deprecated_env_var(monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_PICKLE_DESERIALIZATION", "true")
    with pytest.warns(FutureWarning, match="MLFLOW_ALLOW_MODEL_CODE_EXECUTION"):
        assert mlflow_model_utils._is_pickle_loading_allowed()


@pytest.mark.parametrize(
    "databricks_env_fn", ["is_in_databricks_runtime", "is_in_databricks_model_serving_environment"]
)
def test_is_pickle_loading_allowed_in_databricks(monkeypatch, databricks_env_fn):
    monkeypatch.setenv("MLFLOW_ALLOW_MODEL_CODE_EXECUTION", "false")
    with mock.patch(f"mlflow.utils.model_utils.{databricks_env_fn}", return_value=True):
        assert mlflow_model_utils._is_pickle_loading_allowed()
        assert not mlflow_model_utils._is_pickle_loading_allowed(allow_in_databricks=False)


@pytest.mark.parametrize(
    ("model_code", "pickle", "expected"),
    [
        (None, None, True),
        ("false", None, False),
        # The deprecated variable keeps refusing only pickle, as before.
        (None, "false", True),
        ("true", "false", True),
    ],
)
def test_is_model_code_loading_allowed(monkeypatch, model_code, pickle, expected):
    for name, value in [
        ("MLFLOW_ALLOW_MODEL_CODE_EXECUTION", model_code),
        ("MLFLOW_ALLOW_PICKLE_DESERIALIZATION", pickle),
    ]:
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert mlflow_model_utils._is_model_code_loading_allowed() is expected


@pytest.mark.parametrize(
    ("pickle", "mentions_deprecated_variable"),
    [(None, False), ("true", False), ("false", True)],
)
def test_deprecated_pickle_variable_hint(monkeypatch, pickle, mentions_deprecated_variable):
    if pickle is None:
        monkeypatch.delenv("MLFLOW_ALLOW_PICKLE_DESERIALIZATION", raising=False)
    else:
        monkeypatch.setenv("MLFLOW_ALLOW_PICKLE_DESERIALIZATION", pickle)
    hint = mlflow_model_utils._deprecated_pickle_variable_hint()
    assert ("MLFLOW_ALLOW_PICKLE_DESERIALIZATION" in hint) is mentions_deprecated_variable


def test_add_code_refuses_code_directory_when_model_code_execution_disallowed(
    tmp_path, monkeypatch
):
    (tmp_path / "code").mkdir()
    monkeypatch.setenv("MLFLOW_ALLOW_MODEL_CODE_EXECUTION", "false")
    sys_path = list(sys.path)
    with pytest.raises(MlflowException, match="a model with a code directory"):
        mlflow_model_utils._add_code_from_conf_to_system_path(str(tmp_path), {"code": "code"})
    assert sys.path == sys_path


def test_add_code_refuses_model_on_python_path_when_model_code_execution_disallowed(
    tmp_path, monkeypatch
):
    # Without a code directory, a model can still import its own files when the model directory
    # is under a `sys.path` entry, such as the current directory.
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("MLFLOW_ALLOW_MODEL_CODE_EXECUTION", "false")
    with pytest.raises(MlflowException, match="on the Python path"):
        mlflow_model_utils._add_code_from_conf_to_system_path(str(model_dir), {})


def test_load_model_from_code_refused_when_model_code_execution_disallowed(tmp_path, monkeypatch):
    marker = tmp_path / "marker"
    code_file = tmp_path / "model.py"
    code_file.write_text(
        "import mlflow\n"
        f"open({str(marker)!r}, 'w').close()\n"
        "mlflow.models.set_model(lambda model_input: model_input)\n"
    )
    model_dir = tmp_path / "model"
    monkeypatch.setenv("MLFLOW_ALLOW_MODEL_CODE_EXECUTION", "false")
    # Saving runs the user's own code, so it stays allowed.
    mlflow.pyfunc.save_model(model_dir, python_model=str(code_file), pip_requirements=["mlflow"])
    marker.unlink()

    with pytest.raises(MlflowException, match="Loading PyFunc models is disallowed"):
        mlflow.pyfunc.load_model(str(model_dir))
    # Inferring pip requirements loads the model in a subprocess that skips the PyFunc load check.
    mlflow.models.infer_pip_requirements(str(model_dir), "python_function", fallback=[])
    assert not marker.exists()


def test_save_model_with_infer_code_paths_refused_when_model_code_execution_disallowed(
    tmp_path, monkeypatch
):
    code_file = tmp_path / "model.py"
    code_file.write_text(
        "import mlflow\nmlflow.models.set_model(lambda model_input: model_input)\n"
    )
    monkeypatch.setenv("MLFLOW_ALLOW_MODEL_CODE_EXECUTION", "false")
    with pytest.raises(MlflowException, match="infer_code_paths=True"):
        mlflow.pyfunc.save_model(
            tmp_path / "model",
            python_model=str(code_file),
            pip_requirements=["mlflow"],
            infer_code_paths=True,
        )
