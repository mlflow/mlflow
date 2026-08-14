import warnings
from unittest import mock

import pytest

import mlflow.genai as mlflow_genai


def test_warns_when_numpy_below_2_and_databricks_connect_present(monkeypatch):
    # databricks-connect pins numpy<2, so its presence alongside an old numpy is a
    # strong signal that installing databricks-agents downgraded numpy (see #24690).
    monkeypatch.setattr(mlflow_genai.np, "__version__", "1.26.4")
    with mock.patch("mlflow.genai.importlib.util.find_spec") as mock_find_spec:
        with pytest.warns(UserWarning, match="numpy 1.26.4"):
            mlflow_genai._warn_if_databricks_connect_downgraded_numpy()
    mock_find_spec.assert_called_once_with("databricks.connect")


def test_no_warning_when_numpy_is_2_or_above(monkeypatch):
    monkeypatch.setattr(mlflow_genai.np, "__version__", "2.1.0")
    with mock.patch("mlflow.genai.importlib.util.find_spec") as mock_find_spec:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mlflow_genai._warn_if_databricks_connect_downgraded_numpy()
    mock_find_spec.assert_not_called()


def test_no_warning_when_databricks_connect_not_installed(monkeypatch):
    monkeypatch.setattr(mlflow_genai.np, "__version__", "1.26.4")
    with mock.patch("mlflow.genai.importlib.util.find_spec", return_value=None):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mlflow_genai._warn_if_databricks_connect_downgraded_numpy()


def test_no_warning_when_databricks_namespace_package_missing(monkeypatch):
    # find_spec("databricks.connect") raises ModuleNotFoundError (rather than
    # returning None) when the "databricks" parent package isn't installed at all.
    monkeypatch.setattr(mlflow_genai.np, "__version__", "1.26.4")
    with mock.patch("mlflow.genai.importlib.util.find_spec", side_effect=ModuleNotFoundError):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mlflow_genai._warn_if_databricks_connect_downgraded_numpy()
