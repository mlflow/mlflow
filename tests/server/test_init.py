import sys
from unittest import mock

import pytest

from mlflow import server
from mlflow.exceptions import MlflowException


@pytest.fixture
def mock_exec_cmd():
    with mock.patch("mlflow.server._exec_cmd") as m:
        yield m


def test_find_app_custom_app_plugin():
    """This test requires the package in tests/resources/mlflow-test-plugin to be installed"""
    assert server._find_app("custom_app") == "mlflow_test_plugin.app:custom_app"


def test_find_app_non_existing_app():
    with pytest.raises(MlflowException, match=r"Failed to find app 'does_not_exist'"):
        server._find_app("does_not_exist")


def test_build_waitress_command():
    assert server._build_waitress_command(
        "", "localhost", "5000", f"{server.__name__}:app", is_factory=True
    ) == [
        sys.executable,
        "-m",
        "waitress",
        "--host=localhost",
        "--port=5000",
        "--ident=mlflow",
        "--call",
        "mlflow.server:app",
    ]
    assert server._build_waitress_command(
        "", "localhost", "5000", f"{server.__name__}:app", is_factory=False
    ) == [
        sys.executable,
        "-m",
        "waitress",
        "--host=localhost",
        "--port=5000",
        "--ident=mlflow",
        "mlflow.server:app",
    ]


def test_build_gunicorn_command():
    assert server._build_gunicorn_command(
        "", "localhost", "5000", "4", f"{server.__name__}:app"
    ) == [
        sys.executable,
        "-m",
        "gunicorn",
        "-b",
        "localhost:5000",
        "-w",
        "4",
        "mlflow.server:app",
    ]


def test_run_server(mock_exec_cmd):
    """Make sure this runs."""
    with mock.patch("sys.platform", return_value="linux"):
        server._run_server("", "", "", "", "", "", "", "")
    mock_exec_cmd.assert_called_once()


def test_run_server_win32(mock_exec_cmd):
    """Make sure this runs."""
    with mock.patch("sys.platform", return_value="win32"):
        server._run_server("", "", "", "", "", "", "", "")
    mock_exec_cmd.assert_called_once()
