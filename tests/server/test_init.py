from unittest import mock

import entrypoints
import pytest

from mlflow.exceptions import MlflowException
from mlflow import server


@pytest.fixture()
def mock_exec_cmd():
    with mock.patch("mlflow.server._exec_cmd") as m:
        yield m


def test_get_app_name():
    # Hard to test if the test plugin is installed.
    assert server._get_app_name() == f"{server.__name__}:app"


def test_get_app_name_two_plugins():
    """Two server apps cannot exist, which one would be served?"""
    plugins = [
        entrypoints.EntryPoint("one", "one.app", "app"),
        entrypoints.EntryPoint("two", "two.app", "two"),
    ]
    with mock.patch("mlflow.server.entrypoints.get_group_all", return_value=plugins):
        with pytest.raises(MlflowException, match=r"one.*two"):
            server._get_app_name()


def test_get_app_name_custom_app_plugin():
    """This test requires the package in tests/resources/mlflow-test-plugin to be installed"""
    from mlflow_test_plugin import app

    assert server._get_app_name() == f"{app.__name__}:custom_app"


def test_build_waitress_command():
    assert server._build_waitress_command("", "localhost", "5000", f"{server.__name__}:app") == [
        "waitress-serve",
        "--host=localhost",
        "--port=5000",
        "--ident=mlflow",
        "mlflow.server:app",
    ]


def test_build_gunicorn_command():
    assert server._build_gunicorn_command(
        "", "localhost", "5000", "4", f"{server.__name__}:app"
    ) == ["gunicorn", "-b", "localhost:5000", "-w", "4", "mlflow.server:app"]


def test_run_server(mock_exec_cmd):
    """Make sure this runs."""
    with mock.patch("sys.platform", "linux"):
        server._run_server("", "", "", "", "", "", "", "")
    mock_exec_cmd.assert_called_once()


def test_run_server_win32(mock_exec_cmd):
    """Make sure this runs."""
    with mock.patch("sys.platform", "win32"):
        server._run_server("", "", "", "", "", "", "", "")
    mock_exec_cmd.assert_called_once()
