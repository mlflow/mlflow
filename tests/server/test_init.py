import os
import shutil
import sys
from unittest import mock

import pytest

from mlflow import server
from mlflow.environment_variables import _MLFLOW_SGI_NAME
from mlflow.exceptions import MlflowException


@pytest.fixture
def mock_exec_cmd():
    with mock.patch("mlflow.server._exec_cmd") as m:
        yield m


def test_find_app_custom_app_plugin():
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


def test_build_uvicorn_command():
    assert server._build_uvicorn_command(
        "", "localhost", "5000", "4", "mlflow.server.fastapi_app:app"
    ) == [
        sys.executable,
        "-m",
        "uvicorn",
        "--host",
        "localhost",
        "--port",
        "5000",
        "--workers",
        "4",
        "mlflow.server.fastapi_app:app",
    ]

    # Test with custom uvicorn options
    assert server._build_uvicorn_command(
        "--reload --log-level debug", "localhost", "5000", "4", "mlflow.server.fastapi_app:app"
    ) == [
        sys.executable,
        "-m",
        "uvicorn",
        "--reload",
        "--log-level",
        "debug",
        "--host",
        "localhost",
        "--port",
        "5000",
        "--workers",
        "4",
        "mlflow.server.fastapi_app:app",
    ]

    assert server._build_uvicorn_command(
        "", "localhost", "5000", "4", "mlflow.server.fastapi_app:app", None, is_factory=True
    ) == [
        sys.executable,
        "-m",
        "uvicorn",
        "--host",
        "localhost",
        "--port",
        "5000",
        "--workers",
        "4",
        "--factory",
        "mlflow.server.fastapi_app:app",
    ]


def test_build_uvicorn_command_with_env_file():
    cmd = server._build_uvicorn_command(
        uvicorn_opts=None,
        host="localhost",
        port=5000,
        workers=4,
        app_name="app:app",
        env_file="/path/to/.env",
    )

    assert "--env-file" in cmd
    assert "/path/to/.env" in cmd
    # Verify the order - env-file should come before the app name
    env_file_idx = cmd.index("--env-file")
    env_file_path_idx = cmd.index("/path/to/.env")
    app_name_idx = cmd.index("app:app")
    assert env_file_idx < app_name_idx
    assert env_file_path_idx == env_file_idx + 1
    assert env_file_path_idx < app_name_idx


def test_run_server(mock_exec_cmd, monkeypatch):
    monkeypatch.setenv("MLFLOW_SERVER_ENABLE_JOB_EXECUTION", "false")
    with mock.patch("sys.platform", return_value="linux"):
        server._run_server(
            file_store_path="",
            registry_store_uri="",
            default_artifact_root="",
            serve_artifacts="",
            artifacts_only="",
            artifacts_destination="",
            host="",
            port="",
        )
    mock_exec_cmd.assert_called_once()


def test_run_server_win32(mock_exec_cmd, monkeypatch):
    monkeypatch.setenv("MLFLOW_SERVER_ENABLE_JOB_EXECUTION", "false")
    with mock.patch("sys.platform", return_value="win32"):
        server._run_server(
            file_store_path="",
            registry_store_uri="",
            default_artifact_root="",
            serve_artifacts="",
            artifacts_only="",
            artifacts_destination="",
            host="",
            port="",
        )
    mock_exec_cmd.assert_called_once()


def test_run_server_with_uvicorn(mock_exec_cmd, monkeypatch):
    monkeypatch.setenv("MLFLOW_SERVER_ENABLE_JOB_EXECUTION", "false")
    with mock.patch("sys.platform", return_value="linux"):
        server._run_server(
            file_store_path="",
            registry_store_uri="",
            default_artifact_root="",
            serve_artifacts="",
            artifacts_only="",
            artifacts_destination="",
            host="localhost",
            port="5000",
            uvicorn_opts="--reload",
        )
    expected_command = [
        sys.executable,
        "-m",
        "uvicorn",
        "--reload",
        "--host",
        "localhost",
        "--port",
        "5000",
        "--workers",
        "4",
        "mlflow.server.fastapi_app:app",
    ]
    mock_exec_cmd.assert_called_once_with(
        expected_command,
        extra_env={_MLFLOW_SGI_NAME.name: "uvicorn"},
        capture_output=False,
        synchronous=False,
    )


@pytest.mark.skipif(os.name == "nt", reason="MLflow job execution is not supported on Windows")
def test_run_server_with_jobs_without_uv(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MLFLOW_SERVER_ENABLE_JOB_EXECUTION", "true")
    original_which = shutil.which

    def patched_which(cmd):
        if cmd == "uv":
            return None
        return original_which(cmd)

    with (
        mock.patch("shutil.which", side_effect=patched_which) as which_patch,
        pytest.raises(MlflowException, match="MLflow job backend requires 'uv'"),
    ):
        server._run_server(
            file_store_path="",
            registry_store_uri="",
            default_artifact_root="",
            serve_artifacts="",
            artifacts_only="",
            artifacts_destination="",
            host="",
            port="",
        )
    which_patch.assert_called_once_with("uv")
