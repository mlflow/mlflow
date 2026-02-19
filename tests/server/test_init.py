import signal
import socket
import subprocess
import sys
import time
from unittest import mock

import pytest

from mlflow import server
from mlflow.environment_variables import _MLFLOW_SGI_NAME
from mlflow.exceptions import MlflowException
from mlflow.utils import find_free_port
from mlflow.utils.os import is_windows


@pytest.fixture
def mock_exec_cmd():
    with mock.patch("mlflow.server._exec_cmd") as m:
        yield m


def _wait_for_port(host: str, port: int, proc: subprocess.Popen, timeout: int = 15) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            raise AssertionError(
                "MLflow server exited before accepting connections.\n"
                f"stdout:\n{stdout}\n"
                f"stderr:\n{stderr}"
            )
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(0.1)
    raise AssertionError(f"Timed out waiting for {host}:{port} to accept connections")


def _wait_for_port_closed(host: str, port: int, timeout: int = 15) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                time.sleep(0.1)
        except OSError:
            return
    raise AssertionError(f"Timed out waiting for {host}:{port} to close")


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


@pytest.mark.parametrize(
    "sig",
    [
        pytest.param(
            signal.SIGTERM,
            marks=pytest.mark.skipif(is_windows(), reason="SIGTERM is a hard kill on Windows"),
        ),
        signal.SIGINT,
    ],
)
def test_mlflow_server_shuts_down_on_signal(sig: signal.Signals, tmp_path):
    port = find_free_port()
    db_path = tmp_path / "mlflow.db"
    cmd = [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--workers",
        "1",
        "--backend-store-uri",
        f"sqlite:///{db_path}",
    ]
    if is_windows():
        proc = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        proc = subprocess.Popen(cmd)
    try:
        _wait_for_port("127.0.0.1", port, proc, timeout=60 if is_windows() else 15)
        if is_windows():
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.send_signal(sig)
        proc.wait(timeout=30 if is_windows() else 15)
        _wait_for_port_closed("127.0.0.1", port)
        # Exit code 0 means graceful shutdown (signal was caught and handled)
        # -sig or 128+sig means the process was killed by the signal
        # On Windows, CTRL_BREAK_EVENT maps to 0xC000013A.
        if is_windows():
            assert proc.returncode in (0, 0xC000013A)
        else:
            assert proc.returncode in (0, -sig, 128 + sig)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
