import signal
import socket
import subprocess
import sys
import time
from datetime import date
from unittest import mock

import pytest

from mlflow import server
from mlflow.environment_variables import _MLFLOW_SERVER_BOOT_ID, _MLFLOW_SGI_NAME
from mlflow.exceptions import MlflowException
from mlflow.store.tracking.dbmodels.models import SqlTraceMetricDailyRollup
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
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
        "--log-config",
        str(server._UVICORN_LOG_CONFIG),
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
        "--log-config",
        str(server._UVICORN_LOG_CONFIG),
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
        "--log-config",
        str(server._UVICORN_LOG_CONFIG),
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
    assert "--log-config" in cmd
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


def test_run_server_rejects_invalid_enabled_rollup_schedule(mock_exec_cmd, monkeypatch):
    monkeypatch.setenv("MLFLOW_SERVER_ENABLE_JOB_EXECUTION", "true")
    monkeypatch.setenv("MLFLOW_SQL_TRACE_ROLLUPS_ENABLED", "true")
    monkeypatch.setenv("MLFLOW_TRACE_ROLLUPS_SCHEDULE", "invalid")

    with (
        mock.patch("sys.platform", return_value="linux"),
        mock.patch("mlflow.server.jobs.utils._check_requirements"),
        pytest.raises(MlflowException, match="five-field UTC cron"),
    ):
        server._run_server(
            file_store_path="sqlite:///primary.db",
            registry_store_uri="",
            default_artifact_root="file:///artifacts",
            serve_artifacts="",
            artifacts_only="",
            artifacts_destination="",
            host="localhost",
            port="5000",
        )

    mock_exec_cmd.assert_not_called()


@pytest.mark.parametrize(
    "variable",
    ["MLFLOW_TRACE_ROLLUPS_MAX_PARTITIONS_PER_RUN", "MLFLOW_TRACE_ROLLUPS_MAX_WORKERS"],
)
@pytest.mark.parametrize("value", ["abc", "0", "-1"])
def test_run_server_rejects_invalid_enabled_rollup_limits(
    mock_exec_cmd, monkeypatch, variable, value
):
    monkeypatch.setenv("MLFLOW_SERVER_ENABLE_JOB_EXECUTION", "true")
    monkeypatch.setenv("MLFLOW_SQL_TRACE_ROLLUPS_ENABLED", "true")
    monkeypatch.setenv(variable, value)

    with (
        mock.patch("sys.platform", return_value="linux"),
        mock.patch("mlflow.server.jobs.utils._check_requirements"),
        pytest.raises(MlflowException, match=variable),
    ):
        server._run_server(
            file_store_path="sqlite:///primary.db",
            registry_store_uri="",
            default_artifact_root="file:///artifacts",
            serve_artifacts="",
            artifacts_only="",
            artifacts_destination="",
            host="localhost",
            port="5000",
        )

    mock_exec_cmd.assert_not_called()


def test_run_server_rejects_missing_job_backend_when_rollups_are_enabled(
    mock_exec_cmd, monkeypatch
):
    monkeypatch.setenv("MLFLOW_SERVER_ENABLE_JOB_EXECUTION", "true")
    monkeypatch.setenv("MLFLOW_SQL_TRACE_ROLLUPS_ENABLED", "true")

    with (
        mock.patch("sys.platform", return_value="linux"),
        mock.patch(
            "mlflow.server.jobs.utils._check_requirements",
            side_effect=MlflowException("database backend required"),
        ),
        pytest.raises(MlflowException, match="available SQL job-execution backend"),
    ):
        server._run_server(
            file_store_path="",
            registry_store_uri="",
            default_artifact_root="",
            serve_artifacts="",
            artifacts_only="",
            artifacts_destination="",
            host="localhost",
            port="5000",
        )

    mock_exec_cmd.assert_not_called()


def test_run_server_rejects_disabled_rollups_when_materialized_rows_exist(
    mock_exec_cmd, monkeypatch, tmp_path
):
    database_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    store = SqlAlchemyStore(database_uri, artifact_root.as_uri())
    experiment_id = store.create_experiment("rollup-startup-check")
    with store.ManagedSessionMaker(read_only=False) as session:
        session.add(
            SqlTraceMetricDailyRollup(
                experiment_id=int(experiment_id),
                rollup_day=date(1970, 1, 1),
                metric_name="trace_count",
                grouping_set="global",
                sample_count=1,
            )
        )

    monkeypatch.setenv("MLFLOW_SERVER_ENABLE_JOB_EXECUTION", "false")
    monkeypatch.setenv("MLFLOW_SQL_TRACE_ROLLUPS_ENABLED", "false")

    with (
        mock.patch("sys.platform", return_value="linux"),
        pytest.raises(MlflowException, match="mlflow db delete-trace-rollups"),
    ):
        server._run_server(
            file_store_path=database_uri,
            registry_store_uri="",
            default_artifact_root=artifact_root.as_uri(),
            serve_artifacts="",
            artifacts_only="",
            artifacts_destination="",
            host="localhost",
            port="5000",
        )

    mock_exec_cmd.assert_not_called()


def test_run_server_allows_disabled_rollups_for_a_new_sql_database(
    mock_exec_cmd, monkeypatch, tmp_path
):
    monkeypatch.setenv("MLFLOW_SERVER_ENABLE_JOB_EXECUTION", "false")
    monkeypatch.setenv("MLFLOW_SQL_TRACE_ROLLUPS_ENABLED", "false")
    database_path = tmp_path / "new.db"

    with mock.patch("sys.platform", return_value="linux"):
        server._run_server(
            file_store_path=f"sqlite:///{database_path}",
            registry_store_uri="",
            default_artifact_root="",
            serve_artifacts="",
            artifacts_only="",
            artifacts_destination="",
            host="localhost",
            port="5000",
        )

    mock_exec_cmd.assert_called_once()
    assert not database_path.exists()


def test_run_server_passes_public_store_config_to_job_runner(mock_exec_cmd, monkeypatch):
    monkeypatch.setenv("MLFLOW_SERVER_ENABLE_JOB_EXECUTION", "true")
    monkeypatch.setenv("MLFLOW_SQL_TRACE_ROLLUPS_ENABLED", "false")
    mock_exec_cmd.return_value.pid = 123

    with (
        mock.patch("sys.platform", return_value="linux"),
        mock.patch("mlflow.server.jobs.utils._check_requirements"),
        mock.patch("mlflow.server.jobs.utils._launch_job_runner") as launch_job_runner,
        mock.patch("mlflow.tracing.trace_rollup_service.validate_sql_trace_rollup_startup"),
    ):
        server._run_server(
            file_store_path="sqlite:///primary.db",
            registry_store_uri="",
            default_artifact_root="file:///artifacts",
            serve_artifacts="",
            artifacts_only="",
            artifacts_destination="",
            host="localhost",
            port="5000",
        )

    job_env = launch_job_runner.call_args.args[0]
    assert job_env["MLFLOW_BACKEND_STORE_URI"] == "sqlite:///primary.db"
    assert job_env["MLFLOW_DEFAULT_ARTIFACT_ROOT"] == "file:///artifacts"


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
        "--log-config",
        str(server._UVICORN_LOG_CONFIG),
        "--host",
        "localhost",
        "--port",
        "5000",
        "--workers",
        "4",
        "mlflow.server.fastapi_app:app",
    ]
    mock_exec_cmd.assert_called_once()
    call = mock_exec_cmd.call_args
    assert call.args[0] == expected_command
    assert call.kwargs["capture_output"] is False
    assert call.kwargs["synchronous"] is False
    extra_env = call.kwargs["extra_env"]
    assert extra_env[_MLFLOW_SGI_NAME.name] == "uvicorn"
    # Each server generation is stamped with a boot id (used to reap orphaned sandbox containers
    # left by a previous generation); its value is a random per-boot uuid.
    assert extra_env[_MLFLOW_SERVER_BOOT_ID.name]


@pytest.mark.parametrize(
    "uvicorn_opts",
    [
        "--log-config /custom/path.yaml",
        "--log-config=/custom/path.yaml",
    ],
)
def test_build_uvicorn_command_user_log_config_takes_precedence(uvicorn_opts):
    cmd = server._build_uvicorn_command(
        uvicorn_opts, "localhost", "5000", "4", "mlflow.server.fastapi_app:app"
    )
    assert not any("uvicorn_log_config.yaml" in o for o in cmd)


# flaky: auto-detected from CI re-runs; see the weekly flaky-test report
@pytest.mark.flaky(attempts=2)
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
