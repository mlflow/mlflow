import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import date
from unittest import mock

import pytest

from mlflow import server
from mlflow.environment_variables import (
    _MLFLOW_AUTH_ADMIN_BOOTSTRAPPED,
    _MLFLOW_IN_JOB_EXECUTOR,
    _MLFLOW_SERVER_BOOT_ID,
    _MLFLOW_SERVER_TLS_LISTENER,
    _MLFLOW_SGI_NAME,
    MLFLOW_ARTIFACTS_PRESIGNED_ONLY,
    MLFLOW_FLASK_SERVER_SECRET_KEY,
)
from mlflow.exceptions import MlflowException
from mlflow.server.jobs.executor import EXECUTOR_JOB_RUNNER_FATAL_EXIT_CODE
from mlflow.store.tracking.dbmodels.models import SqlTraceMetricDailyRollup
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.utils import find_free_port
from mlflow.utils.os import is_windows
from mlflow.utils.process import ShellCommandException


@pytest.fixture
def mock_exec_cmd():
    with mock.patch("mlflow.server._exec_cmd") as m:
        m.return_value.wait.return_value = 0
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


@pytest.mark.parametrize(
    ("uvicorn_opts", "expected"),
    [
        (None, False),
        ("--log-level debug", False),
        ("--ssl-keyfile key.pem --ssl-certfile cert.pem", True),
        ("--ssl-certfile=cert.pem", True),
    ],
)
def test_uvicorn_opts_enable_tls(uvicorn_opts, expected):
    assert server._uvicorn_opts_enable_tls(uvicorn_opts) is expected


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


@pytest.mark.parametrize(
    ("uvicorn_opts", "inherited", "expected"),
    [
        ("--ssl-keyfile key.pem --ssl-certfile cert.pem", None, "true"),
        (None, None, "false"),
        # A marker inherited from a parent server must not leak into a child without TLS.
        (None, "true", "false"),
    ],
)
def test_run_server_marks_a_tls_listener_for_workers(
    mock_exec_cmd, monkeypatch, uvicorn_opts, inherited, expected
):
    monkeypatch.setenv("MLFLOW_SERVER_ENABLE_JOB_EXECUTION", "false")
    if inherited is not None:
        monkeypatch.setenv(_MLFLOW_SERVER_TLS_LISTENER.name, inherited)
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
            uvicorn_opts=uvicorn_opts,
        )

    extra_env = mock_exec_cmd.call_args.kwargs["extra_env"]
    assert extra_env[_MLFLOW_SERVER_TLS_LISTENER.name] == expected


def test_run_server_rejects_invalid_enabled_rollup_schedule(mock_exec_cmd, monkeypatch):
    monkeypatch.setenv("MLFLOW_SERVER_ENABLE_JOB_EXECUTION", "true")
    monkeypatch.setenv("MLFLOW_SQL_TRACE_ROLLUPS_ENABLED", "true")
    monkeypatch.setenv("MLFLOW_TRACE_ROLLUPS_SCHEDULE", "invalid")

    with (
        mock.patch(
            "mlflow.server.jobs.local_executor.LocalJobExecutor.check_requirements"
        ) as check_executor_requirements,
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

    check_executor_requirements.assert_called_once()
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
        mock.patch(
            "mlflow.server.jobs.local_executor.LocalJobExecutor.check_requirements"
        ) as check_executor_requirements,
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

    check_executor_requirements.assert_called_once()
    mock_exec_cmd.assert_not_called()


def test_run_server_rejects_missing_job_backend_when_rollups_are_enabled(
    mock_exec_cmd, monkeypatch
):
    monkeypatch.setenv("MLFLOW_SERVER_ENABLE_JOB_EXECUTION", "true")
    monkeypatch.setenv("MLFLOW_SQL_TRACE_ROLLUPS_ENABLED", "true")

    with (
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


@pytest.mark.parametrize("static_prefix", [None, "", "/mlflow", "/nested/mlflow"])
@pytest.mark.parametrize("host", ["localhost", "0.0.0.0"])
def test_run_server_passes_public_store_config_to_job_runner(
    mock_exec_cmd, monkeypatch, static_prefix, host
):
    monkeypatch.setenv("MLFLOW_SERVER_ENABLE_JOB_EXECUTION", "true")
    monkeypatch.setenv("MLFLOW_SQL_TRACE_ROLLUPS_ENABLED", "false")
    monkeypatch.delenv("MLFLOW_GATEWAY_URI", raising=False)
    mock_exec_cmd.return_value.pid = 123

    with (
        mock.patch(
            "mlflow.server.jobs.local_executor.LocalJobExecutor.check_requirements"
        ) as check_executor_requirements,
        mock.patch("mlflow.server.jobs.utils._check_requirements") as check_requirements,
        mock.patch("mlflow.server.jobs.utils._launch_job_runner") as launch_job_runner,
        mock.patch(
            "mlflow.tracing.trace_rollup_service.validate_sql_trace_rollup_startup"
        ) as validate_rollup_startup,
    ):
        server._run_server(
            file_store_path="sqlite:///primary.db",
            registry_store_uri="",
            default_artifact_root="file:///artifacts",
            serve_artifacts="",
            artifacts_only="",
            artifacts_destination="",
            host=host,
            port="5000",
            static_prefix=static_prefix,
        )

    mock_exec_cmd.assert_called_once()
    check_requirements.assert_called_once_with("sqlite:///primary.db")
    check_executor_requirements.assert_called_once()
    validate_rollup_startup.assert_called_once_with("sqlite:///primary.db")
    launch_job_runner.assert_called_once()
    job_env = launch_job_runner.call_args.args[0]
    assert job_env["MLFLOW_BACKEND_STORE_URI"] == "sqlite:///primary.db"
    assert job_env["MLFLOW_DEFAULT_ARTIFACT_ROOT"] == "file:///artifacts"
    assert job_env["MLFLOW_TRACKING_URI"] == f"http://{host}:5000{static_prefix or ''}"
    assert job_env["MLFLOW_GATEWAY_URI"] == f"http://{host}:5000{static_prefix or ''}"


def test_run_server_propagates_presigned_only_mode(mock_exec_cmd, monkeypatch):
    monkeypatch.setenv("MLFLOW_SERVER_ENABLE_JOB_EXECUTION", "false")
    with mock.patch("sys.platform", return_value="linux"):
        server._run_server(
            file_store_path="",
            registry_store_uri="",
            default_artifact_root="",
            serve_artifacts=True,
            artifacts_only=False,
            artifacts_presigned_only=True,
            artifacts_destination="s3://bucket",
            host="",
            port="",
        )

    assert (
        mock_exec_cmd.call_args.kwargs["extra_env"][MLFLOW_ARTIFACTS_PRESIGNED_ONLY.name] == "true"
    )


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
    # The custom-scorer reconstruction marker is forced off for server workers so it can never be
    # inherited from the ambient environment (the server must never execute custom scorer code).
    assert extra_env[_MLFLOW_IN_JOB_EXECUTOR.name] == "false"


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


def test_run_server_bootstraps_basic_auth_admin_before_spawning_workers(mock_exec_cmd, monkeypatch):
    monkeypatch.setenv("MLFLOW_SERVER_ENABLE_JOB_EXECUTION", "false")
    monkeypatch.setenv(MLFLOW_FLASK_SERVER_SECRET_KEY.name, "my-secret-key")
    with mock.patch("mlflow.server.auth.bootstrap_admin_user") as bootstrap:
        server._run_server(
            file_store_path="",
            registry_store_uri="",
            default_artifact_root="",
            serve_artifacts="",
            artifacts_only="",
            artifacts_destination="",
            host="",
            port="",
            app_name="basic-auth",
        )
    bootstrap.assert_called_once_with()
    mock_exec_cmd.assert_called_once()
    # Workers are told the bootstrap already happened so they skip the PBKDF2 checks.
    assert mock_exec_cmd.call_args.kwargs["extra_env"][_MLFLOW_AUTH_ADMIN_BOOTSTRAPPED.name] == (
        "true"
    )


def test_run_server_fails_before_spawning_workers_when_admin_bootstrap_fails(
    mock_exec_cmd, monkeypatch
):
    monkeypatch.setenv("MLFLOW_SERVER_ENABLE_JOB_EXECUTION", "false")
    monkeypatch.setenv(MLFLOW_FLASK_SERVER_SECRET_KEY.name, "my-secret-key")
    with (
        mock.patch(
            "mlflow.server.auth.bootstrap_admin_user",
            side_effect=MlflowException("no admin password"),
        ),
        pytest.raises(MlflowException, match="no admin password"),
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
            app_name="basic-auth",
        )
    mock_exec_cmd.assert_not_called()


def test_run_server_fails_before_spawning_workers_when_secret_key_missing(
    mock_exec_cmd, monkeypatch
):
    monkeypatch.setenv("MLFLOW_SERVER_ENABLE_JOB_EXECUTION", "false")
    monkeypatch.delenv(MLFLOW_FLASK_SERVER_SECRET_KEY.name, raising=False)
    with (
        mock.patch("mlflow.server.auth.bootstrap_admin_user") as bootstrap,
        pytest.raises(MlflowException, match="MLFLOW_FLASK_SERVER_SECRET_KEY"),
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
            app_name="basic-auth",
        )
    bootstrap.assert_not_called()
    mock_exec_cmd.assert_not_called()


def test_run_server_skips_admin_bootstrap_for_default_app(mock_exec_cmd, monkeypatch):
    monkeypatch.setenv("MLFLOW_SERVER_ENABLE_JOB_EXECUTION", "false")
    with mock.patch("mlflow.server.auth.bootstrap_admin_user") as bootstrap:
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
    bootstrap.assert_not_called()
    assert _MLFLOW_AUTH_ADMIN_BOOTSTRAPPED.name not in mock_exec_cmd.call_args.kwargs["extra_env"]


@pytest.mark.parametrize(
    ("sgi", "warms"), [("gunicorn", True), ("waitress", True), ("uvicorn", False)]
)
def test_flask_app_checks_the_job_executor_backends_on_the_first_request(monkeypatch, sgi, warms):
    monkeypatch.setenv("_MLFLOW_SGI_NAME", sgi)
    monkeypatch.setattr(server, "_executor_registry_warm_started", False)
    warm = mock.Mock()
    monkeypatch.setattr(server, "warm_executor_registry_in_background", warm)

    with server.app.test_client() as client:
        client.get("/health")
        client.get("/health")

    # Once per worker, and only for the servers that use the Flask app directly.
    assert warm.call_count == (1 if warms else 0)


def test_flask_app_serves_the_request_when_the_job_executor_check_cannot_start(monkeypatch):
    monkeypatch.setenv("_MLFLOW_SGI_NAME", "gunicorn")
    monkeypatch.setattr(server, "_executor_registry_warm_started", False)
    monkeypatch.setattr(
        server,
        "warm_executor_registry_in_background",
        mock.Mock(side_effect=RuntimeError("can't start new thread")),
    )

    with server.app.test_client() as client:
        assert client.get("/health").status_code == 200


def _process(*, returncode, running=True):
    """A ``subprocess.Popen`` stand-in for the job runner watcher."""
    proc = mock.Mock()
    proc.wait.return_value = returncode
    proc.poll.return_value = None if running else returncode
    return proc


def _watch_job_runner_test_cases():
    # Built at runtime: Windows has no SIGKILL, and the module must still be collected there.
    cases = [(EXECUTOR_JOB_RUNNER_FATAL_EXIT_CODE, "See the job runner's error above")]
    if hasattr(signal, "SIGKILL"):
        cases.append((-signal.SIGKILL, "was killed with SIGKILL"))
    return cases


@pytest.mark.parametrize(
    ("returncode", "reason"),
    _watch_job_runner_test_cases(),
)
def test_watch_job_runner_stops_the_server_when_the_runner_fails(caplog, returncode, reason):
    server_proc = _process(returncode=None)
    runner_failed = threading.Event()

    server._watch_job_runner(
        _process(returncode=returncode), server_proc, threading.Event(), runner_failed
    )

    assert runner_failed.is_set()
    server_proc.terminate.assert_called_once()
    server_proc.kill.assert_not_called()
    assert reason in caplog.text


@pytest.mark.parametrize(
    ("returncode", "shutting_down", "server_running"),
    [
        # A SIGTERM is also how the runner stops when the server stops.
        (-signal.SIGTERM, False, True),
        (0, False, True),
        # The server is shutting down, or already stopped.
        (EXECUTOR_JOB_RUNNER_FATAL_EXIT_CODE, True, True),
        (EXECUTOR_JOB_RUNNER_FATAL_EXIT_CODE, False, False),
    ],
)
def test_watch_job_runner_leaves_the_server_running(returncode, shutting_down, server_running):
    server_proc = _process(returncode=0, running=server_running)
    shutdown = threading.Event()
    if shutting_down:
        shutdown.set()
    runner_failed = threading.Event()

    server._watch_job_runner(_process(returncode=returncode), server_proc, shutdown, runner_failed)

    assert not runner_failed.is_set()
    server_proc.terminate.assert_not_called()


def test_watch_job_runner_kills_a_server_that_does_not_stop(caplog):
    server_proc = _process(returncode=None)
    server_proc.wait.side_effect = subprocess.TimeoutExpired("server", 30)

    server._watch_job_runner(
        _process(returncode=EXECUTOR_JOB_RUNNER_FATAL_EXIT_CODE),
        server_proc,
        threading.Event(),
        threading.Event(),
    )

    server_proc.terminate.assert_called_once()
    server_proc.wait.assert_called_once_with(timeout=server._SERVER_STOP_TIMEOUT_SECONDS)
    server_proc.kill.assert_called_once()
    assert "did not stop within 30 seconds" in caplog.text


def _python(code):
    return subprocess.Popen([sys.executable, "-c", code])


def _run_server_with_job_runner(
    monkeypatch, server_proc, launch_job_runner, engine="executor", handlers=None
):
    monkeypatch.setenv("MLFLOW_SERVER_ENABLE_JOB_EXECUTION", "true")
    monkeypatch.setenv("MLFLOW_SQL_TRACE_ROLLUPS_ENABLED", "false")
    if engine:
        monkeypatch.setenv("MLFLOW_SERVER_JOB_EXECUTION_ENGINE", engine)
    else:
        monkeypatch.delenv("MLFLOW_SERVER_JOB_EXECUTION_ENGINE", raising=False)
    launcher = "_launch_executor_runner" if engine == "executor" else "_launch_job_runner"
    handlers = {} if handlers is None else handlers
    with (
        mock.patch("mlflow.server._exec_cmd", return_value=server_proc),
        # Keep the handlers _run_server installs out of the test process.
        mock.patch("signal.signal", side_effect=lambda signum, h: handlers.update({signum: h})),
        mock.patch("mlflow.server.jobs.utils._check_requirements"),
        mock.patch("mlflow.server.jobs.executor_registry.validate_executor_config"),
        mock.patch(f"mlflow.server.jobs.utils.{launcher}", side_effect=launch_job_runner),
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


@pytest.mark.skipif(is_windows(), reason="Job execution does not support Windows")
def test_run_server_stops_when_the_job_runner_fails(monkeypatch):
    server_proc = _python("import time; time.sleep(10)")
    runner_code = (
        f"import os, time; time.sleep(0.5); os._exit({EXECUTOR_JOB_RUNNER_FATAL_EXIT_CODE})"
    )

    started = time.monotonic()
    with pytest.raises(ShellCommandException, match="The job runner failed"):
        _run_server_with_job_runner(monkeypatch, server_proc, lambda *args: _python(runner_code))

    assert server_proc.returncode == -signal.SIGTERM
    assert time.monotonic() - started < 8


@pytest.mark.skipif(is_windows(), reason="Job execution does not support Windows")
def test_run_server_kills_a_server_that_does_not_stop(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_SERVER_STOP_TIMEOUT_SECONDS", 0.5)
    ready = tmp_path / "ready"
    server_proc = _python(
        "import pathlib, signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"pathlib.Path({str(ready)!r}).touch(); time.sleep(10)"
    )

    def launch_job_runner(*args):
        _wait_for_file(ready)
        return _python(f"import os; os._exit({EXECUTOR_JOB_RUNNER_FATAL_EXIT_CODE})")

    with pytest.raises(ShellCommandException, match="The job runner failed"):
        _run_server_with_job_runner(monkeypatch, server_proc, launch_job_runner)

    assert server_proc.returncode == -signal.SIGKILL


@pytest.mark.skipif(is_windows(), reason="Job execution does not support Windows")
def test_run_server_reports_a_server_that_fails_before_the_job_runner(monkeypatch):
    server_proc = _python("import sys; sys.exit(3)")
    # The job runner stops itself once the server is gone, and here with the failure code too.
    runner_code = f"import os, time; time.sleep(1); os._exit({EXECUTOR_JOB_RUNNER_FATAL_EXIT_CODE})"

    with pytest.raises(ShellCommandException, match="The server process exited with code 3"):
        _run_server_with_job_runner(monkeypatch, server_proc, lambda *args: _python(runner_code))


@pytest.mark.skipif(is_windows(), reason="Job execution does not support Windows")
def test_run_server_shutdown_is_not_reported_as_a_job_runner_failure(monkeypatch, tmp_path):
    ready = tmp_path / "ready"
    # The server is still up when the job runner exits during the shutdown.
    server_proc = _python(
        "import pathlib, signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"pathlib.Path({str(ready)!r}).touch(); time.sleep(1.5)"
    )
    handlers = {}

    def launch_job_runner(*args):
        _wait_for_file(ready)
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        return _python(f"import os; os._exit({EXECUTOR_JOB_RUNNER_FATAL_EXIT_CODE})")

    _run_server_with_job_runner(monkeypatch, server_proc, launch_job_runner, handlers=handlers)

    assert server_proc.returncode == 0


def test_run_server_does_not_watch_the_default_job_runner(monkeypatch):
    server_proc = mock.Mock(pid=123)
    server_proc.wait.return_value = 0

    with mock.patch.object(server, "_stop_server_if_job_runner_exits") as watch:
        _run_server_with_job_runner(
            monkeypatch, server_proc, lambda *args: mock.Mock(), engine=None
        )

    watch.assert_not_called()


def test_run_server_fails_when_the_server_process_fails(mock_exec_cmd, monkeypatch):
    monkeypatch.setenv("MLFLOW_SERVER_ENABLE_JOB_EXECUTION", "false")
    mock_exec_cmd.return_value.wait.return_value = 3

    with (
        mock.patch("signal.signal"),
        pytest.raises(ShellCommandException, match="exited with code 3"),
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


def _wait_for_file(path, timeout=10):
    deadline = time.monotonic() + timeout
    while not path.exists():
        assert time.monotonic() < deadline, f"{path} was not created"
        time.sleep(0.05)
