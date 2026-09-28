import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest import mock
from uuid import uuid4

import pytest

from mlflow.entities._job_status import JobStatus
from mlflow.environment_variables import MLFLOW_GATEWAY_URI, MLFLOW_TRACKING_URI, MLFLOW_WORKSPACE
from mlflow.exceptions import MlflowException
from mlflow.server.jobs import job
from mlflow.server.jobs.executor import JobExecutionContext, JobExecutorConfig
from mlflow.server.jobs.local_executor import (
    _STDERR_TAIL_MAX_CHARS,
    LocalJobExecutor,
    _LocalJobProcess,
)
from mlflow.server.jobs.utils import (
    MLFLOW_ORIGINAL_PARENT_PID_ENV_VAR,
    MLFLOW_SERVER_JOB_ID_ENV_VAR,
    MLFLOW_SERVER_JOB_NAME_ENV_VAR,
    _prepare_job_subprocess,
    _PreparedJobSetupCommand,
    _PreparedJobSubprocess,
)
from mlflow.utils import PYTHON_VERSION
from mlflow.utils.environment import _PythonEnv

from tests.server.jobs.helpers import wait_for_process_exit

pytestmark = [
    pytest.mark.skipif(os.name == "nt", reason="MLflow job execution is not supported on Windows"),
]

# CI-safe timing allowances. Readiness/liveness polling must tolerate slow CI
# boxes, and short job timeouts must comfortably exceed subprocess startup so
# that readiness files are written before the deadline fires.
_READINESS_TIMEOUT = 30.0
_SHORT_JOB_TIMEOUT = 2.0


class LocalExecutorTransientError(RuntimeError):
    pass


@job(name="local_executor_add", max_workers=1)
def local_executor_add(x, y):
    return x + y


@job(name="local_executor_runtime_error", max_workers=1)
def local_executor_runtime_error():
    raise RuntimeError("boom")


@job(
    name="local_executor_transient_error",
    max_workers=1,
    transient_error_classes=[LocalExecutorTransientError],
)
def local_executor_transient_error():
    raise LocalExecutorTransientError("retry me")


@job(name="local_executor_sleep", max_workers=1)
def local_executor_sleep(sleep_secs, pid_path):
    Path(pid_path).write_text(str(os.getpid()))
    time.sleep(sleep_secs)


@job(name="local_executor_spawn_child", max_workers=1)
def local_executor_spawn_child(sleep_secs, parent_pid_path, child_pid_path):
    subprocess.Popen([
        sys.executable,
        "-c",
        (
            "import os, sys, time; "
            "from pathlib import Path; "
            "Path(sys.argv[1]).write_text(str(os.getpid())); "
            "time.sleep(float(sys.argv[2]))"
        ),
        child_pid_path,
        str(sleep_secs),
    ])
    Path(parent_pid_path).write_text(str(os.getpid()))
    time.sleep(sleep_secs)


@job(name="local_executor_spawn_child_and_return", max_workers=1)
def local_executor_spawn_child_and_return(sleep_secs, child_pid_path):
    subprocess.Popen([
        sys.executable,
        "-c",
        (
            "import os, sys, time; "
            "from pathlib import Path; "
            "Path(sys.argv[1]).write_text(str(os.getpid())); "
            "time.sleep(float(sys.argv[2]))"
        ),
        child_pid_path,
        str(sleep_secs),
    ])


@job(name="local_executor_context", max_workers=1)
def local_executor_context():
    return {
        "tracking_uri": os.environ.get(MLFLOW_TRACKING_URI.name),
        "gateway_uri": os.environ.get(MLFLOW_GATEWAY_URI.name),
        "workspace": os.environ.get(MLFLOW_WORKSPACE.name),
        "job_id": os.environ.get(MLFLOW_SERVER_JOB_ID_ENV_VAR),
        "job_name": os.environ.get(MLFLOW_SERVER_JOB_NAME_ENV_VAR),
    }


@job(name="local_executor_os_exit", max_workers=1)
def local_executor_os_exit(exit_code):
    os._exit(exit_code)


@job(name="local_executor_stderr_crash", max_workers=1)
def local_executor_stderr_crash():
    sys.stderr.write("boom-on-stderr-marker\n")
    sys.stderr.flush()
    os._exit(3)


@job(name="local_executor_stderr_flood", max_workers=1)
def local_executor_stderr_flood():
    # Emit far more stderr than the retained tail, then a unique final marker so
    # the test can assert only the bounded tail (including that marker) survives.
    for _ in range(2000):
        sys.stderr.write("x" * 100 + "\n")
    sys.stderr.write("final-stderr-marker\n")
    sys.stderr.flush()
    os._exit(5)


@job(name="local_executor_stderr_detached_child", max_workers=1)
def local_executor_stderr_detached_child(sleep_secs, child_pid_path):
    # Spawn a detached grandchild (its own session, so a process-group kill of the
    # job cannot reach it) that inherits the stderr PIPE and outlives this job.
    # The pipe therefore never reaches EOF after the job exits, exercising the
    # stderr reader's interruptible drain loop.
    subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import os, sys, time; "
                "from pathlib import Path; "
                "Path(sys.argv[1]).write_text(str(os.getpid())); "
                "time.sleep(float(sys.argv[2]))"
            ),
            child_pid_path,
            str(sleep_secs),
        ],
        start_new_session=True,
    )
    sys.stderr.write("detached-child-stderr-marker\n")
    sys.stderr.flush()
    os._exit(4)


def local_executor_undecorated():
    return None


def _make_context(
    job_id: str,
    tracking_uri: str = "http://tracking.test",
    gateway_uri: str | None = None,
    workspace: str | None = None,
) -> JobExecutionContext:
    return JobExecutionContext(
        job_id=job_id,
        tracking_uri=tracking_uri,
        gateway_uri=gateway_uri,
        workspace=workspace,
    )


def _wait_for_file(path: Path, timeout: float = _READINESS_TIMEOUT) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    pytest.fail(f"{path} was not created within {timeout}s")


@pytest.fixture
def executor():
    local_executor = LocalJobExecutor(JobExecutorConfig(default_timeout=60.0))
    try:
        yield local_executor
    finally:
        local_executor.stop_executor()


def test_local_executor_submit_and_wait_succeeds(executor):
    job_id = str(uuid4())
    context = _make_context(job_id)

    executor.submit_job(
        job_id=job_id,
        job_name="local_executor_add",
        fn_fullname="tests.server.jobs.test_local_executor.local_executor_add",
        params={"x": 1, "y": 2},
        context=context,
    )
    result = executor.wait_for_job(job_id)

    assert result.status == JobStatus.SUCCEEDED
    assert result.result == "3"
    assert result.error_message is None
    assert executor._processes == {}


def test_local_executor_job_exception_returns_failed(executor):
    job_id = str(uuid4())
    context = _make_context(job_id)

    executor.submit_job(
        job_id=job_id,
        job_name="local_executor_runtime_error",
        fn_fullname="tests.server.jobs.test_local_executor.local_executor_runtime_error",
        params={},
        context=context,
    )
    result = executor.wait_for_job(job_id)

    assert result.status == JobStatus.FAILED
    assert "RuntimeError('boom')" in result.error_message
    assert result.is_transient_error is False


def test_local_executor_declared_transient_error_sets_flag(executor):
    job_id = str(uuid4())
    context = _make_context(job_id)

    executor.submit_job(
        job_id=job_id,
        job_name="local_executor_transient_error",
        fn_fullname="tests.server.jobs.test_local_executor.local_executor_transient_error",
        params={},
        context=context,
    )
    result = executor.wait_for_job(job_id)

    assert result.status == JobStatus.FAILED
    assert "LocalExecutorTransientError('retry me')" in result.error_message
    assert result.is_transient_error is True


def test_local_executor_nonzero_exit_returns_infrastructure_failure(executor):
    job_id = str(uuid4())
    context = _make_context(job_id)

    executor.submit_job(
        job_id=job_id,
        job_name="local_executor_os_exit",
        fn_fullname="tests.server.jobs.test_local_executor.local_executor_os_exit",
        params={"exit_code": 7},
        context=context,
    )
    result = executor.wait_for_job(job_id)

    assert result.status == JobStatus.FAILED
    assert "error code 7" in result.error_message


def test_local_executor_result_protocol_failure_returns_failed(executor):
    job_id = str(uuid4())
    context = _make_context(job_id)

    executor.submit_job(
        job_id=job_id,
        job_name="local_executor_add",
        fn_fullname="tests.server.jobs.test_local_executor.local_executor_add",
        params={"x": 1, "y": 2},
        context=context,
    )
    with mock.patch(
        "mlflow.server.jobs.local_executor._SubprocessJobResult.load",
        side_effect=ValueError("bad result"),
    ):
        result = executor.wait_for_job(job_id)

    assert result.status == JobStatus.FAILED
    assert "Failed to load the local subprocess result" in result.error_message


def test_local_executor_timeout_kills_and_reaps_child(executor, tmp_path: Path):
    job_id = str(uuid4())
    pid_path = tmp_path / "timeout.pid"
    context = _make_context(job_id)

    executor.submit_job(
        job_id=job_id,
        job_name="local_executor_sleep",
        fn_fullname="tests.server.jobs.test_local_executor.local_executor_sleep",
        params={"sleep_secs": 10, "pid_path": str(pid_path)},
        context=context,
        timeout=_SHORT_JOB_TIMEOUT,
    )
    _wait_for_file(pid_path)
    pid = int(pid_path.read_text())

    result = executor.wait_for_job(job_id)

    wait_for_process_exit(pid, timeout=_READINESS_TIMEOUT)
    assert result.status == JobStatus.TIMEOUT
    assert job_id in result.error_message
    assert f"timed out after {_SHORT_JOB_TIMEOUT} seconds" in result.error_message
    assert executor._processes == {}


@pytest.mark.parametrize(
    ("termination", "expected_status"),
    [("cancel", JobStatus.CANCELED), ("timeout", JobStatus.TIMEOUT)],
)
def test_local_executor_termination_kills_descendant_processes(
    executor,
    tmp_path: Path,
    termination: str,
    expected_status: str,
):
    job_id = str(uuid4())
    parent_pid_path = tmp_path / f"{termination}-parent.pid"
    child_pid_path = tmp_path / f"{termination}-child.pid"

    executor.submit_job(
        job_id=job_id,
        job_name="local_executor_spawn_child",
        fn_fullname="tests.server.jobs.test_local_executor.local_executor_spawn_child",
        params={
            "sleep_secs": 10,
            "parent_pid_path": str(parent_pid_path),
            "child_pid_path": str(child_pid_path),
        },
        context=_make_context(job_id),
        timeout=_SHORT_JOB_TIMEOUT if termination == "timeout" else None,
    )
    _wait_for_file(parent_pid_path)
    _wait_for_file(child_pid_path)
    parent_pid = int(parent_pid_path.read_text())
    child_pid = int(child_pid_path.read_text())

    if termination == "cancel":
        executor.cancel_job(job_id)
    result = executor.wait_for_job(job_id)

    wait_for_process_exit(parent_pid, timeout=_READINESS_TIMEOUT)
    wait_for_process_exit(child_pid, timeout=_READINESS_TIMEOUT)
    assert result.status == expected_status


def test_local_executor_cancel_kills_descendant_after_job_process_exits(
    executor,
    tmp_path: Path,
):
    job_id = str(uuid4())
    child_pid_path = tmp_path / "exited-parent-child.pid"

    executor.submit_job(
        job_id=job_id,
        job_name="local_executor_spawn_child_and_return",
        fn_fullname="tests.server.jobs.test_local_executor.local_executor_spawn_child_and_return",
        params={"sleep_secs": 10, "child_pid_path": str(child_pid_path)},
        context=_make_context(job_id),
    )
    _wait_for_file(child_pid_path)
    child_pid = int(child_pid_path.read_text())
    result_path = Path(executor._processes[job_id].result_path)
    _wait_for_file(result_path)
    time.sleep(0.1)

    executor.cancel_job(job_id)
    result = executor.wait_for_job(job_id)

    wait_for_process_exit(child_pid)
    assert result.status == JobStatus.CANCELED


def test_local_executor_cancel_loses_race_to_completed_job(
    monkeypatch,
    executor,
    tmp_path: Path,
):
    job_id = str(uuid4())
    child_pid_path = tmp_path / "completed-job-child.pid"
    result_holder = {}

    executor.submit_job(
        job_id=job_id,
        job_name="local_executor_spawn_child_and_return",
        fn_fullname="tests.server.jobs.test_local_executor.local_executor_spawn_child_and_return",
        params={"sleep_secs": 10, "child_pid_path": str(child_pid_path)},
        context=_make_context(job_id),
    )
    _wait_for_file(child_pid_path)
    child_pid = int(child_pid_path.read_text())
    process = executor._processes[job_id].active_process
    original_wait = process.wait
    process_reaped = threading.Event()
    release_waiter = threading.Event()

    def _wait_then_pause(*args, **kwargs):
        returncode = original_wait(*args, **kwargs)
        process_reaped.set()
        release_waiter.wait(timeout=5)
        return returncode

    monkeypatch.setattr(process, "wait", _wait_then_pause)
    wait_thread = threading.Thread(
        target=lambda: result_holder.setdefault("result", executor.wait_for_job(job_id)),
        name="local-executor-completion-race-waiter",
    )
    wait_thread.start()
    assert process_reaped.wait(timeout=5)
    assert process.returncode == 0
    executor.cancel_job(job_id)
    release_waiter.set()
    wait_thread.join(timeout=5)
    try:
        os.kill(child_pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    wait_for_process_exit(child_pid)

    assert not wait_thread.is_alive()
    assert result_holder["result"].status == JobStatus.SUCCEEDED


def test_local_executor_cancel_before_wait_returns_canceled(executor, tmp_path: Path):
    job_id = str(uuid4())
    pid_path = tmp_path / "cancel-before.pid"
    context = _make_context(job_id)

    executor.submit_job(
        job_id=job_id,
        job_name="local_executor_sleep",
        fn_fullname="tests.server.jobs.test_local_executor.local_executor_sleep",
        params={"sleep_secs": 10, "pid_path": str(pid_path)},
        context=context,
    )
    _wait_for_file(pid_path)
    pid = int(pid_path.read_text())

    executor.cancel_job(job_id)
    result = executor.wait_for_job(job_id)

    wait_for_process_exit(pid)
    assert result.status == JobStatus.CANCELED
    assert executor._processes == {}


def test_local_executor_cancel_while_waiting_unblocks_waiter(executor, tmp_path: Path):
    job_id = str(uuid4())
    pid_path = tmp_path / "cancel-during-wait.pid"
    context = _make_context(job_id)
    result_holder = {}

    executor.submit_job(
        job_id=job_id,
        job_name="local_executor_sleep",
        fn_fullname="tests.server.jobs.test_local_executor.local_executor_sleep",
        params={"sleep_secs": 10, "pid_path": str(pid_path)},
        context=context,
    )
    _wait_for_file(pid_path)
    pid = int(pid_path.read_text())

    wait_thread = threading.Thread(
        target=lambda: result_holder.setdefault("result", executor.wait_for_job(job_id)),
        name="local-executor-waiter",
    )
    wait_thread.start()
    time.sleep(0.1)
    executor.cancel_job(job_id)
    wait_thread.join(timeout=5)

    wait_for_process_exit(pid)
    assert result_holder["result"].status == JobStatus.CANCELED
    assert executor._processes == {}


def test_local_executor_rejects_duplicate_submission(executor, tmp_path: Path):
    job_id = str(uuid4())
    pid_path = tmp_path / "duplicate.pid"
    context = _make_context(job_id)

    executor.submit_job(
        job_id=job_id,
        job_name="local_executor_sleep",
        fn_fullname="tests.server.jobs.test_local_executor.local_executor_sleep",
        params={"sleep_secs": 10, "pid_path": str(pid_path)},
        context=context,
    )

    with pytest.raises(MlflowException, match="already being managed"):
        executor.submit_job(
            job_id=job_id,
            job_name="local_executor_sleep",
            fn_fullname="tests.server.jobs.test_local_executor.local_executor_sleep",
            params={"sleep_secs": 10, "pid_path": str(pid_path)},
            context=context,
        )

    executor.cancel_job(job_id)
    executor.wait_for_job(job_id)


def test_local_executor_wait_unknown_job_raises(executor):
    with pytest.raises(MlflowException, match="Unknown job ID"):
        executor.wait_for_job("missing-job")


def test_local_executor_cancel_unknown_job_is_noop(executor):
    executor.cancel_job("missing-job")


def test_local_executor_rejects_undecorated_function_with_public_decorator_name(executor):
    job_id = str(uuid4())

    with pytest.raises(MlflowException, match=r"mlflow\.server\.jobs\.job'"):
        executor.submit_job(
            job_id=job_id,
            job_name="local_executor_undecorated",
            fn_fullname="tests.server.jobs.test_local_executor.local_executor_undecorated",
            params={},
            context=_make_context(job_id),
        )

    assert executor._processes == {}


def test_local_executor_cancel_during_environment_setup(
    monkeypatch,
    executor,
    tmp_path: Path,
):
    job_id = str(uuid4())
    setup_pid_path = tmp_path / "setup.pid"
    setup_child_pid_path = tmp_path / "setup-child.pid"
    final_process_marker = tmp_path / "final-process-started"
    result_path = tmp_path / "result.json"
    setup_child_script = (
        "import os, sys, time; "
        "from pathlib import Path; "
        "Path(sys.argv[1]).write_text(str(os.getpid())); "
        "time.sleep(30)"
    )
    setup_script = (
        "import os, subprocess, sys, time; "
        "from pathlib import Path; "
        f"subprocess.Popen([sys.executable, '-c', {setup_child_script!r}, sys.argv[2]]); "
        "Path(sys.argv[1]).write_text(str(os.getpid())); "
        "time.sleep(30)"
    )

    def _fake_prepare_job_subprocess(**kwargs):
        return _PreparedJobSubprocess(
            command=[
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(final_process_marker)!r}).touch()",
            ],
            env=os.environ.copy(),
            result_path=str(result_path),
            setup_commands=(
                _PreparedJobSetupCommand(
                    command=[
                        sys.executable,
                        "-c",
                        setup_script,
                        str(setup_pid_path),
                        str(setup_child_pid_path),
                    ]
                ),
            ),
        )

    monkeypatch.setattr(
        "mlflow.server.jobs.local_executor._prepare_job_subprocess",
        _fake_prepare_job_subprocess,
    )
    submit_errors = []

    def _submit_job():
        try:
            executor.submit_job(
                job_id=job_id,
                job_name="local_executor_add",
                fn_fullname="tests.server.jobs.test_local_executor.local_executor_add",
                params={"x": 1, "y": 2},
                context=_make_context(job_id),
            )
        except Exception as exc:
            submit_errors.append(exc)

    submit_thread = threading.Thread(
        target=_submit_job,
        name="local-executor-submit-during-setup",
    )
    submit_thread.start()
    _wait_for_file(setup_pid_path)
    _wait_for_file(setup_child_pid_path)
    setup_pid = int(setup_pid_path.read_text())
    setup_child_pid = int(setup_child_pid_path.read_text())

    executor.cancel_job(job_id)
    submit_thread.join(timeout=5)
    result = executor.wait_for_job(job_id)

    wait_for_process_exit(setup_pid)
    wait_for_process_exit(setup_child_pid)
    assert not submit_thread.is_alive()
    assert submit_errors == []
    assert result.status == JobStatus.CANCELED
    assert not final_process_marker.exists()


def test_local_executor_propagates_context_env(executor):
    job_id = str(uuid4())
    context = _make_context(
        job_id,
        tracking_uri="http://tracking.example",
        gateway_uri="http://gateway.example",
        workspace="workspace-a",
    )

    executor.submit_job(
        job_id=job_id,
        job_name="local_executor_context",
        fn_fullname="tests.server.jobs.test_local_executor.local_executor_context",
        params={},
        context=context,
    )
    result = executor.wait_for_job(job_id)
    payload = json.loads(result.result)

    assert result.status == JobStatus.SUCCEEDED
    assert payload == {
        "tracking_uri": "http://tracking.example",
        "gateway_uri": "http://gateway.example",
        "workspace": "workspace-a",
        "job_id": job_id,
        "job_name": "local_executor_context",
    }


def test_local_executor_context_overrides_inherited_env(monkeypatch, executor):
    monkeypatch.setenv(MLFLOW_GATEWAY_URI.name, "http://inherited-gateway.example")
    monkeypatch.setenv(MLFLOW_WORKSPACE.name, "inherited-workspace")
    job_id = str(uuid4())
    context = _make_context(job_id, tracking_uri="http://tracking.example")

    executor.submit_job(
        job_id=job_id,
        job_name="local_executor_context",
        fn_fullname="tests.server.jobs.test_local_executor.local_executor_context",
        params={},
        context=context,
    )
    result = executor.wait_for_job(job_id)
    payload = json.loads(result.result)

    assert result.status == JobStatus.SUCCEEDED
    assert payload["tracking_uri"] == "http://tracking.example"
    assert payload["gateway_uri"] == "http://tracking.example"
    assert payload["workspace"] is None


def test_local_executor_normalizes_python_env(monkeypatch, executor):
    job_id = str(uuid4())
    context = _make_context(job_id)
    captured = {}
    mock_process = mock.Mock()
    mock_process.poll.return_value = 0

    def _fake_prepare_job_subprocess(**kwargs):
        captured["python_env"] = kwargs["python_env"]
        return _PreparedJobSubprocess(
            command=["python", "-c", "print('ok')"],
            env={},
            result_path="/tmp/result.json",
        )

    monkeypatch.setattr(
        "mlflow.server.jobs.local_executor._prepare_job_subprocess",
        _fake_prepare_job_subprocess,
    )
    monkeypatch.setattr(
        "mlflow.server.jobs.local_executor.subprocess.Popen",
        lambda *a, **k: mock_process,
    )

    executor.submit_job(
        job_id=job_id,
        job_name="local_executor_add",
        fn_fullname="tests.server.jobs.test_local_executor.local_executor_add",
        params={"x": 1, "y": 2},
        context=context,
        python_env=_PythonEnv(
            python="3.11.9",
            build_dependencies=["pip==24.0"],
            dependencies=["pytest<9"],
        ),
    )

    normalized = captured["python_env"]
    assert normalized.python == PYTHON_VERSION
    assert normalized.build_dependencies == ["pip==24.0"]
    assert normalized.dependencies == ["pytest<9"]


def test_local_executor_recover_jobs_kills_known_processes_before_requeue(executor, tmp_path: Path):
    job_id = str(uuid4())
    pid_path = tmp_path / "recover.pid"
    context = _make_context(job_id)

    executor.submit_job(
        job_id=job_id,
        job_name="local_executor_sleep",
        fn_fullname="tests.server.jobs.test_local_executor.local_executor_sleep",
        params={"sleep_secs": 10, "pid_path": str(pid_path)},
        context=context,
    )
    _wait_for_file(pid_path)
    pid = int(pid_path.read_text())

    results = executor.recover_jobs([job_id, "missing-job"])

    wait_for_process_exit(pid)
    assert [(result.job_id, result.action) for result in results] == [
        (job_id, "requeue"),
        ("missing-job", "requeue"),
    ]
    assert executor._processes == {}


def test_local_executor_stop_executor_kills_children_and_is_idempotent(executor, tmp_path: Path):
    job_id = str(uuid4())
    pid_path = tmp_path / "stop.pid"
    context = _make_context(job_id)

    executor.submit_job(
        job_id=job_id,
        job_name="local_executor_sleep",
        fn_fullname="tests.server.jobs.test_local_executor.local_executor_sleep",
        params={"sleep_secs": 10, "pid_path": str(pid_path)},
        context=context,
    )
    _wait_for_file(pid_path)
    pid = int(pid_path.read_text())

    executor.stop_executor()
    executor.stop_executor()

    wait_for_process_exit(pid)
    assert executor._processes == {}


def test_local_executor_submit_requires_matching_context_job_id(executor):
    with pytest.raises(MlflowException, match="Job ID mismatch"):
        executor.submit_job(
            job_id="job-1",
            job_name="local_executor_add",
            fn_fullname="tests.server.jobs.test_local_executor.local_executor_add",
            params={"x": 1, "y": 2},
            context=_make_context("job-2"),
        )


def test_local_executor_check_requirements_only_rejects_windows(executor):
    executor.check_requirements()

    with mock.patch("mlflow.server.jobs.local_executor.os.name", "nt"):
        with pytest.raises(MlflowException, match="does not support Windows"):
            executor.check_requirements()


def test_local_executor_remote_execution_defaults_false(executor):
    assert executor.remote_execution is False


def test_prepare_job_subprocess_installs_build_dependencies_first(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("mlflow.server.jobs.utils.shutil.which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(
        "mlflow.utils.virtualenv._get_mlflow_virtualenv_root",
        lambda: str(tmp_path / "venvs"),
    )

    with_build = _prepare_job_subprocess(
        function_fullname="tests.server.jobs.test_local_executor.local_executor_add",
        params={"x": 1, "y": 2},
        python_env=_PythonEnv(
            python=PYTHON_VERSION,
            build_dependencies=["pip==24.0", "setuptools==69.0"],
            dependencies=["pytest<9"],
        ),
        transient_error_classes=None,
        tmpdir=str(tmp_path),
        job_id="job-build",
        job_name="local_executor_add",
        workspace=None,
    )

    assert len(with_build.setup_commands) == 1
    command = with_build.setup_commands[0].command
    build_req_file = Path(command[command.index("--build-requirements-file") + 1])
    req_file = Path(command[command.index("--requirements-file") + 1])
    assert build_req_file.read_text().splitlines() == ["pip==24.0", "setuptools==69.0"]
    assert req_file.read_text().splitlines() == ["pytest<9"]

    without_build = _prepare_job_subprocess(
        function_fullname="tests.server.jobs.test_local_executor.local_executor_add",
        params={"x": 1, "y": 2},
        python_env=_PythonEnv(
            python=PYTHON_VERSION,
            build_dependencies=[],
            dependencies=["pytest<9"],
        ),
        transient_error_classes=None,
        tmpdir=str(tmp_path),
        job_id="job-nobuild",
        job_name="local_executor_add",
        workspace=None,
    )

    assert "--build-requirements-file" not in without_build.setup_commands[0].command


def test_setup_job_environment_installs_build_requirements_before_requirements(
    tmp_path: Path, monkeypatch
):
    from mlflow.server.jobs import _job_env_setup

    env_dir = tmp_path / "venvs" / "env"
    requirements_file = tmp_path / "requirements.txt"
    requirements_file.write_text("pytest<9")
    build_requirements_file = tmp_path / "build-requirements.txt"
    build_requirements_file.write_text("pip==24.0")

    calls = []
    monkeypatch.setattr(
        _job_env_setup, "_get_uv_env_creation_command", lambda *a, **k: ["uv", "venv"]
    )
    monkeypatch.setattr(
        _job_env_setup, "_exec_cmd", lambda command, **kwargs: calls.append(command)
    )

    _job_env_setup._setup_job_environment(
        env_dir=env_dir,
        python_version=PYTHON_VERSION,
        requirements_file=requirements_file,
        build_requirements_file=build_requirements_file,
    )

    # Order: create the env, install build requirements, then install requirements.
    assert len(calls) == 3
    assert calls[0] == ["uv", "venv"]
    assert "build-requirements.txt" in " ".join(map(str, calls[1]))
    # The final call installs the regular requirements, not the build requirements
    # ("build-requirements.txt" contains "requirements.txt" as a substring, so the
    # negative assertion is what makes this check meaningful).
    calls_2_str = " ".join(map(str, calls[2]))
    assert "requirements.txt" in calls_2_str
    assert "build-requirements.txt" not in calls_2_str


def test_setup_job_environment_skips_build_requirements_when_absent(tmp_path: Path, monkeypatch):
    from mlflow.server.jobs import _job_env_setup

    env_dir = tmp_path / "venvs" / "env"
    requirements_file = tmp_path / "requirements.txt"
    requirements_file.write_text("pytest<9")

    calls = []
    monkeypatch.setattr(
        _job_env_setup, "_get_uv_env_creation_command", lambda *a, **k: ["uv", "venv"]
    )
    monkeypatch.setattr(
        _job_env_setup, "_exec_cmd", lambda command, **kwargs: calls.append(command)
    )

    _job_env_setup._setup_job_environment(
        env_dir=env_dir,
        python_version=PYTHON_VERSION,
        requirements_file=requirements_file,
    )

    # Only env creation + a single requirements install.
    assert len(calls) == 2
    assert calls[0] == ["uv", "venv"]
    assert "requirements.txt" in " ".join(map(str, calls[1]))


def test_local_executor_nonzero_exit_surfaces_stderr_tail(executor):
    job_id = str(uuid4())
    context = _make_context(job_id)

    executor.submit_job(
        job_id=job_id,
        job_name="local_executor_stderr_crash",
        fn_fullname="tests.server.jobs.test_local_executor.local_executor_stderr_crash",
        params={},
        context=context,
    )
    result = executor.wait_for_job(job_id)

    assert result.status == JobStatus.FAILED
    assert "error code 3" in result.error_message
    assert "boom-on-stderr-marker" in result.error_message
    assert executor._processes == {}


def test_local_executor_setup_timeout_returns_timeout(monkeypatch, executor, tmp_path: Path):
    job_id = str(uuid4())
    setup_pid_path = tmp_path / "setup-timeout.pid"
    setup_child_pid_path = tmp_path / "setup-timeout-child.pid"
    final_process_marker = tmp_path / "final-process-started"
    result_path = tmp_path / "result.json"
    setup_child_script = (
        "import os, sys, time; "
        "from pathlib import Path; "
        "Path(sys.argv[1]).write_text(str(os.getpid())); "
        "time.sleep(30)"
    )
    setup_script = (
        "import os, subprocess, sys, time; "
        "from pathlib import Path; "
        f"subprocess.Popen([sys.executable, '-c', {setup_child_script!r}, sys.argv[2]]); "
        "Path(sys.argv[1]).write_text(str(os.getpid())); "
        "time.sleep(30)"
    )

    def _fake_prepare_job_subprocess(**kwargs):
        return _PreparedJobSubprocess(
            command=[
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(final_process_marker)!r}).touch()",
            ],
            env=os.environ.copy(),
            result_path=str(result_path),
            setup_commands=(
                _PreparedJobSetupCommand(
                    command=[
                        sys.executable,
                        "-c",
                        setup_script,
                        str(setup_pid_path),
                        str(setup_child_pid_path),
                    ]
                ),
            ),
        )

    monkeypatch.setattr(
        "mlflow.server.jobs.local_executor._prepare_job_subprocess",
        _fake_prepare_job_subprocess,
    )

    # submit_job blocks during setup until the hard deadline elapses, proving the
    # timeout covers environment setup and not just job execution.
    executor.submit_job(
        job_id=job_id,
        job_name="local_executor_add",
        fn_fullname="tests.server.jobs.test_local_executor.local_executor_add",
        params={"x": 1, "y": 2},
        context=_make_context(job_id),
        timeout=_SHORT_JOB_TIMEOUT,
    )
    result = executor.wait_for_job(job_id)

    _wait_for_file(setup_pid_path)
    _wait_for_file(setup_child_pid_path)
    setup_pid = int(setup_pid_path.read_text())
    setup_child_pid = int(setup_child_pid_path.read_text())

    wait_for_process_exit(setup_pid, timeout=_READINESS_TIMEOUT)
    wait_for_process_exit(setup_child_pid, timeout=_READINESS_TIMEOUT)
    assert result.status == JobStatus.TIMEOUT
    assert "during environment setup" in result.error_message
    assert not final_process_marker.exists()
    assert executor._processes == {}


def test_local_executor_orphan_watcher_kills_group_on_parent_crash(tmp_path: Path):
    # Simulates a server crash: a job subprocess (a process-group leader started
    # with start_new_session) is reparented, so its orphan watcher must kill the
    # whole process group, taking any spawned descendant with it.
    repo_root = Path(__file__).resolve().parents[3]
    server_script = tmp_path / "fake_server.py"
    server_script.write_text(
        "import sys\n"
        "import time\n"
        "from pathlib import Path\n"
        "\n"
        "from mlflow.server.jobs.executor import JobExecutionContext, JobExecutorConfig\n"
        "from mlflow.server.jobs.local_executor import LocalJobExecutor\n"
        "\n"
        "job_id, parent_pid_path, child_pid_path, ready_path = sys.argv[1:5]\n"
        "executor = LocalJobExecutor(JobExecutorConfig(default_timeout=120.0))\n"
        "executor.submit_job(\n"
        "    job_id=job_id,\n"
        "    job_name='local_executor_spawn_child',\n"
        "    fn_fullname='tests.server.jobs.test_local_executor.local_executor_spawn_child',\n"
        "    params={\n"
        "        'sleep_secs': 120,\n"
        "        'parent_pid_path': parent_pid_path,\n"
        "        'child_pid_path': child_pid_path,\n"
        "    },\n"
        "    context=JobExecutionContext(job_id=job_id, tracking_uri='http://tracking.test'),\n"
        ")\n"
        "Path(ready_path).write_text('ready')\n"
        "time.sleep(120)\n"
    )

    job_id = str(uuid4())
    parent_pid_path = tmp_path / "orphan-parent.pid"
    child_pid_path = tmp_path / "orphan-child.pid"
    ready_path = tmp_path / "server-ready"

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(repo_root), env.get("PYTHONPATH", "")]).rstrip(
        os.pathsep
    )

    server = subprocess.Popen(
        [
            sys.executable,
            str(server_script),
            job_id,
            str(parent_pid_path),
            str(child_pid_path),
            str(ready_path),
        ],
        env=env,
    )
    parent_pid = None
    child_pid = None
    try:
        _wait_for_file(ready_path)
        _wait_for_file(parent_pid_path)
        _wait_for_file(child_pid_path)
        parent_pid = int(parent_pid_path.read_text())
        child_pid = int(child_pid_path.read_text())

        server.kill()
        server.wait(timeout=_READINESS_TIMEOUT)

        # The orphan watcher polls parentage, so allow time for detection + killpg.
        wait_for_process_exit(parent_pid, timeout=_READINESS_TIMEOUT)
        wait_for_process_exit(child_pid, timeout=_READINESS_TIMEOUT)
    finally:
        server.kill()
        for pid in (parent_pid, child_pid):
            if pid is not None:
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass


def _make_record(**overrides) -> _LocalJobProcess:
    fields = {
        "active_process": None,
        "temporary_directory": tempfile.TemporaryDirectory(),
        "result_path": "",
        "function_fullname": "fn",
        "timeout": 60.0,
        "deadline": time.monotonic() + 60.0,
    }
    fields.update(overrides)
    return _LocalJobProcess(**fields)


def test_local_executor_cancel_latches_during_setup(executor):
    # A cancel that arrives while a setup subprocess is the active process (the
    # job subprocess has not been launched yet) must latch. Even when the kill is
    # reported as not delivered (e.g. the setup process already exited),
    # cancel_requested must stay set so submit_job does not go on to launch the
    # job the caller asked to cancel.
    job_id = str(uuid4())
    setup_process = mock.Mock()
    setup_process.returncode = 0  # _kill_process reports "not delivered" for this
    record = _make_record(active_process=setup_process, job_process_launched=False)
    executor._processes[job_id] = record
    try:
        executor.cancel_job(job_id)
        assert record.cancel_requested is True
    finally:
        record.temporary_directory.cleanup()
        executor._processes.pop(job_id, None)


def test_local_executor_cancel_rolls_back_after_job_process_completed(executor):
    # Once the job subprocess is running, a cancel that loses the race to a job
    # that already completed must still roll back so the completed result wins.
    job_id = str(uuid4())
    job_process = mock.Mock()
    job_process.returncode = 0
    record = _make_record(active_process=job_process, job_process_launched=True)
    executor._processes[job_id] = record
    try:
        executor.cancel_job(job_id)
        assert record.cancel_requested is False
    finally:
        record.temporary_directory.cleanup()
        executor._processes.pop(job_id, None)


def test_local_executor_deadline_elapsed_before_launch_returns_timeout(
    monkeypatch, executor, tmp_path: Path
):
    # The hard deadline is checked under the lock immediately before the job
    # subprocess is launched, so a job whose budget is consumed during setup is
    # reported as a timeout and its subprocess is never started.
    job_id = str(uuid4())
    final_process_marker = tmp_path / "final-process-started"
    result_path = tmp_path / "result.json"

    def _fake_prepare_job_subprocess(**kwargs):
        # Consume the whole timeout budget before the job subprocess is launched.
        time.sleep(_SHORT_JOB_TIMEOUT + 0.5)
        return _PreparedJobSubprocess(
            command=[
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(final_process_marker)!r}).touch()",
            ],
            env=os.environ.copy(),
            result_path=str(result_path),
        )

    monkeypatch.setattr(
        "mlflow.server.jobs.local_executor._prepare_job_subprocess",
        _fake_prepare_job_subprocess,
    )

    executor.submit_job(
        job_id=job_id,
        job_name="local_executor_add",
        fn_fullname="tests.server.jobs.test_local_executor.local_executor_add",
        params={"x": 1, "y": 2},
        context=_make_context(job_id),
        timeout=_SHORT_JOB_TIMEOUT,
    )
    result = executor.wait_for_job(job_id)

    assert result.status == JobStatus.TIMEOUT
    assert "during environment setup" in result.error_message
    assert not final_process_marker.exists()
    assert executor._processes == {}


def test_local_executor_bounds_captured_stderr_tail(executor):
    # A job that floods stderr before crashing must surface only a bounded tail so
    # the server never holds the full stderr in memory.
    job_id = str(uuid4())

    executor.submit_job(
        job_id=job_id,
        job_name="local_executor_stderr_flood",
        fn_fullname="tests.server.jobs.test_local_executor.local_executor_stderr_flood",
        params={},
        context=_make_context(job_id),
    )
    result = executor.wait_for_job(job_id)

    assert result.status == JobStatus.FAILED
    assert "error code 5" in result.error_message
    # The retained tail is truncated and still ends with the final marker line.
    assert "...(truncated)..." in result.error_message
    assert "final-stderr-marker" in result.error_message
    _, _, stderr_section = result.error_message.partition("Subprocess stderr (tail):\n")
    assert stderr_section
    assert len(stderr_section) <= _STDERR_TAIL_MAX_CHARS + len("...(truncated)...\n")
    assert executor._processes == {}


def test_local_executor_stderr_reader_does_not_block_on_detached_descendant(
    executor, tmp_path: Path
):
    # A job may spawn a detached descendant that inherits the stderr PIPE and keeps
    # its write end open long after the job exits. wait_for_job must still return
    # promptly (draining the reader is bounded, not gated on that descendant's
    # lifetime) while still surfacing the stderr the job itself emitted.
    job_id = str(uuid4())
    child_pid_path = tmp_path / "detached-descendant.pid"
    descendant_sleep = 30.0

    executor.submit_job(
        job_id=job_id,
        job_name="local_executor_stderr_detached_child",
        fn_fullname=("tests.server.jobs.test_local_executor.local_executor_stderr_detached_child"),
        params={"sleep_secs": descendant_sleep, "child_pid_path": str(child_pid_path)},
        context=_make_context(job_id),
    )
    start = time.monotonic()
    result = executor.wait_for_job(job_id)
    elapsed = time.monotonic() - start

    # Clean up the detached descendant regardless of assertion outcomes.
    _wait_for_file(child_pid_path)
    child_pid = int(child_pid_path.read_text())
    try:
        os.kill(child_pid, signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass
    wait_for_process_exit(child_pid, timeout=_READINESS_TIMEOUT)

    assert result.status == JobStatus.FAILED
    assert "error code 4" in result.error_message
    assert "detached-child-stderr-marker" in result.error_message
    # Must not stall for the descendant's whole sleep; the drain loop stops within
    # a poll interval once the job process is reaped.
    assert elapsed < descendant_sleep / 2
    assert executor._processes == {}


def test_prepare_job_subprocess_setup_carries_parent_pid_for_orphan_watch(
    tmp_path: Path, monkeypatch
):
    # The setup command must record the server pid so the setup subprocess can
    # detect being orphaned (and release its environment lock) if the server dies.
    monkeypatch.setattr("mlflow.server.jobs.utils.shutil.which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(
        "mlflow.utils.virtualenv._get_mlflow_virtualenv_root",
        lambda: str(tmp_path / "venvs"),
    )

    prepared = _prepare_job_subprocess(
        function_fullname="tests.server.jobs.test_local_executor.local_executor_add",
        params={"x": 1, "y": 2},
        python_env=_PythonEnv(
            python=PYTHON_VERSION,
            build_dependencies=[],
            dependencies=["pytest<9"],
        ),
        transient_error_classes=None,
        tmpdir=str(tmp_path),
        job_id="job-orphan",
        job_name="local_executor_add",
        workspace=None,
    )

    assert len(prepared.setup_commands) == 1
    assert prepared.setup_commands[0].extra_env == {
        MLFLOW_ORIGINAL_PARENT_PID_ENV_VAR: str(os.getpid())
    }


def _run_env_setup_main(monkeypatch, tmp_path: Path, watcher_started: threading.Event) -> None:
    from mlflow.server.jobs import _job_env_setup

    (tmp_path / "req.txt").write_text("")
    monkeypatch.setattr(
        _job_env_setup, "_exit_when_orphaned", lambda *a, **k: watcher_started.set()
    )
    monkeypatch.setattr(_job_env_setup, "_setup_job_environment", lambda *a, **k: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "_job_env_setup",
            "--env-dir",
            str(tmp_path / "env"),
            "--python-version",
            PYTHON_VERSION,
            "--requirements-file",
            str(tmp_path / "req.txt"),
        ],
    )
    _job_env_setup.main()


def test_job_env_setup_starts_orphan_watcher_when_parent_pid_set(monkeypatch, tmp_path: Path):
    watcher_started = threading.Event()
    monkeypatch.setenv(MLFLOW_ORIGINAL_PARENT_PID_ENV_VAR, str(os.getpid()))
    _run_env_setup_main(monkeypatch, tmp_path, watcher_started)
    assert watcher_started.wait(timeout=5)


def test_job_env_setup_skips_orphan_watcher_without_parent_pid(monkeypatch, tmp_path: Path):
    watcher_started = threading.Event()
    monkeypatch.delenv(MLFLOW_ORIGINAL_PARENT_PID_ENV_VAR, raising=False)
    _run_env_setup_main(monkeypatch, tmp_path, watcher_started)
    time.sleep(0.2)
    assert not watcher_started.is_set()
