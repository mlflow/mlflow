import json
import os
import signal
import subprocess
import sys
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
from mlflow.server.jobs.local_executor import LocalJobExecutor
from mlflow.server.jobs.utils import (
    MLFLOW_SERVER_JOB_ID_ENV_VAR,
    MLFLOW_SERVER_JOB_NAME_ENV_VAR,
    _PreparedJobSetupCommand,
    _PreparedJobSubprocess,
)
from mlflow.utils import PYTHON_VERSION
from mlflow.utils.environment import _PythonEnv

from tests.server.jobs.helpers import wait_for_process_exit

pytestmark = [
    pytest.mark.skipif(os.name == "nt", reason="MLflow job execution is not supported on Windows"),
]


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


def _wait_for_file(path: Path, timeout: float = 5) -> None:
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
        "mlflow.server.jobs.local_executor.LegacyJobResult.load",
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
        timeout=0.2,
    )
    _wait_for_file(pid_path)
    pid = int(pid_path.read_text())

    result = executor.wait_for_job(job_id)

    wait_for_process_exit(pid)
    assert result.status == JobStatus.TIMEOUT
    assert job_id in result.error_message
    assert "timed out after 0.2 seconds" in result.error_message
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
        timeout=0.5 if termination == "timeout" else None,
    )
    _wait_for_file(parent_pid_path)
    _wait_for_file(child_pid_path)
    parent_pid = int(parent_pid_path.read_text())
    child_pid = int(child_pid_path.read_text())

    if termination == "cancel":
        executor.cancel_job(job_id)
    result = executor.wait_for_job(job_id)

    wait_for_process_exit(parent_pid)
    wait_for_process_exit(child_pid)
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
    process = executor._processes[job_id].process
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
