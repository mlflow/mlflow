import json
import os
from pathlib import Path

import pytest
import requests

from mlflow.entities._job_status import JobStatus
from mlflow.exceptions import MlflowException
from mlflow.server.jobs import docker_executor as de
from mlflow.server.jobs import job
from mlflow.server.jobs.executor import JobExecutionContext, JobExecutorConfig
from mlflow.server.jobs.utils import (
    MLFLOW_SERVER_JOB_FUNCTION_FULLNAME_ENV_VAR,
    MLFLOW_SERVER_JOB_ID_ENV_VAR,
    MLFLOW_SERVER_JOB_PARAMS_ENV_VAR,
    _SubprocessJobResult,
)

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="The docker job executor requires a POSIX host"
)


@job(name="fake_docker_job", max_workers=1)
def fake_docker_job(x):
    return x


def _fake_planner(params, input_dir, container_input_dir):
    (Path(input_dir) / "inputs.json").write_text(json.dumps(params))
    return de.DockerJobPlan(
        fn_fullname="some.module.compute",
        params={"inputs_path": f"{container_input_dir}/inputs.json"},
        finalize=lambda value: {"finalized": value},
    )


class _FakeContainer:
    def __init__(self, client, kwargs):
        self.id = f"container-{len(client.started)}"
        self.short_id = self.id
        self.kwargs = kwargs
        self.killed = False
        self.removed = False
        self.on_wait = client.on_wait

    def _output_dir(self):
        for host, spec in self.kwargs["volumes"].items():
            if spec["bind"] == de._CONTAINER_OUTPUT_DIR:
                return Path(host)

    def wait(self, timeout=None):
        return self.on_wait(self)

    def logs(self, tail=None):
        self.logs_tail = tail
        return b"container log line"

    def kill(self):
        self.killed = True

    def remove(self, force=False):
        self.removed = True


class _FakeContainers:
    def __init__(self, client):
        self._client = client

    def run(self, image, **kwargs):
        container = _FakeContainer(self._client, kwargs)
        container.image = image
        if self._client.start_error is not None:
            # Like docker-py: the container is created, then starting it fails.
            self._client.leftovers.append(container)
            raise self._client.start_error
        self._client.started.append(container)
        return container

    def get(self, container_id):
        if self._client.get_error is not None:
            raise self._client.get_error
        return next(c for c in self._client.started if c.id == container_id)

    def list(self, all=False, filters=None):
        return list(self._client.leftovers)


class _FakeClient:
    def __init__(self, on_wait=None):
        self.started = []
        self.leftovers = []
        self.on_wait = on_wait or _succeed_with({"score": 1})
        self.start_error = None
        self.get_error = None
        self.containers = _FakeContainers(self)

    def ping(self):
        return True


def _succeed_with(value):
    def on_wait(container):
        _SubprocessJobResult(succeeded=True, result=json.dumps(value)).dump(
            str(container._output_dir() / de._RESULT_FILE)
        )
        return {"StatusCode": 0}

    return on_wait


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setitem(de._JOB_PLANNERS, "fake_docker_job", f"{__name__}._fake_planner")
    return _FakeClient()


@pytest.fixture
def executor(client):
    ex = de.DockerJobExecutor(JobExecutorConfig(default_timeout=30.0))
    ex._client = client
    return ex


def _submit(executor, job_id="job-1", job_name="fake_docker_job", **kwargs):
    executor.submit_job(
        job_id=job_id,
        job_name=job_name,
        fn_fullname=f"{__name__}.fake_docker_job",
        params={"x": 1},
        context=JobExecutionContext(job_id=job_id, tracking_uri="sqlite:///unused.db"),
        **kwargs,
    )


_CUSTOM_SCORER = {
    "name": "custom",
    "call_source": "return True",
    "call_signature": "(outputs)",
    "original_func_name": "custom",
}


@pytest.mark.parametrize(
    ("job_name", "params", "expected"),
    [
        ("fake_docker_job", {"x": 1}, True),
        ("invoke_scorer", {"serialized_scorer": json.dumps(_CUSTOM_SCORER)}, True),
        (
            "invoke_scorer",
            {"serialized_scorer": json.dumps({"name": "s", "builtin_scorer_class": "Safety"})},
            False,
        ),
        ("invoke_scorer", {"serialized_scorer": "not json"}, False),
        ("invoke_scorer", {}, False),
        ("invoke_genai_evaluate", {"serialized_scorers": [json.dumps(_CUSTOM_SCORER)]}, False),
        ("run_online_trace_scorer", {}, False),
    ],
)
def test_supports_only_planned_jobs(executor, job_name, params, expected):
    assert executor.supports_job(job_name, params) is expected


def test_container_is_hardened_and_has_no_network(executor, client):
    _submit(executor)
    container = client.started[0]
    kwargs = container.kwargs

    assert kwargs["network_mode"] == "none"
    assert kwargs["read_only"] is True
    assert kwargs["cap_drop"] == ["ALL"]
    assert kwargs["security_opt"] == ["no-new-privileges:true"]
    assert kwargs["pids_limit"] == de._PIDS_LIMIT
    assert kwargs["mem_limit"] == kwargs["memswap_limit"] == de._DEFAULT_MEMORY_BYTES
    assert kwargs["nano_cpus"] == de._DEFAULT_NANO_CPUS
    assert kwargs["labels"][de.LABEL_JOB_ID] == "job-1"
    assert kwargs["labels"][de.LABEL_JOB_NAME] == "fake_docker_job"
    mounts = {spec["bind"]: spec["mode"] for spec in kwargs["volumes"].values()}
    assert mounts == {de._CONTAINER_INPUT_DIR: "ro", de._CONTAINER_OUTPUT_DIR: "rw"}
    env = kwargs["environment"]
    # The container runs the plan's function and gets no tracking or gateway address.
    assert env[MLFLOW_SERVER_JOB_FUNCTION_FULLNAME_ENV_VAR] == "some.module.compute"
    assert json.loads(env[MLFLOW_SERVER_JOB_PARAMS_ENV_VAR]) == {
        "inputs_path": f"{de._CONTAINER_INPUT_DIR}/inputs.json"
    }
    assert not any(key in env for key in ("MLFLOW_TRACKING_URI", "MLFLOW_GATEWAY_URI"))
    assert MLFLOW_SERVER_JOB_ID_ENV_VAR not in env
    assert kwargs["log_config"] == de._LOG_CONFIG
    (fsize,) = kwargs["ulimits"]
    assert (fsize.name, fsize.soft, fsize.hard) == (
        "fsize",
        de._MAX_RESULT_BYTES,
        de._MAX_RESULT_BYTES,
    )


def test_successful_run_returns_the_finalized_result(executor, client):
    _submit(executor)
    result = executor.wait_for_job("job-1")

    assert result.status == JobStatus.SUCCEEDED
    assert json.loads(result.result) == {"finalized": {"score": 1}}
    container = client.started[0]
    assert container.removed
    workdir = next(iter(container.kwargs["volumes"]))
    assert not Path(workdir).exists()


def test_missing_result_fails_with_container_logs(executor, client):
    client.on_wait = lambda container: {"StatusCode": 137}
    _submit(executor)
    result = executor.wait_for_job("job-1")

    assert result.status == JobStatus.FAILED
    assert "exited with code 137" in result.error_message
    assert "container log line" in result.error_message
    assert client.started[0].logs_tail == de._LOG_TAIL_LINES


def test_failed_job_reports_transient_flag(executor, client):
    def on_wait(container):
        _SubprocessJobResult(succeeded=False, is_transient_error=True, error="boom").dump(
            str(container._output_dir() / de._RESULT_FILE)
        )
        return {"StatusCode": 0}

    client.on_wait = on_wait
    _submit(executor)
    result = executor.wait_for_job("job-1")

    assert result.status == JobStatus.FAILED
    assert result.error_message == "boom"
    assert result.is_transient_error


@pytest.mark.parametrize(
    "contents",
    [
        "not json",
        json.dumps({"succeeded": True, "extra": 1}),
        json.dumps(["succeeded"]),
        json.dumps({"succeeded": "no"}),
        json.dumps({"succeeded": False, "error": {"not": "a string"}}),
        json.dumps({"succeeded": False, "is_transient_error": "yes"}),
        json.dumps({"succeeded": True, "result": 1}),
        "[" * 100_000,
    ],
)
def test_malformed_result_fails(executor, client, contents):
    def on_wait(container):
        (container._output_dir() / de._RESULT_FILE).write_text(contents)
        return {"StatusCode": 0}

    client.on_wait = on_wait
    _submit(executor)
    result = executor.wait_for_job("job-1")

    assert result.status == JobStatus.FAILED
    assert "malformed result" in result.error_message


def test_oversized_result_fails(executor, client, monkeypatch):
    monkeypatch.setattr(de, "_MAX_RESULT_BYTES", 10)
    client.on_wait = _succeed_with({"score": "x" * 100})
    _submit(executor)
    result = executor.wait_for_job("job-1")

    assert result.status == JobStatus.FAILED
    assert "larger than" in result.error_message


def _write_result_entry(make_entry):
    def on_wait(container):
        make_entry(container._output_dir() / de._RESULT_FILE)
        return {"StatusCode": 0}

    return on_wait


def test_result_symlink_to_host_file_is_not_followed(executor, client, tmp_path):
    host_file = tmp_path / "host.json"
    _SubprocessJobResult(succeeded=True, result=json.dumps({"leak": "host"})).dump(str(host_file))
    client.on_wait = _write_result_entry(lambda path: path.symlink_to(host_file))
    _submit(executor)
    result = executor.wait_for_job("job-1")

    assert result.status == JobStatus.FAILED
    assert "could not be opened" in result.error_message


def test_result_symlink_to_device_is_not_read(executor, client):
    client.on_wait = _write_result_entry(lambda path: path.symlink_to("/dev/zero"))
    _submit(executor)
    result = executor.wait_for_job("job-1")

    assert result.status == JobStatus.FAILED


def test_result_fifo_is_rejected_without_blocking(executor, client):
    client.on_wait = _write_result_entry(lambda path: os.mkfifo(path))
    _submit(executor)
    result = executor.wait_for_job("job-1")

    assert result.status == JobStatus.FAILED
    assert "not a regular file" in result.error_message


def test_result_directory_is_rejected(executor, client):
    client.on_wait = _write_result_entry(lambda path: path.mkdir())
    _submit(executor)
    result = executor.wait_for_job("job-1")

    assert result.status == JobStatus.FAILED


@pytest.mark.skipif(os.name != "nt" and os.geteuid() == 0, reason="root can read any file")
def test_unreadable_result_fails_the_job(executor, client):
    def make_unreadable(path):
        _SubprocessJobResult(succeeded=True, result="1").dump(str(path))
        path.chmod(0)

    client.on_wait = _write_result_entry(make_unreadable)
    _submit(executor)
    result = executor.wait_for_job("job-1")

    assert result.status == JobStatus.FAILED
    assert "could not be opened" in result.error_message


def test_finalize_error_fails_the_job(executor, client, monkeypatch):
    def planner(params, input_dir, container_input_dir):
        def finalize(value):
            raise MlflowException("results for traces outside this job")

        return de.DockerJobPlan(fn_fullname="m.f", params={}, finalize=finalize)

    monkeypatch.setattr(
        de,
        "_load_function",
        lambda name: planner if name == de._JOB_PLANNERS["fake_docker_job"] else fake_docker_job,
    )
    _submit(executor)
    result = executor.wait_for_job("job-1")

    assert result.status == JobStatus.FAILED
    assert "outside this job" in result.error_message


def test_timeout_kills_the_container(executor, client):
    def on_wait(container):
        raise requests.exceptions.ReadTimeout("Read timed out.")

    client.on_wait = on_wait
    _submit(executor, timeout=1.0)
    result = executor.wait_for_job("job-1")

    assert result.status == JobStatus.TIMEOUT
    assert client.started[0].killed
    assert client.started[0].removed


def test_deadline_used_up_before_the_container_runs_reports_timeout(executor, client):
    # The deadline also covers the host-side planning step, which can use all of it.
    _submit(executor, timeout=1e-9)
    result = executor.wait_for_job("job-1")

    assert result.status == JobStatus.TIMEOUT
    assert client.started[0].killed


def test_lost_container_fails_the_job(executor, client):
    _submit(executor)
    client.get_error = RuntimeError("No such container")
    result = executor.wait_for_job("job-1")

    assert result.status == JobStatus.FAILED
    assert "Lost track of the container" in result.error_message
    assert executor._jobs == {}


def test_cancel_kills_the_container_and_reports_canceled(executor, client):
    def on_wait(container):
        executor.cancel_job("job-1")
        assert container.killed
        return {"StatusCode": 137}

    client.on_wait = on_wait
    _submit(executor)
    result = executor.wait_for_job("job-1")

    assert result.status == JobStatus.CANCELED


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"job_name": "invoke_genai_evaluate"}, "does not support job type"),
        ({"python_env": object()}, "cannot install"),
    ],
)
def test_rejects_unsupported_submissions(executor, client, kwargs, match):
    with pytest.raises(MlflowException, match=match):
        _submit(executor, **kwargs)
    assert client.started == []


def test_container_that_fails_to_start_is_removed(executor, client):
    client.start_error = RuntimeError("exec: python: not found")
    with pytest.raises(RuntimeError, match="python: not found"):
        _submit(executor)

    assert [c.removed for c in client.leftovers] == [True]
    assert executor._jobs == {}


def test_planner_error_fails_submission_and_cleans_up(executor, client, monkeypatch):
    def planner(params, input_dir, container_input_dir):
        raise MlflowException("Traces not found: ['tr-1']")

    monkeypatch.setattr(de, "_load_function", lambda name: planner)
    with pytest.raises(MlflowException, match="Traces not found"):
        _submit(executor)
    assert client.started == []
    assert executor._jobs == {}


def test_recover_removes_leftover_containers_and_requeues(executor, client):
    leftover = _FakeContainer(client, {"volumes": {}})
    client.leftovers = [leftover]

    results = executor.recover_jobs(["job-1"])

    assert [(r.job_id, r.action) for r in results] == [("job-1", "requeue")]
    assert leftover.killed
    assert leftover.removed


def test_stop_executor_kills_in_flight_containers(executor, client):
    _submit(executor)
    executor.stop_executor()

    assert client.started[0].killed
    with pytest.raises(MlflowException, match="stopped"):
        _submit(executor, job_id="job-2")


def test_job_stopped_mid_run_is_left_for_recovery(executor, client):
    # Raising (instead of reporting FAILED) makes the runner mark the job for recovery.
    def on_wait(container):
        executor.stop_executor()
        return {"StatusCode": 137}

    client.on_wait = on_wait
    _submit(executor)
    with pytest.raises(MlflowException, match="recovered on the next start"):
        executor.wait_for_job("job-1")
    assert client.started[0].removed


def test_no_container_starts_after_stop(executor, client, monkeypatch):
    def planner(params, input_dir, container_input_dir):
        # Shutdown happens while the host is still preparing the job's inputs.
        executor.stop_executor()
        return de.DockerJobPlan(fn_fullname="m.f", params={}, finalize=lambda value: value)

    monkeypatch.setattr(
        de,
        "_load_function",
        lambda name: planner if name == de._JOB_PLANNERS["fake_docker_job"] else fake_docker_job,
    )
    _submit(executor)

    assert client.started == []
    with pytest.raises(MlflowException, match="recovered on the next start"):
        executor.wait_for_job("job-1")


def test_check_requirements_rejects_malformed_container_settings(executor, monkeypatch):
    monkeypatch.setenv("MLFLOW_JOB_EXTRA_LABELS", "not json")
    with pytest.raises(MlflowException, match="MLFLOW_JOB_EXTRA_LABELS"):
        executor.check_requirements()


def test_extra_labels_cannot_override_job_labels(executor, client, monkeypatch):
    monkeypatch.setenv(
        "MLFLOW_JOB_EXTRA_LABELS", json.dumps({"team": "ml", de.LABEL_JOB_ID: "spoofed"})
    )
    _submit(executor)
    labels = client.started[0].kwargs["labels"]

    assert labels["team"] == "ml"
    assert labels[de.LABEL_JOB_ID] == "job-1"


@pytest.mark.parametrize("raw", ["not json", json.dumps(["a"]), json.dumps({"k": 1})])
def test_invalid_extra_labels_are_rejected(raw, monkeypatch):
    monkeypatch.setenv("MLFLOW_JOB_EXTRA_LABELS", raw)
    with pytest.raises(MlflowException, match="MLFLOW_JOB_EXTRA_LABELS"):
        de._container_labels("job-1", "fake_docker_job")


def test_resource_limit_override(monkeypatch):
    monkeypatch.setenv(
        "MLFLOW_JOB_RESOURCE_LIMITS_fake_docker_job", json.dumps({"cpu": "500m", "memory": "512Mi"})
    )
    assert de._resource_limits("fake_docker_job") == (512 * 1024**2, 500_000_000)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1048576", 1048576),
        ("1Ki", 1024),
        ("2Gi", 2 * 1024**3),
        ("1G", 1000**3),
        ("512M", 512 * 1000**2),
    ],
)
def test_parse_memory(value, expected):
    assert de._parse_memory(value) == expected


@pytest.mark.parametrize(("value", "expected"), [("500m", 500_000_000), ("2", 2_000_000_000)])
def test_parse_cpu(value, expected):
    assert de._parse_cpu(value) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        json.dumps({"cpu": "-1"}),
        json.dumps({"memory": "lots"}),
        json.dumps({"cpu": "inf"}),
        json.dumps({"memory": "inf"}),
        json.dumps({"memory": "nan"}),
    ],
)
def test_invalid_resource_limits_are_rejected(raw, monkeypatch):
    monkeypatch.setenv("MLFLOW_JOB_RESOURCE_LIMITS_fake_docker_job", raw)
    with pytest.raises(MlflowException, match="MLFLOW_JOB_RESOURCE_LIMITS_fake_docker_job"):
        de._resource_limits("fake_docker_job")
