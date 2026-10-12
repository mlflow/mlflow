import json
import os
from pathlib import Path

import pytest
import requests

from mlflow.entities._job_status import JobStatus
from mlflow.server.jobs import job_container as jc
from mlflow.server.jobs.utils import MLFLOW_SERVER_JOB_PARAMS_ENV_VAR, _SubprocessJobResult

pytestmark = pytest.mark.skipif(os.name == "nt", reason="Job containers require a POSIX host")


class _TransientError(Exception):
    pass


class _FakeContainer:
    def __init__(self, kwargs, on_wait):
        self.id = "container-1"
        self.short_id = self.id
        self.kwargs = kwargs
        self.on_wait = on_wait
        self.killed = False
        self.removed = False

    def output_dir(self) -> Path:
        for host, spec in self.kwargs["volumes"].items():
            if spec["bind"] == jc._CONTAINER_OUTPUT_DIR:
                return Path(host)

    def wait(self, timeout=None):
        return self.on_wait(self)

    def logs(self, tail=None):
        return b"container log line"

    def kill(self):
        self.killed = True

    def remove(self, force=False):
        self.removed = True


class _FakeContainers:
    def __init__(self, client):
        self._client = client

    def run(self, image, **kwargs):
        container = _FakeContainer(kwargs, self._client.on_wait)
        if self._client.start_error is not None:
            # Like docker-py: the container is created, then starting it fails.
            self._client.leftovers.append(container)
            raise self._client.start_error
        self._client.started.append(container)
        return container

    def list(self, all=False, filters=None):
        return list(self._client.leftovers)


class _FakeClient:
    def __init__(self, on_wait):
        self.on_wait = on_wait
        self.started = []
        self.leftovers = []
        self.start_error = None
        self.containers = _FakeContainers(self)


def _succeed_with(value):
    def on_wait(container):
        _SubprocessJobResult(succeeded=True, result=json.dumps(value)).dump(
            str(container.output_dir() / jc._RESULT_FILE)
        )
        return {"StatusCode": 0}

    return on_wait


def _plan():
    return jc.DockerJobPlan(
        fn_fullname="some.module.compute",
        params={"inputs_path": f"{jc._CONTAINER_INPUT_DIR}/inputs.json"},
        finalize=lambda value: {"finalized": value},
    )


def _run(client, tmp_path, timeout=30.0, **kwargs):
    jc.make_job_container_dirs(tmp_path)
    return jc.run_plan_in_container(
        client,
        "mlflow-job:test",
        job_id="online-1",
        job_name="run_online_trace_scorer",
        plan=_plan(),
        workdir=tmp_path,
        timeout=timeout,
        **kwargs,
    )


def test_runs_the_plan_and_finalizes_its_result(tmp_path):
    client = _FakeClient(_succeed_with({"score": 1}))

    result = _run(client, tmp_path)

    assert result.status == JobStatus.SUCCEEDED
    assert json.loads(result.result) == {"finalized": {"score": 1}}
    container = client.started[0]
    assert container.kwargs["network_mode"] == "none"
    assert json.loads(container.kwargs["environment"][MLFLOW_SERVER_JOB_PARAMS_ENV_VAR]) == {
        "inputs_path": f"{jc._CONTAINER_INPUT_DIR}/inputs.json"
    }
    assert container.removed


def test_applies_resource_limits_and_transient_error_classes(tmp_path):
    client = _FakeClient(_succeed_with(None))

    _run(
        client,
        tmp_path,
        transient_error_classes=[_TransientError],
        resource_limits={"cpu": "2", "memory": "256Mi"},
    )

    kwargs = client.started[0].kwargs
    assert kwargs["mem_limit"] == 256 * 1024**2
    assert kwargs["nano_cpus"] == 2_000_000_000
    classes = (tmp_path / "input" / jc._TRANSIENT_ERROR_CLASSES_FILE).read_text()
    assert classes == f"{__name__}._TransientError\n"


def test_reports_a_failed_job_and_its_transient_flag(tmp_path):
    def on_wait(container):
        _SubprocessJobResult(succeeded=False, error="boom", is_transient_error=True).dump(
            str(container.output_dir() / jc._RESULT_FILE)
        )
        return {"StatusCode": 1}

    client = _FakeClient(on_wait)

    result = _run(client, tmp_path)

    assert result.status == JobStatus.FAILED
    assert result.error_message == "boom"
    assert result.is_transient_error
    assert client.started[0].removed


def test_missing_result_fails_with_container_logs(tmp_path):
    client = _FakeClient(lambda container: {"StatusCode": 137})

    result = _run(client, tmp_path)

    assert result.status == JobStatus.FAILED
    assert "without writing a result" in result.error_message
    assert "container log line" in result.error_message


def test_timeout_kills_and_removes_the_container(tmp_path):
    def on_wait(container):
        raise requests.exceptions.ReadTimeout("Read timed out.")

    client = _FakeClient(on_wait)

    result = _run(client, tmp_path, timeout=1.0)

    assert result.status == JobStatus.TIMEOUT
    assert client.started[0].killed
    assert client.started[0].removed


def test_lost_container_fails_and_is_removed(tmp_path):
    def on_wait(container):
        raise requests.exceptions.ConnectionError("daemon went away")

    client = _FakeClient(on_wait)

    result = _run(client, tmp_path)

    assert result.status == JobStatus.FAILED
    assert "Lost track of the container" in result.error_message
    assert client.started[0].removed


def test_container_that_fails_to_start_is_removed(tmp_path):
    client = _FakeClient(_succeed_with(None))
    client.start_error = RuntimeError("exec: python: not found")

    with pytest.raises(RuntimeError, match="python: not found"):
        _run(client, tmp_path)

    assert [c.removed for c in client.leftovers] == [True]
