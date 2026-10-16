import json
import os

import pytest
import requests

from mlflow.entities._job_status import JobStatus
from mlflow.server.jobs import job_container as jc
from mlflow.server.jobs.utils import MLFLOW_SERVER_JOB_PARAMS_ENV_VAR, _SubprocessJobResult

from tests.server.jobs.docker_fakes import FakeClient, FakeContainer, succeed_with

pytestmark = pytest.mark.skipif(os.name == "nt", reason="Job containers require a POSIX host")


class _TransientError(Exception):
    pass


def _plan():
    return jc.DockerJobPlan(
        fn_fullname="some.module.compute",
        params={"inputs_path": f"{jc._CONTAINER_INPUT_DIR}/inputs.json"},
        finalize=lambda value: {"finalized": value},
    )


def _run(client, tmp_path, timeout=30.0, plan=None, **kwargs):
    input_dir, output_dir = jc.make_job_container_dirs(tmp_path)
    return jc.run_plan_in_container(
        client,
        "mlflow-job:test",
        job_id="online-1",
        job_name="run_online_trace_scorer",
        plan=plan or _plan(),
        input_dir=input_dir,
        output_dir=output_dir,
        timeout=timeout,
        **kwargs,
    )


def test_runs_the_plan_and_finalizes_its_result(tmp_path):
    client = FakeClient(succeed_with({"score": 1}))

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
    client = FakeClient(succeed_with(None))

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
            str(container._output_dir() / jc._RESULT_FILE)
        )
        return {"StatusCode": 1}

    client = FakeClient(on_wait)

    result = _run(client, tmp_path)

    assert result.status == JobStatus.FAILED
    assert result.error_message == "boom"
    assert result.is_transient_error
    assert client.started[0].removed


def test_missing_result_fails_with_container_logs(tmp_path):
    client = FakeClient(lambda container: {"StatusCode": 137})

    result = _run(client, tmp_path)

    assert result.status == JobStatus.FAILED
    assert "without writing a result" in result.error_message
    assert "container log line" in result.error_message


def test_timeout_kills_and_removes_the_container(tmp_path):
    def on_wait(container):
        raise requests.exceptions.ReadTimeout("Read timed out.")

    client = FakeClient(on_wait)

    result = _run(client, tmp_path, timeout=1.0)

    assert result.status == JobStatus.TIMEOUT
    assert client.started[0].killed
    assert client.started[0].removed


def test_lost_container_fails_and_is_removed(tmp_path):
    def on_wait(container):
        raise requests.exceptions.ConnectionError("daemon went away")

    client = FakeClient(on_wait)

    result = _run(client, tmp_path)

    assert result.status == JobStatus.FAILED
    assert "Lost track of the container" in result.error_message
    assert client.started[0].removed


def test_container_that_fails_to_start_is_removed(tmp_path):
    client = FakeClient(succeed_with(None))
    client.start_error = RuntimeError("exec: python: not found")

    with pytest.raises(RuntimeError, match="python: not found"):
        _run(client, tmp_path)

    assert [c.removed for c in client.leftovers] == [True]


def test_no_container_starts_without_time_left(tmp_path):
    client = FakeClient(succeed_with(None))

    result = _run(client, tmp_path, timeout=0)

    assert result.status == JobStatus.TIMEOUT
    assert client.started == []


def test_finalize_error_fails_the_run_and_removes_the_container(tmp_path):
    def finalize(value):
        raise ValueError("bad result")

    plan = jc.DockerJobPlan(fn_fullname="some.module.compute", params={}, finalize=finalize)
    client = FakeClient(succeed_with({"score": 1}))

    result = _run(client, tmp_path, plan=plan)

    assert result.status == JobStatus.FAILED
    assert "bad result" in result.error_message
    assert client.started[0].removed


def test_cleanup_removes_only_this_servers_containers_for_the_job():
    client = FakeClient(succeed_with(None))
    labels = jc._container_labels("online-1", "run_online_trace_scorer")
    ours = FakeContainer(client, {"volumes": {}, "labels": labels})
    other_job = FakeContainer(client, {"volumes": {}, "labels": {**labels, jc.LABEL_JOB_ID: "x"}})
    other_server = FakeContainer(
        client, {"volumes": {}, "labels": {**labels, jc.LABEL_BACKEND_STORE: "elsewhere"}}
    )
    client.leftovers = [ours, other_job, other_server]

    jc.remove_job_containers(client, "online-1")

    assert [c.removed for c in client.leftovers] == [True, False, False]
