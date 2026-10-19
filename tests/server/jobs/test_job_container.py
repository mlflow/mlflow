import json
import os

import docker
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


def _killed_for_memory(container):
    container.oom_killed = True
    return {"StatusCode": 137}


def test_container_out_of_memory_names_the_limit_and_the_setting(tmp_path):
    client = FakeClient(_killed_for_memory)

    result = _run(client, tmp_path, resource_limits={"memory": "256Mi"})

    assert result.status == JobStatus.FAILED
    assert "exited with code 137" in result.error_message
    assert "Docker reports that the container was stopped because it ran out of memory" in (
        result.error_message
    )
    assert "its memory limit is 256Mi" in result.error_message
    assert (
        'MLFLOW_JOB_RESOURCE_LIMITS_run_online_trace_scorer, for example {"memory": "512Mi"}'
        in (result.error_message)
    )
    assert "container log line" in result.error_message
    # The container's attributes are from when it started, so the final state needs a reload.
    assert client.started[0].reloads == 1
    assert client.started[0].removed


def test_out_of_memory_is_named_when_a_partial_result_was_written(tmp_path):
    def on_wait(container):
        (container._output_dir() / jc._RESULT_FILE).write_text('{"succeeded": tr')
        container.oom_killed = True
        return {"StatusCode": 137}

    result = _run(FakeClient(on_wait), tmp_path)

    assert result.status == JobStatus.FAILED
    assert "ran out of memory (its memory limit is 1Gi)" in result.error_message


def test_out_of_memory_is_named_with_a_failure_result(tmp_path):
    def on_wait(container):
        _SubprocessJobResult(succeeded=False, error="MemoryError").dump(
            str(container._output_dir() / jc._RESULT_FILE)
        )
        container.oom_killed = True
        return {"StatusCode": 1}

    result = _run(FakeClient(on_wait), tmp_path)

    assert result.error_message.startswith("MemoryError Docker reports that the container")


def test_exit_without_running_out_of_memory_keeps_the_plain_message(tmp_path):
    result = _run(FakeClient(lambda container: {"StatusCode": 137}), tmp_path)

    assert result.status == JobStatus.FAILED
    assert "without writing a result. Container logs:" in result.error_message
    assert "memory" not in result.error_message


def _state_unreadable(self):
    raise docker.errors.NotFound("No such container")


def test_unreadable_container_state_keeps_the_plain_message(tmp_path, monkeypatch):
    monkeypatch.setattr(FakeContainer, "reload", _state_unreadable)

    result = _run(FakeClient(_killed_for_memory), tmp_path)

    assert result.status == JobStatus.FAILED
    assert "without writing a result. Container logs:" in result.error_message
    assert "memory" not in result.error_message


@pytest.mark.parametrize(
    ("num_bytes", "shown"),
    [(2 * 1024**3, "2Gi"), (512 * 1024**2, "512Mi"), (10**9, "1G"), (3 * 1024**4, "3Ti"), (7, "7")],
)
def test_format_memory_uses_the_suffixes_limits_accept(num_bytes, shown):
    assert jc._format_memory(num_bytes) == shown
    assert jc._parse_memory(shown) == num_bytes


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
