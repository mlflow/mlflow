import json
import uuid
from unittest import mock

import pytest
from docker import errors as docker_errors

import mlflow
from mlflow.entities._job_status import JobStatus
from mlflow.genai import scorer
from mlflow.genai.scorers.online import container_scoring
from mlflow.genai.scorers.online.container_scoring import (
    OnlineContainerScorer,
    container_scoring_enabled,
)
from mlflow.genai.scorers.online.entities import OnlineScorer, OnlineScoringConfig
from mlflow.server.constants import BACKEND_STORE_URI_ENV_VAR
from mlflow.server.jobs.executor import JobResult
from mlflow.server.jobs.job_container import _CONTAINER_INPUT_DIR
from mlflow.server.jobs.utils import _load_function
from mlflow.server.sandbox.container import SandboxUnavailableError
from mlflow.tracking._tracking_service.utils import _get_store

_JOB_NAME = "run_online_trace_scorer"


def _custom_online_scorer(experiment_id: str) -> OnlineScorer:
    @scorer
    def output_length(outputs) -> int:
        return len(str(outputs))

    return OnlineScorer(
        name="output_length",
        serialized_scorer=json.dumps(output_length.model_dump()),
        online_config=OnlineScoringConfig(
            online_scoring_config_id=uuid.uuid4().hex,
            scorer_id=uuid.uuid4().hex,
            sample_rate=1.0,
            experiment_id=experiment_id,
            filter_string=None,
        ),
    )


@pytest.fixture
def traces(db_uri, monkeypatch):
    mlflow.set_tracking_uri(db_uri)
    monkeypatch.setenv(BACKEND_STORE_URI_ENV_VAR, db_uri)
    monkeypatch.setenv("MLFLOW_SERVER_ENABLE_CUSTOM_SCORERS", "true")
    experiment_id = mlflow.set_experiment("online-container").experiment_id
    trace_ids = []
    for i in range(3):
        with mlflow.start_span(name=f"span_{i}") as span:
            span.set_inputs({"question": i})
            span.set_outputs({"answer": "x" * (i + 1)})
        trace_ids.append(span.trace_id)
    with mock.patch("mlflow.genai.scorers.job._get_tracking_store", return_value=_get_store()):
        yield experiment_id, [mlflow.get_trace(trace_id) for trace_id in trace_ids]


@pytest.fixture
def docker(monkeypatch):
    client = mock.Mock(name="docker-client")
    calls = mock.Mock()
    monkeypatch.setattr(container_scoring, "job_docker_client", lambda: client)
    monkeypatch.setattr(container_scoring, "prepare_job_image", calls.prepare_job_image)
    monkeypatch.setattr(container_scoring, "remove_job_containers", calls.remove_job_containers)
    monkeypatch.setattr(container_scoring, "job_image", lambda: "mlflow-job:test")
    return client, calls


def _run_in_process(client, image, *, plan, input_dir, **kwargs):
    # Same function the job container runs, reading the inputs the host wrote for it.
    params = {
        key: value.replace(_CONTAINER_INPUT_DIR, str(input_dir))
        if isinstance(value, str)
        else value
        for key, value in plan.params.items()
    }
    value = json.loads(json.dumps(_load_function(plan.fn_fullname)(**params)))
    return JobResult(status=JobStatus.SUCCEEDED, result=json.dumps(plan.finalize(value)))


def test_container_scoring_is_enabled_by_the_docker_custom_scorer_backend(monkeypatch):
    monkeypatch.delenv("MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND", raising=False)
    assert not container_scoring_enabled()
    monkeypatch.setenv("MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND", "docker")
    assert container_scoring_enabled()


def test_scores_in_batches_and_logs_on_the_host(traces, docker, monkeypatch):
    experiment_id, trace_list = traces
    client, calls = docker
    monkeypatch.setenv("MLFLOW_SERVER_SCORER_INVOKE_BATCH_SIZE", "2")
    run = mock.Mock(side_effect=_run_in_process)
    monkeypatch.setattr(container_scoring, "run_plan_in_container", run)
    on_failure = mock.Mock()

    OnlineContainerScorer(experiment_id, _JOB_NAME).score(
        [(_custom_online_scorer(experiment_id), trace_list)], on_failure
    )

    on_failure.assert_not_called()
    assert run.call_count == 2
    assert [c.kwargs["job_name"] for c in run.call_args_list] == [_JOB_NAME, _JOB_NAME]
    label = f"{_JOB_NAME}-{experiment_id}"
    assert {c.kwargs["job_id"] for c in run.call_args_list} == {label}
    # Leftovers from an earlier job are removed and the image prepared once, before the first run.
    calls.remove_job_containers.assert_called_once_with(client, label)
    calls.prepare_job_image.assert_called_once_with(client)
    for trace in trace_list:
        assessments = mlflow.get_trace(trace.info.trace_id).info.assessments
        assert [a.name for a in assessments] == ["output_length"]


def test_reports_each_failed_batch(traces, docker, monkeypatch):
    experiment_id, trace_list = traces
    monkeypatch.setenv("MLFLOW_SERVER_SCORER_INVOKE_BATCH_SIZE", "2")
    results = [
        JobResult(status=JobStatus.FAILED, error_message="scorer crashed"),
        RuntimeError("daemon unreachable"),
    ]
    monkeypatch.setattr(container_scoring, "run_plan_in_container", mock.Mock(side_effect=results))
    on_failure = mock.Mock()
    online_scorer = _custom_online_scorer(experiment_id)

    OnlineContainerScorer(experiment_id, _JOB_NAME).score([(online_scorer, trace_list)], on_failure)

    assert on_failure.call_count == 2
    first, second = on_failure.call_args_list
    assert first.args == (online_scorer, trace_list[:2], "scorer crashed")
    assert second.args[:2] == (online_scorer, trace_list[2:])
    assert "daemon unreachable" in second.args[2]
    for trace in trace_list:
        assert mlflow.get_trace(trace.info.trace_id).info.assessments == []


def test_batches_share_the_time_left_and_take_at_most_half_the_job_timeout(
    traces, docker, monkeypatch
):
    experiment_id, trace_list = traces
    monkeypatch.setenv("MLFLOW_SERVER_SCORER_INVOKE_BATCH_SIZE", "1")
    monkeypatch.setenv("MLFLOW_SERVER_JOB_DEFAULT_TIMEOUT", "100")
    timeouts = []

    def run(client, image, *, timeout, **kwargs):
        timeouts.append(timeout)
        return JobResult(status=JobStatus.SUCCEEDED, result="{}")

    monkeypatch.setattr(container_scoring, "run_plan_in_container", run)
    # A clock that does not move, so the time left only shrinks by what batches do not use.
    monkeypatch.setattr(container_scoring, "time", mock.Mock(monotonic=lambda: 1000.0))
    online_scorer = _custom_online_scorer(experiment_id)

    OnlineContainerScorer(experiment_id, _JOB_NAME).score(
        [(online_scorer, trace_list)], mock.Mock()
    )

    # Batches share the time left minus a reserve of 10% of the job timeout, so time a fast batch
    # does not use goes to the next ones, and no batch gets more than half the job timeout.
    assert timeouts == [30, 45, 50]

    timeouts.clear()
    OnlineContainerScorer(experiment_id, _JOB_NAME).score(
        [(online_scorer, trace_list[:1])], mock.Mock()
    )
    assert timeouts == [50]


def test_reports_traces_the_scorer_failed_on_or_left_out(traces, docker, monkeypatch):
    experiment_id, trace_list = traces
    first, second, third = (t.info.trace_id for t in trace_list)
    container_result = {
        first: {"assessments": [], "failures": [{"error_code": "E", "error_message": "boom"}]},
        second: {"assessments": [{"name": "output_length"}], "failures": []},
    }
    monkeypatch.setattr(
        container_scoring,
        "run_plan_in_container",
        lambda *args, **kwargs: JobResult(
            status=JobStatus.SUCCEEDED, result=json.dumps(container_result)
        ),
    )
    on_failure = mock.Mock()
    online_scorer = _custom_online_scorer(experiment_id)

    OnlineContainerScorer(experiment_id, _JOB_NAME).score([(online_scorer, trace_list)], on_failure)

    reported = {call.args[1][0].info.trace_id: call.args[2] for call in on_failure.call_args_list}
    assert reported == {
        first: "E: boom",
        third: "The job container returned no result for this trace.",
    }


def test_failed_setup_is_retried_a_few_times_then_not_for_every_batch(traces, docker, monkeypatch):
    experiment_id, trace_list = traces
    _, calls = docker
    monkeypatch.setenv("MLFLOW_SERVER_SCORER_INVOKE_BATCH_SIZE", "1")
    calls.prepare_job_image.side_effect = RuntimeError("cannot pull image")
    clock = mock.Mock(monotonic=lambda: 1000.0)
    monkeypatch.setattr(container_scoring, "time", clock)
    run = mock.Mock()
    monkeypatch.setattr(container_scoring, "run_plan_in_container", run)
    on_failure = mock.Mock()

    OnlineContainerScorer(experiment_id, _JOB_NAME).score(
        [(_custom_online_scorer(experiment_id), trace_list)], on_failure
    )

    # Retried after each delay during the first batch, and not again for the later batches.
    assert calls.prepare_job_image.call_count == 4
    assert [c.args[0] for c in clock.sleep.call_args_list] == [5, 15, 30]
    run.assert_not_called()
    assert on_failure.call_count == 3
    assert all(
        "Could not set up job containers" in call.args[2] and "cannot pull image" in call.args[2]
        for call in on_failure.call_args_list
    )


def test_scoring_continues_when_a_failed_setup_works_on_retry(traces, docker, monkeypatch):
    experiment_id, trace_list = traces
    _, calls = docker
    calls.prepare_job_image.side_effect = [RuntimeError("daemon restarting"), None]
    clock = mock.Mock(monotonic=lambda: 1000.0)
    monkeypatch.setattr(container_scoring, "time", clock)
    monkeypatch.setattr(container_scoring, "run_plan_in_container", _run_in_process)
    on_failure = mock.Mock()

    OnlineContainerScorer(experiment_id, _JOB_NAME).score(
        [(_custom_online_scorer(experiment_id), trace_list)], on_failure
    )

    clock.sleep.assert_called_once_with(5)
    on_failure.assert_not_called()
    for trace in trace_list:
        assert [a.name for a in mlflow.get_trace(trace.info.trace_id).info.assessments] == [
            "output_length"
        ]


def test_failed_setup_is_not_retried_without_time_left_for_it(traces, docker, monkeypatch):
    experiment_id, trace_list = traces
    _, calls = docker
    # A 5 second job timeout leaves 4.5 seconds after the reserve, less than the first delay.
    monkeypatch.setenv("MLFLOW_SERVER_JOB_DEFAULT_TIMEOUT", "5")
    calls.prepare_job_image.side_effect = RuntimeError("cannot pull image")
    clock = mock.Mock(monotonic=lambda: 1000.0)
    monkeypatch.setattr(container_scoring, "time", clock)
    on_failure = mock.Mock()

    OnlineContainerScorer(experiment_id, _JOB_NAME).score(
        [(_custom_online_scorer(experiment_id), trace_list)], on_failure
    )

    calls.prepare_job_image.assert_called_once()
    clock.sleep.assert_not_called()
    on_failure.assert_called_once()
    assert "Could not set up job containers" in on_failure.call_args.args[2]


def _moving_clock(monkeypatch):
    # A clock that moves only by the time slept.
    now = [1000.0]
    clock = mock.Mock(monotonic=lambda: now[0])
    clock.sleep.side_effect = lambda seconds: now.__setitem__(0, now[0] + seconds)
    monkeypatch.setattr(container_scoring, "time", clock)
    return clock


def test_failed_setup_is_retried_only_while_the_job_has_time_left(traces, docker, monkeypatch):
    experiment_id, trace_list = traces
    _, calls = docker
    # 36 seconds to use after the 4 second reserve: the 5 and 15 second retries fit, 30 does not.
    monkeypatch.setenv("MLFLOW_SERVER_JOB_DEFAULT_TIMEOUT", "40")
    calls.prepare_job_image.side_effect = RuntimeError("cannot pull image")
    clock = _moving_clock(monkeypatch)

    OnlineContainerScorer(experiment_id, _JOB_NAME).score(
        [(_custom_online_scorer(experiment_id), trace_list)], mock.Mock()
    )

    assert [c.args[0] for c in clock.sleep.call_args_list] == [5, 15]
    assert calls.prepare_job_image.call_count == 3


def test_batches_are_not_given_the_time_a_retried_setup_took(traces, docker, monkeypatch):
    experiment_id, trace_list = traces
    _, calls = docker
    monkeypatch.setenv("MLFLOW_SERVER_SCORER_INVOKE_BATCH_SIZE", "1")
    monkeypatch.setenv("MLFLOW_SERVER_JOB_DEFAULT_TIMEOUT", "100")
    calls.prepare_job_image.side_effect = [RuntimeError("daemon restarting"), None]
    _moving_clock(monkeypatch)
    timeouts = []

    def run(client, image, *, timeout, **kwargs):
        timeouts.append(timeout)
        return JobResult(status=JobStatus.SUCCEEDED, result="{}")

    monkeypatch.setattr(container_scoring, "run_plan_in_container", run)

    OnlineContainerScorer(experiment_id, _JOB_NAME).score(
        [(_custom_online_scorer(experiment_id), trace_list)], mock.Mock()
    )

    # After the 5 second retry, 85 of the 90 seconds before the reserve are left for 3 batches.
    assert timeouts[0] == pytest.approx(85 / 3)


def test_lasting_setup_errors_are_not_retried(traces, docker, monkeypatch):
    experiment_id, trace_list = traces
    _, calls = docker
    calls.prepare_job_image.side_effect = docker_errors.ImageNotFound("no such image")
    clock = _moving_clock(monkeypatch)
    on_failure = mock.Mock()

    OnlineContainerScorer(experiment_id, _JOB_NAME).score(
        [(_custom_online_scorer(experiment_id), trace_list)], on_failure
    )

    calls.prepare_job_image.assert_called_once()
    clock.sleep.assert_not_called()
    assert "no such image" in on_failure.call_args.args[2]


def test_a_failed_default_image_build_is_not_retried(traces, docker, monkeypatch):
    experiment_id, trace_list = traces
    _, calls = docker
    build_error = SandboxUnavailableError("pip install failed")
    build_error.__cause__ = docker_errors.BuildError("pip install failed", build_log=[])
    calls.prepare_job_image.side_effect = build_error
    clock = _moving_clock(monkeypatch)

    OnlineContainerScorer(experiment_id, _JOB_NAME).score(
        [(_custom_online_scorer(experiment_id), trace_list)], mock.Mock()
    )

    calls.prepare_job_image.assert_called_once()
    clock.sleep.assert_not_called()
