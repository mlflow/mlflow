import json
import uuid
from unittest import mock

import pytest

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
    online_scorer = _custom_online_scorer(experiment_id)

    OnlineContainerScorer(experiment_id, _JOB_NAME).score(
        [(online_scorer, trace_list)], mock.Mock()
    )

    # Each batch gets 90% of the time left over the batches left, so time a fast batch does not
    # use goes to the next ones, and no batch gets more than half the job timeout.
    assert timeouts == [pytest.approx(30, abs=1), pytest.approx(45, abs=1), 50]

    timeouts.clear()
    OnlineContainerScorer(experiment_id, _JOB_NAME).score(
        [(online_scorer, trace_list[:1])], mock.Mock()
    )
    assert timeouts == [50]
