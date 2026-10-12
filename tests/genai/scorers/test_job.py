import json
from unittest import mock

import pytest

import mlflow
from mlflow.entities import AssessmentSource, AssessmentSourceType
from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.exceptions import MlflowException
from mlflow.genai import scorer
from mlflow.genai.scorers.base import SCORER_BACKEND_TRACKING
from mlflow.genai.scorers.job import _plan_invoke_scorer_job_for_docker, invoke_scorer_job
from mlflow.server.constants import BACKEND_STORE_URI_ENV_VAR
from mlflow.server.jobs.utils import _load_function
from mlflow.tracing.constant import AssessmentMetadataKey, TraceMetadataKey
from mlflow.tracking._tracking_service.utils import _get_store
from mlflow.utils.workspace_context import WorkspaceContext


def test_invoke_scorer_job_restores_registered_version():
    scorer = mock.MagicMock(is_session_level_scorer=False)

    with (
        mock.patch("mlflow.genai.scorers.job.Scorer.model_validate_json", return_value=scorer),
        mock.patch("mlflow.genai.scorers.job._get_tracking_store"),
        mock.patch("mlflow.genai.scorers.job._run_single_turn_scorer_batch", return_value={}),
    ):
        result = invoke_scorer_job(
            experiment_id="exp-123",
            serialized_scorer='{"name":"registered-judge"}',
            scorer_version=3,
            trace_ids=["trace-1"],
        )

    scorer._set_registration_metadata.assert_called_once_with(
        backend=SCORER_BACKEND_TRACKING,
        experiment_id="exp-123",
        sampling_config=None,
        scorer_version=3,
    )
    assert result == {}


def _custom_scorer_json() -> str:
    @scorer
    def output_length(outputs) -> int:
        return len(str(outputs))

    return json.dumps(output_length.model_dump())


@pytest.fixture
def sqlite_tracking(db_uri, monkeypatch):
    mlflow.set_tracking_uri(db_uri)
    monkeypatch.setenv(BACKEND_STORE_URI_ENV_VAR, db_uri)
    monkeypatch.setenv("MLFLOW_SERVER_ENABLE_CUSTOM_SCORERS", "true")
    experiment_id = mlflow.set_experiment("docker-plan").experiment_id
    trace_ids = []
    for i in range(2):
        with mlflow.start_span(name=f"span_{i}") as span:
            span.set_inputs({"question": i})
            span.set_outputs({"answer": "x" * (i + 1)})
        trace_ids.append(span.trace_id)
    with mock.patch("mlflow.genai.scorers.job._get_tracking_store", return_value=_get_store()):
        yield experiment_id, trace_ids


def _run_container_step(plan):
    # Same function the job container runs; it reads only the input file and returns its results.
    return json.loads(json.dumps(_load_function(plan.fn_fullname)(**plan.params)))


def test_docker_plan_scores_in_container_step_and_logs_on_host(sqlite_tracking, tmp_path):
    experiment_id, trace_ids = sqlite_tracking
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    plan = _plan_invoke_scorer_job_for_docker(
        params={
            "experiment_id": experiment_id,
            "serialized_scorer": _custom_scorer_json(),
            "trace_ids": trace_ids,
            "log_assessments": True,
        },
        input_dir=input_dir,
        container_input_dir=str(input_dir),
    )
    final = plan.finalize(_run_container_step(plan))

    assert set(final) == set(trace_ids)
    for trace_id in trace_ids:
        assert final[trace_id]["failures"] == []
        assessments = mlflow.get_trace(trace_id).info.assessments
        assert [a.name for a in assessments] == ["output_length"]


def test_docker_plan_does_not_log_when_log_assessments_is_false(sqlite_tracking, tmp_path):
    experiment_id, trace_ids = sqlite_tracking
    plan = _plan_invoke_scorer_job_for_docker(
        params={
            "experiment_id": experiment_id,
            "serialized_scorer": _custom_scorer_json(),
            "trace_ids": trace_ids,
            "log_assessments": False,
        },
        input_dir=tmp_path,
        container_input_dir=str(tmp_path),
    )
    final = plan.finalize(_run_container_step(plan))

    assert all(final[trace_id]["assessments"] for trace_id in trace_ids)
    assert all(not mlflow.get_trace(trace_id).info.assessments for trace_id in trace_ids)


def test_docker_plan_rejects_results_for_traces_outside_the_job(sqlite_tracking, tmp_path):
    experiment_id, trace_ids = sqlite_tracking
    plan = _plan_invoke_scorer_job_for_docker(
        params={
            "experiment_id": experiment_id,
            "serialized_scorer": _custom_scorer_json(),
            "trace_ids": trace_ids[:1],
        },
        input_dir=tmp_path,
        container_input_dir=str(tmp_path),
    )
    forged = {trace_ids[1]: {"assessments": [], "failures": []}}

    with pytest.raises(MlflowException, match="outside this job"):
        plan.finalize(forged)


@pytest.mark.parametrize(
    "serialized",
    [
        {"name": "safety", "builtin_scorer_class": "Safety"},
        {
            "name": "mixed",
            "ensemble_scorer_data": {
                "ensemble_fn": "majority",
                "scorers": [
                    {
                        "name": "custom",
                        "call_source": "return True",
                        "call_signature": "(outputs)",
                        "original_func_name": "custom",
                    },
                    {"name": "safety", "builtin_scorer_class": "Safety"},
                ],
            },
        },
    ],
)
def test_docker_plan_rejects_scorers_that_need_network_access(serialized, tmp_path):
    with pytest.raises(MlflowException, match="supports only custom @scorer scorers"):
        _plan_invoke_scorer_job_for_docker(
            params={
                "experiment_id": "0",
                "serialized_scorer": json.dumps(serialized),
                "trace_ids": ["tr-1"],
            },
            input_dir=tmp_path,
            container_input_dir=str(tmp_path),
        )


def _docker_plan(sqlite_tracking, tmp_path, log_assessments=True, scorer_version=None):
    experiment_id, trace_ids = sqlite_tracking
    return _plan_invoke_scorer_job_for_docker(
        params={
            "experiment_id": experiment_id,
            "serialized_scorer": _custom_scorer_json(),
            "trace_ids": trace_ids,
            "log_assessments": log_assessments,
            "scorer_version": scorer_version,
        },
        input_dir=tmp_path,
        container_input_dir=str(tmp_path),
    )


def test_docker_plan_cannot_override_or_spoof_existing_feedback(sqlite_tracking, tmp_path):
    _, trace_ids = sqlite_tracking
    human = mlflow.log_feedback(
        trace_id=trace_ids[0],
        name="quality",
        value="good",
        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="reviewer"),
    )
    plan = _docker_plan(sqlite_tracking, tmp_path, scorer_version=3)
    forged = {
        "assessment_name": "quality",
        "assessment_id": "a-forged",
        "source": {"source_type": "HUMAN", "source_id": "reviewer"},
        "feedback": {"value": "bad"},
        "overrides": human.assessment_id,
        "valid": False,
        "metadata": {
            AssessmentMetadataKey.SOURCE_RUN_ID: "some-run",
            AssessmentMetadataKey.SCORER_NAME: "spoofed",
            AssessmentMetadataKey.SCORER_VERSION: "99",
            "user_key": "kept",
        },
    }

    final = plan.finalize({trace_ids[0]: {"assessments": [forged], "failures": []}})

    assessments = {a.assessment_id: a for a in mlflow.get_trace(trace_ids[0]).info.assessments}
    assert assessments[human.assessment_id].valid is not False
    assert "a-forged" not in assessments
    (logged,) = [a for a in assessments.values() if a.assessment_id != human.assessment_id]
    assert logged.source.source_type == AssessmentSourceType.CODE
    assert logged.overrides is None
    # Reserved metadata comes from the host: the submitted scorer and its registered version.
    assert logged.metadata == {
        AssessmentMetadataKey.SCORER_NAME: "output_length",
        AssessmentMetadataKey.SCORER_VERSION: "3",
        "user_key": "kept",
    }
    assert final[trace_ids[0]]["assessments"][0]["source"]["source_type"] == "CODE"


@pytest.mark.parametrize("log_assessments", [True, False])
def test_docker_plan_logs_nothing_when_any_entry_is_malformed(
    sqlite_tracking, tmp_path, log_assessments
):
    _, trace_ids = sqlite_tracking
    plan = _docker_plan(sqlite_tracking, tmp_path, log_assessments=log_assessments)
    valid = _run_container_step(plan)[trace_ids[0]]
    malformed = {"assessments": [{"bogus": 1}], "failures": []}

    with pytest.raises(MlflowException, match="malformed assessment"):
        plan.finalize({trace_ids[0]: valid, trace_ids[1]: malformed})
    assert all(not mlflow.get_trace(trace_id).info.assessments for trace_id in trace_ids)


@pytest.mark.parametrize(
    "trace_result",
    [
        "not a dict",
        {"assessments": "not a list"},
        {"failures": [{"error_code": 1, "error_message": "x"}]},
        {"assessments": [{"assessment_name": "a", "feedback": {"value": 1}, "rationale": 5}]},
        {"assessments": [{"assessment_name": "a", "feedback": {"error": {"error_code": 1}}}]},
    ],
)
def test_docker_plan_rejects_malformed_trace_results(sqlite_tracking, tmp_path, trace_result):
    _, trace_ids = sqlite_tracking
    plan = _docker_plan(sqlite_tracking, tmp_path)

    with pytest.raises(MlflowException, match="malformed"):
        plan.finalize({trace_ids[0]: trace_result})


def test_docker_plan_session_scorer_keeps_trusted_session_marker(sqlite_tracking, tmp_path):
    experiment_id, _ = sqlite_tracking

    @scorer
    def turns(session) -> int:
        return len(session)

    session_ids = []
    for turn in range(2):
        with mlflow.start_span(name=f"turn_{turn}") as span:
            span.set_inputs({"turn": turn})
            span.set_outputs({"reply": turn})
            mlflow.update_current_trace(metadata={TraceMetadataKey.TRACE_SESSION: "session-1"})
        session_ids.append(span.trace_id)
    plan = _plan_invoke_scorer_job_for_docker(
        params={
            "experiment_id": experiment_id,
            "serialized_scorer": json.dumps(turns.model_dump()),
            "trace_ids": session_ids,
        },
        input_dir=tmp_path,
        container_input_dir=str(tmp_path),
    )
    container_output = _run_container_step(plan)
    # The container's own copy of the session marker is not trusted.
    for result in container_output.values():
        for assessment in result["assessments"]:
            assessment["metadata"][TraceMetadataKey.TRACE_SESSION] = "forged-session"

    plan.finalize(container_output)

    (logged,) = mlflow.get_trace(session_ids[0]).info.assessments
    assert logged.value == 2
    assert logged.metadata[TraceMetadataKey.TRACE_SESSION] == "session-1"


def test_docker_plan_logs_assessments_in_the_job_workspace(db_uri, tmp_path, monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    monkeypatch.setenv(BACKEND_STORE_URI_ENV_VAR, db_uri)
    monkeypatch.setenv("MLFLOW_SERVER_ENABLE_CUSTOM_SCORERS", "true")
    mlflow.set_tracking_uri(db_uri)
    store = _get_store()
    with WorkspaceContext("team-a"):
        experiment_id = mlflow.set_experiment("docker-plan-team-a").experiment_id
        with mlflow.start_span(name="span") as span:
            span.set_inputs({"question": "q"})
            span.set_outputs({"answer": "a"})
        trace_id = span.trace_id

    def plan_and_finalize():
        plan = _plan_invoke_scorer_job_for_docker(
            params={
                "experiment_id": experiment_id,
                "serialized_scorer": _custom_scorer_json(),
                "trace_ids": [trace_id],
            },
            input_dir=tmp_path,
            container_input_dir=str(tmp_path),
        )
        return plan.finalize(_run_container_step(plan))

    with mock.patch("mlflow.genai.scorers.job._get_tracking_store", return_value=store):
        with WorkspaceContext("team-a"):
            plan_and_finalize()
            (logged,) = store.get_trace_info(trace_id).assessments
            assert logged.name == "output_length"
        # Another workspace cannot read the job's traces, so its job fails before scoring.
        with WorkspaceContext("team-b"):
            with pytest.raises(MlflowException, match=trace_id):
                plan_and_finalize()
