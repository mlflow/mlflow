import json
from unittest import mock

import pytest

from mlflow.entities._job_status import JobStatus
from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.exceptions import MlflowException
from mlflow.server import app
from mlflow.server.jobs import cancel_job
from mlflow.store.jobs.sqlalchemy_store import SqlAlchemyJobStore
from mlflow.store.jobs.sqlalchemy_workspace_store import WorkspaceAwareSqlAlchemyJobStore
from mlflow.telemetry.events import DiscoverIssuesEvent
from mlflow.utils.workspace_context import WorkspaceContext


@pytest.fixture(params=[False, True], ids=["single-tenant", "workspace"])
def job_store(request, db_uri, monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, str(request.param).lower())
    if request.param:
        with WorkspaceContext("telemetry-team"):
            yield WorkspaceAwareSqlAlchemyJobStore(db_uri)
    else:
        yield SqlAlchemyJobStore(db_uri)


def _records(mock_requests):
    return [
        record["data"]
        for record in mock_requests
        if record["data"]["event_name"] == DiscoverIssuesEvent.name
    ]


@pytest.mark.parametrize(
    ("job_name", "trigger", "trace_count"),
    [("invoke_issue_detection", "adhoc", 2), ("scheduled_issue_detection", "schedule", None)],
)
@pytest.mark.parametrize("running", [False, True])
def test_cancel_issue_detection_records_after_commit(
    job_store, job_name, trigger, trace_count, running, mock_requests, mock_telemetry_client
):
    params = {
        "model": "openai:/gpt-4",
        "categories": [" Correctness ", "private category", "another private"],
        "run_id": "private-run",
        "instructions": "private instructions",
    }
    if trigger == "adhoc":
        params["trace_ids"] = ["t-1", "t-2"]
    job = job_store.create_job(job_name, json.dumps(params))
    if running:
        job_store.start_job(job.job_id)

    with (
        mock.patch("mlflow.server.jobs._get_job_store", return_value=job_store) as get_store,
        mock.patch(
            "mlflow.telemetry.track.get_telemetry_client", return_value=mock_telemetry_client
        ) as get_client,
    ):
        canceled_job = cancel_job(job.job_id)
        assert canceled_job.status == JobStatus.CANCELED
        assert job_store.get_job(job.job_id).status == JobStatus.CANCELED
        with pytest.raises(MlflowException, match="already finalized"):
            cancel_job(job.job_id)
    assert get_store.call_count == 2
    get_client.assert_called()

    mock_telemetry_client.flush()
    records = _records(mock_requests)
    assert len(records) == 1
    assert records[0]["status"] == "canceled"
    assert records[0]["duration_ms"] is None
    assert json.loads(records[0]["params"]) == {
        "job_id": job.job_id,
        "trigger": trigger,
        "model": "openai:/gpt-4",
        "categories": ["correctness", "Custom"],
        "trace_count": trace_count,
        "issue_count": None,
        "issue_counts_by_category": None,
        "total_traces_analyzed": None,
        "total_cost_usd": None,
    }


def test_cancel_detection_endpoint_records_in_server(db_uri, mock_requests, mock_telemetry_client):
    store = SqlAlchemyJobStore(db_uri)
    job = store.create_job("invoke_issue_detection", '{"trace_ids": ["t-1"]}')
    store.start_job(job.job_id)
    with (
        mock.patch("mlflow.server.jobs._get_job_store", return_value=store) as get_store,
        mock.patch(
            "mlflow.telemetry.track.get_telemetry_client", return_value=mock_telemetry_client
        ) as get_client,
        app.test_client() as client,
    ):
        response = client.patch(f"/ajax-api/3.0/mlflow/jobs/cancel/{job.job_id}")
    get_store.assert_called_once_with()
    get_client.assert_called()
    assert response.status_code == 200
    assert response.get_json()["status"] == "CANCELED"
    mock_telemetry_client.flush()
    assert len(records := _records(mock_requests)) == 1
    assert records[0]["status"] == "canceled"


@pytest.mark.parametrize("case", ["missing", "finished", "other-job"])
def test_rejected_and_unrelated_cancellations_do_not_emit(job_store, case):
    job = job_store.create_job(
        "invoke_scorer" if case == "other-job" else "invoke_issue_detection", "{}"
    )
    if case == "finished":
        job_store.start_job(job.job_id)
        job_store.finish_job(job.job_id, "{}")
    with (
        mock.patch("mlflow.server.jobs._get_job_store", return_value=job_store) as get_store,
        mock.patch("mlflow.telemetry.issue_detection._record_event") as record_event,
    ):
        if case == "other-job":
            assert cancel_job(job.job_id).status == JobStatus.CANCELED
        else:
            with pytest.raises(
                MlflowException, match="not found" if case == "missing" else "already finalized"
            ):
                cancel_job("nonexistent" if case == "missing" else job.job_id)
    get_store.assert_called_once_with()
    record_event.assert_not_called()


def test_telemetry_failure_does_not_break_cancellation(job_store):
    job = job_store.create_job("invoke_issue_detection", "{}")
    with (
        mock.patch("mlflow.server.jobs._get_job_store", return_value=job_store) as get_store,
        mock.patch(
            "mlflow.telemetry.issue_detection._record_event",
            side_effect=RuntimeError("unavailable"),
        ) as record_event,
    ):
        assert cancel_job(job.job_id).status == JobStatus.CANCELED
    get_store.assert_called_once_with()
    record_event.assert_called_once()


def test_cancel_issue_detection_respects_workspace_isolation(db_uri, monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    store = WorkspaceAwareSqlAlchemyJobStore(db_uri)
    with WorkspaceContext("team-a"):
        job = store.create_job("invoke_issue_detection", "{}")
    with (
        WorkspaceContext("team-b"),
        mock.patch("mlflow.server.jobs._get_job_store", return_value=store) as get_store,
        mock.patch("mlflow.telemetry.issue_detection._record_event") as record_event,
    ):
        with pytest.raises(MlflowException, match="not found"):
            cancel_job(job.job_id)
    get_store.assert_called_once_with()
    record_event.assert_not_called()
