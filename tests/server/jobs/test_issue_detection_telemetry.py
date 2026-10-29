import concurrent.futures
import json
from unittest import mock

import pytest

from mlflow.entities._job_status import JobStatus
from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.exceptions import MlflowException
from mlflow.store.jobs.abstract_store import JobUpdateStatus
from mlflow.store.jobs.sqlalchemy_store import SqlAlchemyJobStore
from mlflow.store.jobs.sqlalchemy_workspace_store import WorkspaceAwareSqlAlchemyJobStore
from mlflow.telemetry.events import DiscoverIssuesEvent
from mlflow.telemetry.schemas import Status
from mlflow.utils.workspace_context import WorkspaceContext


@pytest.fixture(params=[False, True], ids=["single-tenant", "workspace"])
def job_store(request, db_uri, monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, str(request.param).lower())
    if request.param:
        with WorkspaceContext("telemetry-team"):
            yield WorkspaceAwareSqlAlchemyJobStore(db_uri)
    else:
        yield SqlAlchemyJobStore(db_uri)


@pytest.mark.parametrize("job_name", ["invoke_issue_detection", "scheduled_issue_detection"])
@pytest.mark.parametrize(
    ("transition", "status"),
    [
        ("finish", JobStatus.SUCCEEDED),
        ("fail", JobStatus.FAILED),
        ("timeout", JobStatus.TIMEOUT),
        ("cancel", JobStatus.CANCELED),
        ("report_success", JobStatus.SUCCEEDED),
        ("report_failure", JobStatus.FAILED),
        ("report_timeout", JobStatus.TIMEOUT),
        ("report_cancel", JobStatus.CANCELED),
        ("retry_exhausted", JobStatus.FAILED),
    ],
)
def test_terminal_transition_emits_once_after_commit(
    job_store, monkeypatch, job_name, transition, status
):
    monkeypatch.setenv("MLFLOW_SERVER_JOB_TRANSIENT_ERROR_MAX_RETRIES", "0")
    job = job_store.create_job(job_name, '{"trace_ids": ["trace-1"]}')
    job_store.start_job(job.job_id)

    committed_states = []

    def check_committed(snapshot):
        committed_states.append(job_store.get_job(snapshot.job_id).status)

    with mock.patch(
        "mlflow.store.jobs.sqlalchemy_store.record_issue_detection_job_event",
        side_effect=check_committed,
    ) as record_event:
        match transition:
            case "finish":
                job_store.finish_job(job.job_id, '{"issues": 2}')
            case "fail":
                job_store.fail_job(job.job_id, "failed")
            case "timeout":
                job_store.mark_job_timed_out(job.job_id)
            case "cancel":
                job_store.cancel_job(job.job_id)
            case "report_success":
                job_store.report_job_result(job.job_id, status, result='{"issues": 2}')
            case "report_failure":
                job_store.report_job_result(job.job_id, status, error_message="failed")
            case "report_timeout" | "report_cancel":
                job_store.report_job_result(job.job_id, status)
            case "retry_exhausted":
                assert job_store.retry_or_fail_job(job.job_id, "failed") is None

        record_event.assert_called_once()
        snapshot = record_event.call_args.args[0]
        assert snapshot.job_id == job.job_id
        assert snapshot.status == status
        assert committed_states == [status]
        for late_update in (job_store.cancel_job, job_store.finish_job, job_store.fail_job):
            args = () if late_update == job_store.cancel_job else ("late result",)
            with pytest.raises(MlflowException, match="already finalized"):
                late_update(job.job_id, *args)
        record_event.assert_called_once()


def test_pending_cancel_emits_zero_execution_duration(job_store):
    job = job_store.create_job("invoke_issue_detection", '{"trace_ids": ["trace-1"]}')
    with mock.patch(
        "mlflow.store.jobs.sqlalchemy_store.record_issue_detection_job_event"
    ) as record_event:
        job_store.cancel_job(job.job_id)

    record_event.assert_called_once()
    snapshot = record_event.call_args.args[0]
    assert snapshot.status == JobStatus.CANCELED
    assert snapshot.status_details is None


def test_rolled_back_transition_does_not_emit(job_store):
    job = job_store.create_job("invoke_issue_detection", "{}")
    with mock.patch(
        "mlflow.store.jobs.sqlalchemy_store.record_issue_detection_job_event"
    ) as record_event:

        def update_and_rollback():
            with job_store.ManagedSessionMaker(read_only=False) as session:
                assert (
                    job_store._conditional_status_update(
                        session,
                        job.job_id,
                        (JobStatus.PENDING,),
                        job_store._terminal_update_values(JobStatus.CANCELED, None, 1000),
                    )
                    == JobUpdateStatus.APPLIED
                )
                record_event.assert_not_called()
                raise MlflowException("rollback")

        with pytest.raises(MlflowException, match="rollback"):
            update_and_rollback()
        record_event.assert_not_called()
    assert job_store.get_job(job.job_id).status == JobStatus.PENDING


@pytest.mark.parametrize("reschedule", ["retry_or_fail", "retry", "reset", "requeue"])
def test_start_and_retry_clear_stale_counts_without_emitting(job_store, monkeypatch, reschedule):
    monkeypatch.setenv("MLFLOW_SERVER_JOB_TRANSIENT_ERROR_MAX_RETRIES", "1")
    monkeypatch.setenv("MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_BASE_DELAY", "0")
    job = job_store.create_job("scheduled_issue_detection", "{}")
    with mock.patch(
        "mlflow.store.jobs.sqlalchemy_store.record_issue_detection_job_event"
    ) as record_event:
        with mock.patch(
            "mlflow.store.jobs.sqlalchemy_store.get_current_time_millis", return_value=1000
        ) as clock:
            job_store.start_job(job.job_id)
        clock.assert_called_once()
        assert job_store.get_job(job.job_id).status_details == {"start_time_ms": 1000}
        job_store.update_status_details(
            job.job_id,
            {
                "issues": 3,
                "issue_counts_by_category": {"Custom": 3},
                "trace_count": 10,
                "total_traces_analyzed": 8,
                "total_cost_usd": 0.1,
                "triage_run_id": "triage-run",
                "run_id": "source-run",
            },
        )
        if reschedule == "requeue":
            assert job_store.mark_job_needs_recovery(job.job_id) == JobUpdateStatus.APPLIED
        with mock.patch(
            "mlflow.store.jobs.sqlalchemy_store.get_current_time_millis", return_value=1500
        ) as clock:
            match reschedule:
                case "retry_or_fail":
                    assert job_store.retry_or_fail_job(job.job_id, "transient failure") == 1
                case "retry":
                    assert job_store.retry_job(job.job_id) == 1
                case "reset":
                    job_store.reset_job(job.job_id)
                case "requeue":
                    assert job_store.requeue_job(job.job_id) == JobUpdateStatus.APPLIED
        clock.assert_called_once()
        assert job_store.get_job(job.job_id).status_details == {"execution_duration_ms": 500}
        with mock.patch(
            "mlflow.store.jobs.sqlalchemy_store.get_current_time_millis", return_value=2000
        ) as clock:
            job_store.start_job(job.job_id)
        clock.assert_called_once()
        assert job_store.get_job(job.job_id).status_details == {
            "start_time_ms": 2000,
            "execution_duration_ms": 500,
        }
        record_event.assert_not_called()


def test_other_jobs_do_not_emit_or_receive_telemetry_metadata(job_store):
    job = job_store.create_job("invoke_scorer", "{}")
    with mock.patch(
        "mlflow.store.jobs.sqlalchemy_store.record_issue_detection_job_event"
    ) as record_event:
        job_store.start_job(job.job_id)
        job_store.finish_job(job.job_id, "{}")
        record_event.assert_not_called()
    assert job_store.get_job(job.job_id).status_details is None


def test_telemetry_failure_does_not_fail_terminal_transition(job_store):
    job = job_store.create_job("scheduled_issue_detection", "{}")
    with mock.patch(
        "mlflow.store.jobs.sqlalchemy_store.record_issue_detection_job_event",
        side_effect=RuntimeError("telemetry unavailable"),
    ) as record_event:
        job_store.cancel_job(job.job_id)
    record_event.assert_called_once()
    assert job_store.get_job(job.job_id).status == JobStatus.CANCELED


def test_completion_cancellation_race_emits_one_event(job_store):
    job = job_store.create_job("invoke_issue_detection", "{}")
    job_store.start_job(job.job_id)

    def finalize(cancel):
        try:
            if cancel:
                job_store.cancel_job(job.job_id)
            else:
                job_store.finish_job(job.job_id, json.dumps({"issues": 1}))
        except MlflowException as exc:
            return str(exc)
        return None

    with mock.patch(
        "mlflow.store.jobs.sqlalchemy_store.record_issue_detection_job_event"
    ) as record_event:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="test-job-finalization"
        ) as executor:
            outcomes = list(executor.map(finalize, [False, True]))
        record_event.assert_called_once()
    assert outcomes.count(None) == 1
    assert "already finalized" in next(outcome for outcome in outcomes if outcome is not None)
    assert record_event.call_args.args[0].status == job_store.get_job(job.job_id).status


def test_terminal_event_respects_workspace_isolation(db_uri, monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    store = WorkspaceAwareSqlAlchemyJobStore(db_uri)
    with WorkspaceContext("team-a"):
        job = store.create_job("invoke_issue_detection", "{}")
    with mock.patch(
        "mlflow.store.jobs.sqlalchemy_store.record_issue_detection_job_event"
    ) as record_event:
        with WorkspaceContext("team-b"):
            with pytest.raises(MlflowException, match="not found"):
                store.cancel_job(job.job_id)
            record_event.assert_not_called()
        with WorkspaceContext("team-a"):
            store.cancel_job(job.job_id)
        record_event.assert_called_once()
        assert record_event.call_args.args[0].workspace == "team-a"


def test_committed_job_emits_summary_payload(job_store):
    job = job_store.create_job("invoke_issue_detection", '{"trace_ids": ["t-1", "t-2", "t-3"]}')
    with mock.patch("mlflow.telemetry.issue_detection._record_event") as record_event:
        with mock.patch(
            "mlflow.store.jobs.sqlalchemy_store.get_current_time_millis", return_value=1000
        ) as clock:
            job_store.start_job(job.job_id)
        clock.assert_called_once()
        record_event.assert_not_called()

        with mock.patch(
            "mlflow.store.jobs.sqlalchemy_store.get_current_time_millis", return_value=2500
        ) as clock:
            job_store.finish_job(
                job.job_id,
                json.dumps({
                    "issues": 2,
                    "issue_counts_by_category": {"correctness": 1, "private-category": 1},
                }),
            )
        clock.assert_called_once()

    record_event.assert_called_once_with(
        DiscoverIssuesEvent,
        {
            "job_id": job.job_id,
            "model": None,
            "categories": None,
            "source_run_id": None,
            "triage_run_id": None,
            "total_cost_usd": None,
            "issue_count": 2,
            "issue_counts_by_category": {"correctness": 1, "Custom": 1},
            "trace_count": 3,
            "total_traces_analyzed": None,
            "trigger": "adhoc",
        },
        status=Status.SUCCESS,
        duration_ms=1500,
    )
