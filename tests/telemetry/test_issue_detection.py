import json
from unittest import mock

import pytest

from mlflow.entities._job import Job
from mlflow.entities._job_status import JobStatus
from mlflow.telemetry.events import DiscoverIssuesEvent
from mlflow.telemetry.issue_detection import record_issue_detection_job_event
from mlflow.telemetry.schemas import Status


def _make_job(**kwargs):
    return Job(
        **{
            "job_id": "test-job-id",
            "creation_time": 1000,
            "job_name": "invoke_issue_detection",
            "params": json.dumps({
                "trace_ids": ["trace-1", "trace-2"],
                "run_id": "adhoc-run",
                "instructions": "private",
            }),
            "timeout": None,
            "status": JobStatus.SUCCEEDED,
            "result": None,
            "retry_count": 0,
            "last_update_time": 5000,
            "status_details": {"start_time_ms": 2000},
        }
        | kwargs
    )


@pytest.mark.parametrize(
    ("job_status", "event_status"),
    [
        (JobStatus.SUCCEEDED, Status.SUCCESS),
        (JobStatus.FAILED, Status.FAILURE),
        (JobStatus.TIMEOUT, Status.FAILURE),
        (JobStatus.CANCELED, Status.CANCELED),
    ],
)
@pytest.mark.parametrize(
    ("job_name", "trigger"),
    [("invoke_issue_detection", "adhoc"), ("scheduled_issue_detection", "schedule")],
)
def test_record_terminal_issue_detection_job(job_status, event_status, job_name, trigger):
    job = _make_job(
        job_name=job_name,
        status=job_status,
        params=json.dumps({
            "trace_ids": ["trace-1", "trace-2"],
            "run_id": "adhoc-run",
            "model": "openai:/gpt-5",
            "categories": ["correctness", "secret category", "another secret"],
            "instructions": "private",
        }),
        result=json.dumps({
            "issues": 3,
            "trace_count": 4,
            "total_traces_analyzed": 3,
            "total_cost_usd": 0.12,
            "triage_run_id": "result-triage",
            "run_id": "result-run",
            "summary": "private summary",
        }),
        status_details={
            "start_time_ms": 2000,
            "issues": 1,
            "trace_count": 5,
            "total_traces_analyzed": 2,
            "total_cost_usd": 0.06,
            "triage_run_id": "detail-triage",
            "run_id": "detail-run",
            "issue_counts_by_category": {"correctness": 1, "private category": 1},
        },
    )
    with mock.patch("mlflow.telemetry.issue_detection._record_event") as mock_record:
        record_issue_detection_job_event(job)

    expected_input = 2 if trigger == "adhoc" else 4 if job_status == JobStatus.SUCCEEDED else 5
    mock_record.assert_called_once_with(
        DiscoverIssuesEvent,
        {
            "job_id": "test-job-id",
            "model": "openai:/gpt-5",
            "categories": ["correctness", "Custom"],
            "issue_count": 3 if job_status == JobStatus.SUCCEEDED else 1,
            "issue_counts_by_category": {"correctness": 1, "Custom": 1},
            "trace_count": expected_input,
            "total_traces_analyzed": 3 if job_status == JobStatus.SUCCEEDED else 2,
            "trigger": trigger,
            "source_run_id": (
                "adhoc-run"
                if trigger == "adhoc"
                else "result-run"
                if job_status == JobStatus.SUCCEEDED
                else "detail-run"
            ),
            "triage_run_id": "result-triage"
            if job_status == JobStatus.SUCCEEDED
            else "detail-triage",
            "total_cost_usd": 0.12 if job_status == JobStatus.SUCCEEDED else 0.06,
        },
        status=event_status,
        duration_ms=3000,
    )


def test_result_category_counts_override_progress_and_bound_custom_names():
    job = _make_job(
        result=json.dumps({
            "issues": 4,
            "issue_counts_by_category": {"safety": 1, "secret category": 2, "another secret": 1},
        }),
        status_details={"issue_counts_by_category": {"correctness": 10}},
    )
    with mock.patch("mlflow.telemetry.issue_detection._record_event") as mock_record:
        record_issue_detection_job_event(job)

    mock_record.assert_called_once()
    assert mock_record.call_args.args[1]["issue_counts_by_category"] == {"safety": 1, "Custom": 3}


@pytest.mark.parametrize("result", [None, "invalid json", "[]"])
def test_unknown_findings_remain_unknown(result):
    with mock.patch("mlflow.telemetry.issue_detection._record_event") as mock_record:
        record_issue_detection_job_event(
            _make_job(job_name="scheduled_issue_detection", result=result, status_details=None)
        )

    mock_record.assert_called_once_with(
        DiscoverIssuesEvent,
        {
            "job_id": "test-job-id",
            "model": None,
            "categories": None,
            "issue_count": None,
            "issue_counts_by_category": None,
            "trace_count": None,
            "total_traces_analyzed": None,
            "trigger": "schedule",
            "source_run_id": None,
            "triage_run_id": None,
            "total_cost_usd": None,
        },
        status=Status.SUCCESS,
        duration_ms=0,
    )


def test_quiet_schedule_records_zero_counts():
    job = _make_job(
        job_name="scheduled_issue_detection",
        result=json.dumps({
            "issues": 0,
            "issue_counts_by_category": {},
            "trace_count": 0,
            "total_traces_analyzed": 0,
            "total_cost_usd": 0.0,
            "run_id": "quiet-run",
        }),
    )
    with mock.patch("mlflow.telemetry.issue_detection._record_event") as mock_record:
        record_issue_detection_job_event(job)

    mock_record.assert_called_once()
    assert mock_record.call_args.args[1] == {
        "job_id": "test-job-id",
        "model": None,
        "categories": None,
        "issue_count": 0,
        "issue_counts_by_category": {},
        "trace_count": 0,
        "total_traces_analyzed": 0,
        "trigger": "schedule",
        "source_run_id": "quiet-run",
        "triage_run_id": None,
        "total_cost_usd": 0.0,
    }


def test_selected_trace_count_is_distinct_from_analyzed_trace_count():
    job = _make_job(result=json.dumps({"trace_count": 1, "total_traces_analyzed": 1}))
    with mock.patch("mlflow.telemetry.issue_detection._record_event") as mock_record:
        record_issue_detection_job_event(job)

    mock_record.assert_called_once()
    assert mock_record.call_args.args[1]["trace_count"] == 2
    assert mock_record.call_args.args[1]["total_traces_analyzed"] == 1


def test_unavailable_adhoc_counts_remain_unknown():
    with mock.patch("mlflow.telemetry.issue_detection._record_event") as mock_record:
        record_issue_detection_job_event(
            _make_job(params="{}", status=JobStatus.FAILED, status_details=None)
        )

    mock_record.assert_called_once()
    assert mock_record.call_args.args[1]["trace_count"] is None
    assert mock_record.call_args.args[1]["total_traces_analyzed"] is None


@pytest.mark.parametrize(
    ("status_details", "expected_duration"),
    [
        ({"execution_duration_ms": 1500}, 1500),
        ({"execution_duration_ms": 1500, "start_time_ms": 4000}, 2500),
        ({"execution_duration_ms": 1500, "start_time_ms": 6000}, 1500),
        ({"start_time_ms": 0}, 5000),
    ],
)
def test_duration_excludes_queue_and_retry_backoff(status_details, expected_duration):
    with mock.patch("mlflow.telemetry.issue_detection._record_event") as mock_record:
        record_issue_detection_job_event(
            _make_job(status=JobStatus.CANCELED, status_details=status_details, retry_count=1)
        )

    mock_record.assert_called_once()
    assert mock_record.call_args.kwargs["duration_ms"] == expected_duration


@pytest.mark.parametrize("status", [JobStatus.PENDING, JobStatus.RUNNING, JobStatus.NEEDS_RECOVERY])
def test_nonterminal_jobs_emit_nothing(status):
    with mock.patch("mlflow.telemetry.issue_detection._record_event") as mock_record:
        record_issue_detection_job_event(_make_job(status=status))

    mock_record.assert_not_called()


def test_other_jobs_emit_nothing():
    with mock.patch("mlflow.telemetry.issue_detection._record_event") as mock_record:
        record_issue_detection_job_event(_make_job(job_name="invoke_scorer"))

    mock_record.assert_not_called()


def test_telemetry_failure_does_not_affect_job_completion():
    with mock.patch(
        "mlflow.telemetry.issue_detection._record_event", side_effect=RuntimeError("unavailable")
    ) as mock_record:
        record_issue_detection_job_event(_make_job())

    mock_record.assert_called_once()
