import json
from typing import Any

from mlflow.entities._job import Job
from mlflow.entities._job_status import JobStatus
from mlflow.telemetry.events import DiscoverIssuesEvent
from mlflow.telemetry.issue_detection_utils import (
    normalize_issue_categories,
    normalize_issue_category_counts,
)
from mlflow.telemetry.schemas import Status
from mlflow.telemetry.track import _record_event
from mlflow.telemetry.utils import _log_error

ISSUE_DETECTION_JOB_NAMES = frozenset({"invoke_issue_detection", "scheduled_issue_detection"})


def _load_json_dict(value: str | None) -> dict[str, Any]:
    try:
        parsed = json.loads(value) if value else {}
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def record_issue_detection_job_event(job: Job) -> None:
    if job.job_name not in ISSUE_DETECTION_JOB_NAMES or not JobStatus.is_finalized(job.status):
        return

    try:
        params = _load_json_dict(job.params)
        result = _load_json_dict(job.result) if job.status == JobStatus.SUCCEEDED else {}
        details = job.status_details or {}
        scheduled = job.job_name == "scheduled_issue_detection"
        trace_count = result.get("trace_count", details.get("trace_count"))
        if not scheduled and isinstance(trace_ids := params.get("trace_ids"), list):
            trace_count = len(trace_ids)
        total_traces_analyzed = result.get(
            "total_traces_analyzed", details.get("total_traces_analyzed")
        )
        status = {
            JobStatus.SUCCEEDED: Status.SUCCESS,
            JobStatus.FAILED: Status.FAILURE,
            JobStatus.TIMEOUT: Status.FAILURE,
            JobStatus.CANCELED: Status.CANCELED,
        }[job.status]
        start_time_ms = details.get("start_time_ms")
        duration_ms = details.get("execution_duration_ms", 0)
        if start_time_ms is not None:
            duration_ms += max(job.last_update_time - start_time_ms, 0)
        issue_count = result.get("issues", details.get("issues"))
        _record_event(
            DiscoverIssuesEvent,
            {
                "job_id": job.job_id,
                "model": params.get("model"),
                "categories": normalize_issue_categories(params.get("categories")),
                "issue_count": issue_count if type(issue_count) is int else None,
                "issue_counts_by_category": normalize_issue_category_counts(
                    result.get("issue_counts_by_category", details.get("issue_counts_by_category"))
                ),
                "trace_count": trace_count if type(trace_count) is int else None,
                "total_traces_analyzed": (
                    total_traces_analyzed if type(total_traces_analyzed) is int else None
                ),
                "trigger": "schedule" if scheduled else "adhoc",
                "total_cost_usd": result.get("total_cost_usd", details.get("total_cost_usd")),
            },
            status=status,
            duration_ms=duration_ms,
        )
    except Exception:
        _log_error("Failed to record issue detection job telemetry")
