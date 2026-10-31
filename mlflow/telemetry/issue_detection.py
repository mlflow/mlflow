import json

from mlflow.entities._job import Job
from mlflow.entities._job_status import JobStatus
from mlflow.telemetry.events import DiscoverIssuesEvent
from mlflow.telemetry.issue_detection_utils import normalize_issue_categories
from mlflow.telemetry.schemas import Status
from mlflow.telemetry.track import _record_event
from mlflow.telemetry.utils import _log_error


def record_issue_detection_cancellation(job: Job) -> None:
    triggers = {"invoke_issue_detection": "adhoc", "scheduled_issue_detection": "schedule"}
    if job.status != JobStatus.CANCELED or job.job_name not in triggers:
        return

    try:
        arguments = json.loads(job.params)
        trace_ids = arguments.get("trace_ids")
        params = {
            "job_id": job.job_id,
            "trigger": triggers[job.job_name],
            "model": arguments.get("model"),
            "categories": normalize_issue_categories(arguments.get("categories")),
            "trace_count": len(trace_ids) if isinstance(trace_ids, list) else None,
            **DiscoverIssuesEvent.parse_result(None),
        }
        # The worker is killed without returning results or an execution duration.
        _record_event(DiscoverIssuesEvent, params, status=Status.CANCELED, duration_ms=None)
    except Exception:
        _log_error("Failed to record issue detection cancellation telemetry")
