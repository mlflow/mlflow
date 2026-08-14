"""Job tracking for MLflow jobs."""

from typing import Any

_job_tracker: "JobTracker | NoOpTracker | None" = None


class JobTracker:
    """Tracks job execution by writing directly to database (internal use)."""

    def __init__(self, job_id: str):
        self.job_id = job_id

    def update(self, status_details: dict[str, Any]) -> None:
        from mlflow.server.handlers import _get_job_store

        job_store = _get_job_store()
        job_store.update_status_details(self.job_id, status_details)


class NoOpTracker:
    """No-op tracker used when not running as a job."""

    def update(self, status_details: dict[str, Any]) -> None:
        pass


def _get_job_tracker() -> "JobTracker | NoOpTracker":
    return _job_tracker or NoOpTracker()


def _set_job_tracker(tracker: "JobTracker | None") -> None:
    global _job_tracker
    _job_tracker = tracker


def update_status_details(status_details: dict[str, Any]) -> None:
    """
    Update the current job execution status details.

    When called from a job, writes status details to file for parent process to read.
    When called outside a job context, does nothing (no-op).
    """
    tracker = _get_job_tracker()
    tracker.update(status_details)
