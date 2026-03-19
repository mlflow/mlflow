"""Job metadata tracking for MLflow jobs using thread-local context."""

import json
import threading
from typing import Any

_thread_local = threading.local()


class JobTracker:
    """Tracks job execution (internal use)."""

    def __init__(self, stage_file: str):
        self.stage_file = stage_file

    def update(self, status_details: dict[str, Any]) -> None:
        with open(self.stage_file, "w") as f:
            json.dump(status_details, f)


class NoOpTracker:
    """No-op tracker used when not running as a job."""

    def update(self, status_details: dict[str, Any]) -> None:
        pass


def _get_job_tracker() -> JobTracker | NoOpTracker:
    return getattr(_thread_local, "job_tracker", NoOpTracker())


def _set_job_tracker(tracker: JobTracker | None) -> None:
    if tracker is None:
        if hasattr(_thread_local, "job_tracker"):
            delattr(_thread_local, "job_tracker")
    else:
        _thread_local.job_tracker = tracker


def update_status_details(status_details: dict[str, Any]) -> None:
    """
    Update the current job execution status details.

    When called from a job, writes status details to file for parent process to read.
    When called outside a job context, does nothing (no-op).
    """
    tracker = _get_job_tracker()
    tracker.update(status_details)
