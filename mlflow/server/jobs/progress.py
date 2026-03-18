"""Job metadata tracking for MLflow jobs using thread-local context."""

import json
import threading

_thread_local = threading.local()


class JobTracker:
    """Tracks job execution by writing metadata to a file (internal use)."""

    def __init__(self, stage_file: str):
        self.stage_file = stage_file

    def update(self, metadata: dict[str, str]) -> None:
        with open(self.stage_file, "w") as f:
            json.dump(metadata, f)


class NoOpTracker:
    """No-op tracker used when not running as a job."""

    def update(self, metadata: dict[str, str]) -> None:
        del metadata  # Unused in no-op tracker


def _get_job_tracker() -> JobTracker | NoOpTracker:
    return getattr(_thread_local, "job_tracker", NoOpTracker())


def _set_job_tracker(tracker: JobTracker | None) -> None:
    if tracker is None:
        if hasattr(_thread_local, "job_tracker"):
            delattr(_thread_local, "job_tracker")
    else:
        _thread_local.job_tracker = tracker


def update_metadata(metadata: dict[str, str]) -> None:
    """
    Update the current job execution metadata.

    When called from a job, writes metadata to file for parent process to read.
    When called outside a job context, does nothing (no-op).
    """
    tracker = _get_job_tracker()
    tracker.update(metadata)
