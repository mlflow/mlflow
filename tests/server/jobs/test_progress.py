import json
import threading
from pathlib import Path

from mlflow.server.jobs.progress import (
    JobTracker,
    NoOpTracker,
    _get_job_tracker,
    _set_job_tracker,
    update_metadata,
)


def test_job_tracker_writes_stage_to_file(tmp_path: Path):
    stage_file = tmp_path / "stage.json"
    tracker = JobTracker(str(stage_file))

    tracker.update({"stage": "preprocessing"})

    assert stage_file.exists()
    with open(stage_file) as f:
        data = json.load(f)
    assert data == {"stage": "preprocessing"}


def test_job_tracker_writes_stage_with_metadata(tmp_path: Path):
    stage_file = tmp_path / "stage.json"
    tracker = JobTracker(str(stage_file))

    tracker.update({"stage": "processing", "progress": "50%", "step": "1"})

    assert stage_file.exists()
    with open(stage_file) as f:
        data = json.load(f)
    assert data == {"stage": "processing", "progress": "50%", "step": "1"}


def test_job_tracker_overwrites_previous_stage(tmp_path: Path):
    stage_file = tmp_path / "stage.json"
    tracker = JobTracker(str(stage_file))

    tracker.update({"stage": "stage1", "key": "value1"})
    tracker.update({"stage": "stage2", "key": "value2"})

    with open(stage_file) as f:
        data = json.load(f)
    assert data == {"stage": "stage2", "key": "value2"}


def test_noop_tracker_does_nothing():
    tracker = NoOpTracker()

    tracker.update({"stage": "stage1"})
    tracker.update({"stage": "stage2", "key": "value"})


def test_get_job_tracker_returns_noop_by_default():
    _set_job_tracker(None)

    tracker = _get_job_tracker()
    assert isinstance(tracker, NoOpTracker)


def test_set_and_get_job_tracker(tmp_path: Path):
    stage_file = tmp_path / "stage.json"
    job_tracker = JobTracker(str(stage_file))

    _set_job_tracker(job_tracker)
    tracker = _get_job_tracker()
    assert tracker is job_tracker

    _set_job_tracker(None)
    tracker = _get_job_tracker()
    assert isinstance(tracker, NoOpTracker)


def test_job_tracker_is_thread_local(tmp_path: Path):
    stage_file1 = tmp_path / "stage1.json"
    stage_file2 = tmp_path / "stage2.json"

    tracker1 = JobTracker(str(stage_file1))
    tracker2 = JobTracker(str(stage_file2))

    _set_job_tracker(tracker1)

    result = []

    def thread_func():
        _set_job_tracker(tracker2)
        result.append(_get_job_tracker())

    thread = threading.Thread(target=thread_func)
    thread.start()
    thread.join()

    assert _get_job_tracker() is tracker1
    assert result[0] is tracker2

    _set_job_tracker(None)


def test_update_metadata_uses_active_tracker(tmp_path: Path):
    stage_file = tmp_path / "stage.json"
    tracker = JobTracker(str(stage_file))

    _set_job_tracker(tracker)
    update_metadata({"stage": "my_stage", "info": "test"})

    assert stage_file.exists()
    with open(stage_file) as f:
        data = json.load(f)
    assert data == {"stage": "my_stage", "info": "test"}

    _set_job_tracker(None)


def test_update_metadata_is_noop_without_tracker():
    _set_job_tracker(None)

    update_metadata({"stage": "stage1"})
    update_metadata({"stage": "stage2", "key": "value"})


def test_update_metadata_with_only_stage(tmp_path: Path):
    stage_file = tmp_path / "stage.json"
    tracker = JobTracker(str(stage_file))

    _set_job_tracker(tracker)
    update_metadata({"stage": "my_stage"})

    assert stage_file.exists()
    with open(stage_file) as f:
        data = json.load(f)
    assert data == {"stage": "my_stage"}

    _set_job_tracker(None)


def test_update_metadata_with_additional_fields(tmp_path: Path):
    stage_file = tmp_path / "stage.json"
    tracker = JobTracker(str(stage_file))

    _set_job_tracker(tracker)
    update_metadata({"stage": "my_stage", "progress": "50%", "details": "processing"})

    assert stage_file.exists()
    with open(stage_file) as f:
        data = json.load(f)
    assert data == {"stage": "my_stage", "progress": "50%", "details": "processing"}

    _set_job_tracker(None)
