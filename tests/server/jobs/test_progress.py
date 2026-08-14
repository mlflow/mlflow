from unittest import mock

from mlflow.server.jobs.progress import (
    JobTracker,
    NoOpTracker,
    _get_job_tracker,
    _set_job_tracker,
    update_status_details,
)


def test_job_tracker_writes_to_database():
    job_id = "test-job-123"
    tracker = JobTracker(job_id)

    with mock.patch("mlflow.server.handlers._get_job_store") as mock_get_store:
        mock_store = mock.Mock()
        mock_get_store.return_value = mock_store

        tracker.update({"stage": "preprocessing"})

        mock_get_store.assert_called_once()
        mock_store.update_status_details.assert_called_once_with(job_id, {"stage": "preprocessing"})


def test_job_tracker_writes_status_details_with_metadata():
    job_id = "test-job-456"
    tracker = JobTracker(job_id)

    with mock.patch("mlflow.server.handlers._get_job_store") as mock_get_store:
        mock_store = mock.Mock()
        mock_get_store.return_value = mock_store

        tracker.update({"stage": "processing", "progress": "50%", "step": "1"})

        mock_store.update_status_details.assert_called_once_with(
            job_id, {"stage": "processing", "progress": "50%", "step": "1"}
        )


def test_job_tracker_multiple_updates():
    job_id = "test-job-789"
    tracker = JobTracker(job_id)

    with mock.patch("mlflow.server.handlers._get_job_store") as mock_get_store:
        mock_store = mock.Mock()
        mock_get_store.return_value = mock_store

        tracker.update({"stage": "stage1", "key": "value1"})
        tracker.update({"stage": "stage2", "key": "value2"})

        assert mock_store.update_status_details.call_count == 2
        mock_store.update_status_details.assert_any_call(
            job_id, {"stage": "stage1", "key": "value1"}
        )
        mock_store.update_status_details.assert_any_call(
            job_id, {"stage": "stage2", "key": "value2"}
        )


def test_noop_tracker_does_nothing():
    tracker = NoOpTracker()

    tracker.update({"stage": "stage1"})
    tracker.update({"stage": "stage2", "key": "value"})


def test_get_job_tracker_returns_noop_by_default():
    _set_job_tracker(None)

    tracker = _get_job_tracker()
    assert isinstance(tracker, NoOpTracker)


def test_set_and_get_job_tracker():
    job_id = "test-job-abc"
    job_tracker = JobTracker(job_id)

    _set_job_tracker(job_tracker)
    tracker = _get_job_tracker()
    assert tracker is job_tracker

    _set_job_tracker(None)
    tracker = _get_job_tracker()
    assert isinstance(tracker, NoOpTracker)


def test_job_tracker_is_global():
    job_id = "test-job-global"
    tracker = JobTracker(job_id)

    _set_job_tracker(tracker)

    assert _get_job_tracker() is tracker

    _set_job_tracker(None)


def test_update_status_details_uses_active_tracker():
    job_id = "test-job-xyz"
    tracker = JobTracker(job_id)

    with mock.patch("mlflow.server.handlers._get_job_store") as mock_get_store:
        mock_store = mock.Mock()
        mock_get_store.return_value = mock_store

        _set_job_tracker(tracker)
        update_status_details({"stage": "my_stage", "info": "test"})

        mock_store.update_status_details.assert_called_once_with(
            job_id, {"stage": "my_stage", "info": "test"}
        )

    _set_job_tracker(None)


def test_update_status_details_is_noop_without_tracker():
    _set_job_tracker(None)

    update_status_details({"stage": "stage1"})
    update_status_details({"stage": "stage2", "key": "value"})


def test_update_status_details_with_only_stage():
    job_id = "test-job-stage"
    tracker = JobTracker(job_id)

    with mock.patch("mlflow.server.handlers._get_job_store") as mock_get_store:
        mock_store = mock.Mock()
        mock_get_store.return_value = mock_store

        _set_job_tracker(tracker)
        update_status_details({"stage": "my_stage"})

        mock_store.update_status_details.assert_called_once_with(job_id, {"stage": "my_stage"})

    _set_job_tracker(None)


def test_update_status_details_with_additional_fields():
    job_id = "test-job-fields"
    tracker = JobTracker(job_id)

    with mock.patch("mlflow.server.handlers._get_job_store") as mock_get_store:
        mock_store = mock.Mock()
        mock_get_store.return_value = mock_store

        _set_job_tracker(tracker)
        update_status_details({"stage": "my_stage", "progress": "50%", "details": "processing"})

        mock_store.update_status_details.assert_called_once_with(
            job_id, {"stage": "my_stage", "progress": "50%", "details": "processing"}
        )

    _set_job_tracker(None)
