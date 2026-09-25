import pytest

from mlflow.entities._job_status import JobStatus
from mlflow.protos.jobs_pb2 import JobStatus as ProtoJobStatus


@pytest.mark.parametrize(
    ("status", "expected_proto"),
    [
        (JobStatus.PENDING, ProtoJobStatus.JOB_STATUS_PENDING),
        (JobStatus.RUNNING, ProtoJobStatus.JOB_STATUS_IN_PROGRESS),
        (JobStatus.NEEDS_RECOVERY, ProtoJobStatus.JOB_STATUS_NEEDS_RECOVERY),
        (JobStatus.SUCCEEDED, ProtoJobStatus.JOB_STATUS_COMPLETED),
        (JobStatus.FAILED, ProtoJobStatus.JOB_STATUS_FAILED),
        (JobStatus.TIMEOUT, ProtoJobStatus.JOB_STATUS_FAILED),
        (JobStatus.CANCELED, ProtoJobStatus.JOB_STATUS_CANCELED),
    ],
)
def test_job_status_to_proto(status, expected_proto):
    assert status.to_proto() == expected_proto


@pytest.mark.parametrize(
    ("status", "expected_int"),
    [
        (JobStatus.PENDING, 0),
        (JobStatus.RUNNING, 1),
        (JobStatus.SUCCEEDED, 2),
        (JobStatus.FAILED, 3),
        (JobStatus.TIMEOUT, 4),
        (JobStatus.CANCELED, 5),
        (JobStatus.NEEDS_RECOVERY, 6),
    ],
)
def test_job_status_stable_int_mapping(status, expected_int):
    assert status.to_int() == expected_int
    assert JobStatus.from_int(expected_int) == status


def test_needs_recovery_is_not_finalized():
    assert JobStatus.is_finalized(JobStatus.NEEDS_RECOVERY) is False
