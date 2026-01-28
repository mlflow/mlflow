import pytest

from mlflow.entities._job_status import JobStatus
from mlflow.protos.jobs_pb2 import JobStatus as ProtoJobStatus


@pytest.mark.parametrize(
    ("status", "expected_proto"),
    [
        (JobStatus.PENDING, ProtoJobStatus.JOB_STATUS_PENDING),
        (JobStatus.RUNNING, ProtoJobStatus.JOB_STATUS_IN_PROGRESS),
        (JobStatus.SUCCEEDED, ProtoJobStatus.JOB_STATUS_COMPLETED),
        (JobStatus.FAILED, ProtoJobStatus.JOB_STATUS_FAILED),
        (JobStatus.TIMEOUT, ProtoJobStatus.JOB_STATUS_FAILED),
        (JobStatus.CANCELED, ProtoJobStatus.JOB_STATUS_CANCELED),
    ],
)
def test_job_status_to_proto(status, expected_proto):
    assert status.to_proto() == expected_proto
