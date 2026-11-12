import pytest

from mlflow.entities import IssueDetectionRun, IssueDetectionRunStatus


@pytest.mark.parametrize(
    "run",
    [
        IssueDetectionRun(
            run_id="run-1",
            experiment_id="1",
            status=IssueDetectionRunStatus.FAILED,
            start_time=1700000000000,
            end_time=1700000900000,
            error_message="MlflowException: bad key",
        ),
        IssueDetectionRun(
            run_id="run-1", experiment_id="1", status=IssueDetectionRunStatus.RUNNING
        ),
    ],
    ids=["ended", "running"],
)
def test_issue_detection_run_round_trips(run):
    assert IssueDetectionRun.from_proto(run.to_proto()) == run
    assert IssueDetectionRun.from_dictionary(run.to_dictionary()) == run


def test_issue_detection_run_from_proto_leaves_unset_times_as_none():
    proto = IssueDetectionRun(
        run_id="run-1", experiment_id="1", status=IssueDetectionRunStatus.RUNNING
    ).to_proto()

    run = IssueDetectionRun.from_proto(proto)

    assert run.start_time is None
    assert run.end_time is None
    assert run.error_message is None


@pytest.mark.parametrize(
    ("status", "is_terminal"),
    [
        (IssueDetectionRunStatus.RUNNING, False),
        (IssueDetectionRunStatus.SUCCEEDED, True),
        (IssueDetectionRunStatus.FAILED, True),
        (IssueDetectionRunStatus.CANCELED, True),
    ],
)
def test_issue_detection_run_status_is_terminal(status, is_terminal):
    assert status.is_terminal is is_terminal
    assert str(status) == status.value
