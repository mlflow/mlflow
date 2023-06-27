import pytest

from mlflow.entities import RunStatus


def test_all_status_covered():
    # ensure that all known status are returned. Test will fail if new status are added to PB
    all_statuses = {
        RunStatus.RUNNING,
        RunStatus.SCHEDULED,
        RunStatus.FINISHED,
        RunStatus.FAILED,
        RunStatus.KILLED,
    }
    assert all_statuses == set(RunStatus.all_status())


def test_status_mappings():
    # test enum to string mappings
    assert RunStatus.to_string(RunStatus.RUNNING) == "RUNNING"
    assert RunStatus.RUNNING == RunStatus.from_string("RUNNING")

    assert RunStatus.to_string(RunStatus.SCHEDULED) == "SCHEDULED"
    assert RunStatus.SCHEDULED == RunStatus.from_string("SCHEDULED")

    assert RunStatus.to_string(RunStatus.FINISHED) == "FINISHED"
    assert RunStatus.FINISHED == RunStatus.from_string("FINISHED")

    assert RunStatus.to_string(RunStatus.FAILED) == "FAILED"
    assert RunStatus.FAILED == RunStatus.from_string("FAILED")

    assert RunStatus.to_string(RunStatus.KILLED) == "KILLED"
    assert RunStatus.KILLED == RunStatus.from_string("KILLED")

    with pytest.raises(Exception, match=r"Could not get string corresponding to run status -120"):
        RunStatus.to_string(-120)

    with pytest.raises(
        Exception, match=r"Could not get run status corresponding to string the IMPO"
    ):
        RunStatus.from_string("the IMPOSSIBLE status string")


def test_is_terminated():
    assert RunStatus.is_terminated(RunStatus.FAILED)
    assert RunStatus.is_terminated(RunStatus.FINISHED)
    assert RunStatus.is_terminated(RunStatus.KILLED)
    assert not RunStatus.is_terminated(RunStatus.SCHEDULED)
    assert not RunStatus.is_terminated(RunStatus.RUNNING)
