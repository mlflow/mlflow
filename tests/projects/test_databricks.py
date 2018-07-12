import json
import mock

import pytest

from mlflow.entities.run_status import RunStatus
from tests.projects.utils import validate_exit_status, tracking_uri_mock, GIT_PROJECT_URI
import mlflow


@pytest.fixture()
def runs_cancel_mock():
    with mock.patch("mlflow.projects.databricks._jobs_runs_cancel") as runs_cancel_mock:
        runs_cancel_mock.return_value = None
        yield runs_cancel_mock


@pytest.fixture()
def runs_submit_mock():
    with mock.patch("mlflow.projects.databricks._jobs_runs_submit") as runs_submit_mock:
        runs_submit_mock.return_value = {"run_id": "-1"}
        yield runs_submit_mock


@pytest.fixture()
def runs_get_mock():
    with mock.patch("mlflow.projects.databricks._jobs_runs_get") as runs_get_mock:
        yield runs_get_mock


@pytest.fixture()
def cluster_spec_mock(tmpdir):
    cluster_spec_handle = tmpdir.join("cluster_spec.json")
    cluster_spec_handle.write(json.dumps(dict()))
    yield str(cluster_spec_handle)


def mock_run_state(succeeded):
    if succeeded is None:
        return {"life_cycle_state": "RUNNING", "state_message": ""}
    if succeeded:
        run_result_state = "SUCCESS"
    else:
        run_result_state = "FAILED"
    return {"life_cycle_state": "TERMINATED", "state_message": "", "result_state": run_result_state}


def mock_runs_get_result(succeeded):
    run_state = mock_run_state(succeeded)
    return {"state": run_state, "run_page_url": ""}


def run_databricks_project(cluster_spec_path, block=False):
    return mlflow.projects.run(
        uri=GIT_PROJECT_URI, mode="databricks", cluster_spec=cluster_spec_path, block=block)


def test_run_databricks(tmpdir, runs_cancel_mock, runs_submit_mock, runs_get_mock, tracking_uri_mock, cluster_spec_mock):
    """Test running on Databricks with mocks."""
    assert tmpdir == tracking_uri_mock.return_value
    # Test that MLflow gets the correct run status when performing a Databricks run
    for run_api_result, expected_status in [(True, RunStatus.FINISHED), (False, RunStatus.FAILED)]:
        runs_get_mock.return_value = mock_runs_get_result(run_api_result)
        submitted_run = run_databricks_project(cluster_spec_mock)
        submitted_run.wait()
        assert runs_submit_mock.call_count == 1
        runs_submit_mock.reset_mock()
        # TODO: it's difficult to check run status right now, since we expect it to be set
        # by user code during the Job run.
        # assert validate_exit_status(submitted_run.get_status(), expected_status)

    # Test that MLflow properly handles Databricks run cancellation
    runs_get_mock.return_value = mock_runs_get_result(succeeded=None)
    submitted_run = run_databricks_project(cluster_spec_mock)
    import time
    time.sleep(1) # Need to sleep to provide monitoring process enough time to launch
    submitted_run.cancel()
    # assert validate_exit_status(submitted_run.get_status(), RunStatus.FAILED)
    # Test that we raise an exception when a blocking Databricks run fails
    runs_get_mock.return_value = mock_runs_get_result(succeeded=False)
    with pytest.raises(mlflow.projects.ExecutionException):
        run_databricks_project(cluster_spec_mock, block=True)
