import json
import mock

import pytest

import mlflow
from mlflow.entities.run_status import RunStatus
from mlflow.entities.source_type import SourceType

from tests.projects.utils import validate_exit_status, GIT_PROJECT_URI
from tests.projects.utils import tracking_uri_mock  # pylint: disable=unused-import


@pytest.fixture()
def runs_cancel_mock():
    """Mocks the Jobs Runs Cancel API request"""
    with mock.patch("mlflow.projects.databricks._jobs_runs_cancel") as runs_cancel_mock:
        runs_cancel_mock.return_value = None
        yield runs_cancel_mock


@pytest.fixture()
def runs_submit_mock():
    """Mocks the Jobs Runs Submit API request"""
    with mock.patch("mlflow.projects.databricks._jobs_runs_submit") as runs_submit_mock:
        runs_submit_mock.return_value = {"run_id": "-1"}
        yield runs_submit_mock


@pytest.fixture()
def runs_get_mock():
    """Mocks the Jobs Runs Get API request"""
    with mock.patch("mlflow.projects.databricks._jobs_runs_get") as runs_get_mock:
        yield runs_get_mock


@pytest.fixture()
def cluster_spec_mock(tmpdir):
    cluster_spec_handle = tmpdir.join("cluster_spec.json")
    cluster_spec_handle.write(json.dumps(dict()))
    yield str(cluster_spec_handle)


@pytest.fixture()
def create_databricks_run_mock(tracking_uri_mock):  # pylint: disable=unused-argument
    # Mocks logic for creating an MLflow run against a tracking server to persist the run to a local
    # file store
    def create_run_mock(
            tracking_uri, experiment_id, source_name,  # pylint: disable=unused-argument
            source_version, entry_point_name):
        return mlflow.tracking._create_run(
            experiment_id=experiment_id, source_name=source_name, source_version=source_version,
            entry_point_name=entry_point_name, source_type=SourceType.PROJECT)
    with mock.patch.object(
            mlflow.projects.databricks, "_create_databricks_run",
            new=create_run_mock) as create_db_run_mock:
        yield create_db_run_mock


def _get_mock_run_state(succeeded):
    if succeeded is None:
        return {"life_cycle_state": "RUNNING", "state_message": ""}
    if succeeded:
        run_result_state = "SUCCESS"
    else:
        run_result_state = "FAILED"
    return {"life_cycle_state": "TERMINATED", "state_message": "", "result_state": run_result_state}


def mock_runs_get_result(succeeded):
    run_state = _get_mock_run_state(succeeded)
    return {"state": run_state, "run_page_url": ""}


def run_databricks_project(cluster_spec_path, block=False):
    return mlflow.projects.run(
        uri=GIT_PROJECT_URI, mode="databricks", cluster_spec=cluster_spec_path, block=block)


@pytest.mark.skip(reason="flaky running in travis py2.7")
def test_run_databricks(
        tmpdir, runs_cancel_mock, create_databricks_run_mock,  # pylint: disable=unused-argument
        runs_submit_mock, runs_get_mock, cluster_spec_mock):
    """Test running on Databricks with mocks."""
    # Test that MLflow gets the correct run status when performing a Databricks run
    for run_succeeded, expected_status in [(True, RunStatus.FINISHED), (False, RunStatus.FAILED)]:
        runs_get_mock.return_value = mock_runs_get_result(succeeded=run_succeeded)
        submitted_run = run_databricks_project(cluster_spec_mock)
        submitted_run.wait()
        assert runs_submit_mock.call_count == 1
        runs_submit_mock.reset_mock()
        validate_exit_status(submitted_run.get_status(), expected_status)


@pytest.mark.skip(reason="flaky running in travis py2.7")
def test_run_databricks_cancel(
        tmpdir, create_databricks_run_mock,  # pylint: disable=unused-argument
        runs_submit_mock, runs_cancel_mock,  # pylint: disable=unused-argument
        runs_get_mock, cluster_spec_mock):
    # Test that MLflow properly handles Databricks run cancellation
    runs_get_mock.return_value = mock_runs_get_result(succeeded=None)
    submitted_run = run_databricks_project(cluster_spec_mock)
    submitted_run.cancel()
    validate_exit_status(submitted_run.get_status(), RunStatus.FAILED)
    # Test that we raise an exception when a blocking Databricks run fails
    runs_get_mock.return_value = mock_runs_get_result(succeeded=False)
    with pytest.raises(mlflow.projects.ExecutionException):
        run_databricks_project(cluster_spec_mock, block=True)
