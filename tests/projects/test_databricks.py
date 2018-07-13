import filecmp
import json
import mock
import os
import shutil
import subprocess

from databricks_cli.configure import provider
import pytest

import mlflow
from mlflow.entities.run_status import RunStatus
from mlflow.entities.source_type import SourceType
from mlflow.projects import databricks, ExecutionException
from mlflow.utils import file_utils

from tests.projects.utils import validate_exit_status, GIT_PROJECT_URI
from tests.projects.utils import tracking_uri_mock  # pylint: disable=unused-import
from tests.projects.utils import TEST_PROJECT_DIR


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
    with mock.patch("mlflow.projects.databricks._create_databricks_run") as create_db_run_mock:
        create_db_run_mock.return_value = mlflow.tracking._create_run(
            experiment_id=0, source_name="", source_version="", entry_point_name="",
            source_type=SourceType.PROJECT)
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

    # Test that MLflow properly handles Databricks run cancellation
    runs_get_mock.return_value = mock_runs_get_result(succeeded=None)
    submitted_run = run_databricks_project(cluster_spec_mock)
    submitted_run.cancel()
    validate_exit_status(submitted_run.get_status(), RunStatus.FAILED)
    # # Test that we raise an exception when a blocking Databricks run fails
    runs_get_mock.return_value = mock_runs_get_result(succeeded=False)
    with pytest.raises(mlflow.projects.ExecutionException):
        run_databricks_project(cluster_spec_mock, block=True)


@pytest.fixture()
def dbfs_root_mock(tmpdir):
    yield str(tmpdir.join("dbfs-root"))


@pytest.fixture()
def upload_to_dbfs_mock(dbfs_root_mock):
    def upload_mock_fn(src_path, dbfs_uri, _):
        mock_dbfs_dst = os.path.join(dbfs_root_mock, dbfs_uri.split("dbfs:/")[1])
        shutil.copy(src_path, mock_dbfs_dst)

    with mock.patch.object(
            mlflow.projects.databricks, mlflow.projects.databricks._upload_to_dbfs,
            new=upload_mock_fn) as upload_mock:
        yield upload_mock
    # with mock.patch(mlflow.projects.databricks._upload_to_dbfs) as upload_mock:
    #     yield upload_mock


@pytest.fixture()
def dbfs_path_exists_mock(dbfs_root_mock):
    with mock.patch("mlflow.projects.databricks._dbfs_path_exists") as path_exists_mock:
        yield path_exists_mock


def test_upload_project_to_dbfs(tmpdir,
        dbfs_root_mock, upload_to_dbfs_mock,  # pylint: disable=unused-argument
        dbfs_path_exists_mock):  # pylint: disable=unused-argument
    # Upload project to a mock directory
    dbfs_uri = databricks._upload_project_to_dbfs(
        project_dir=TEST_PROJECT_DIR, experiment_id=0, profile=provider.DEFAULT_SECTION)
    # Fetch & extract the tarred project, verify its contents
    local_tar_path = os.path.join(dbfs_root_mock, dbfs_uri.split("dbfs:/")[1])
    expected_tar_path = str(tmpdir.join("expected.tar.gz"))
    expected_tar = file_utils.make_tarfile(
        output_filename=expected_tar_path, source_dir=TEST_PROJECT_DIR,
        archive_name=databricks.DB_TARFILE_ARCHIVE_NAME)
    assert filecmp.cmp(local_tar_path, expected_tar, shallow=False)


# def test_get_databricks_run_command(tmpdir):
#     """Tests that the databricks run command works as expected"""
#     tarpath = str(tmpdir.join("project.tar.gz"))
#     file_utils.make_tarfile(
#         output_filename=tarpath, source_dir=TEST_PROJECT_DIR,
#         archive_name=databricks.DB_TARFILE_ARCHIVE_NAME)
#     cmd = databricks._get_databricks_run_cmd(
#         dbfs_fuse_tar_uri=TEST_PROJECT_DIR, entry_point="greeter", parameters={"name": "friend"})
#     p = subprocess.Popen(cmd)


def test_run_databricks_validations(databricks_api_req_mock):
    """
    Tests that running on Databricks fails before making any API requests if parameters are
    mis-specified or the Databricks CLI is not installed
    """
    with mock.patch("mlflow.utils.rest_utils.databricks_api_request") as db_api_req_mock:
        with pytest.raises(ExecutionException):
            mlflow.projects.run(TEST_PROJECT_DIR)
            assert databricks_api_req_mock.call_count == 0