import mock
import subprocess

import pytest

from tests.projects.utils import TEST_PROJECT_DIR
import mlflow
from mlflow.projects import databricks, ExecutionException
from mlflow.utils import file_utils

def mymehtod():
    pass

def upload_to_dbfs_mock():
    pass


def dbfs_path_exists_mock():
    pass


def test_upload_project_to_dbfs(upload_to_dbfs_mock, dbfs_path_exists_mock):
    _upload_project_to_dbfs(project_dir, experiment_id, profile)


@pytext.fixture()
def databricks_api_req_mock():
    with mock.patch("mlflow.utils.rest_utils.databricks_api_request") as db_api_req_mock:
        yield db_api_req_mock


def test_get_databricks_run_command(tmpdir):
    """Tests that the databricks run command works as expected"""
    tarpath = str(tmpdir.join("project.tar.gz"))
    file_utils.make_tarfile(
        output_filename=tarpath, source_dir=TEST_PROJECT_DIR,
        archive_name=databricks.DB_TARFILE_ARCHIVE_NAME)
    cmd = databricks._get_databricks_run_cmd(
        dbfs_fuse_tar_uri=TEST_PROJECT_DIR, entry_point="greeter", parameters={"name": "friend"})
    p = subprocess.Popen(cmd)


def test_run_databricks_validations(databricks_api_req_mock):
    """
    Tests that running on Databricks fails before making any API requests if parameters are
    mis-specified or the Databricks CLI is not installed
    """
    with mock.patch("mlflow.databricks.")
    with pytest.raises(ExecutionException):
        mlflow.projects.run(TEST_PROJECT_DIR)
        assert databricks_api_req_mock.call_count == 0

