import subprocess

from tests.projects.utils import TEST_PROJECT_DIR
from mlflow.projects import databricks
from mlflow.utils import file_utils

def mock_upload_to_dbfs():
    pass


def mock_dbfs_path_exists():
    pass




def test_upload_to_dbfs():
    pass


def test_get_databricks_run_command(tmpdir):
    """Tests that the databricks run command works as expected"""
    tarpath = str(tmpdir.join("project.tar.gz"))
    file_utils.make_tarfile(
        output_filename=tarpath, source_dir=TEST_PROJECT_DIR,
        archive_name=databricks.DB_TARFILE_ARCHIVE_NAME)
    cmd = databricks._get_databricks_run_cmd(
        dbfs_fuse_tar_uri=TEST_PROJECT_DIR, entry_point="greeter", parameters={"name": "friend"})
    p = subprocess.Popen(cmd)



def test_run_databricks_validate_command():
    """
    Tests that run_databricks fails before making any API requests if parameters are mis-specified
    """
    pass