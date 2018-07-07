import mock

import pytest

from mlflow.utils.file_utils import TempDir
import mlflow


def mock_run_cancel_result():
    return None


def mock_run_status_result():
    pass


@pytest.fixture()
def mock_jobs_methods():
    with mock.patch('databricks_cli.dbfs.api.DbfsService') as DbfsServiceMock:
        DbfsServiceMock.return_value = mock.MagicMock()
        _dbfs_api = api.DbfsApi(None)
        yield _dbfs_api


def test_wait_databricks():
    with TempDir() as tmp, \
            mock.patch("mlflow.projects.databricks._get_run_result_state") as run_state_mock, \
            mock.patch("mlflow.tracking.get_tracking_uri") as get_tracking_uri_mock, \
            mock.patch("mlflow.projects.databricks._maybe_cancel_run"):
        tmp_dir = tmp.path()
        get_tracking_uri_mock.return_value = tmp_dir
        run_state_mock.return_value = "SUCCESS"
        mlflow.projects.databricks.wait_databricks(databricks_run_id=-1, sleep_interval=3)
        run_state_mock.return_value = "FAILURE"
        with pytest.raises(mlflow.projects.ExecutionException):
            mlflow.projects.databricks.wait_databricks(databricks_run_id=-1, sleep_interval=3)


def test_run_databricks():
    """Test running on Databricks with mocks."""
    # Sequence of API calls: Runs submit, runs get, [runs get...polling] runs cancel
    # Test:
    # * Runs submit fails (or test that in rest_utils?)
    # * Runs get shows failure
    # * Runs get shows success
    # * Runs get shows nothing, then failure
    #
