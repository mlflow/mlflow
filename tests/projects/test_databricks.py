import mock

import pytest

from mlflow.utils.file_utils import TempDir
import mlflow


def test_wait_databricks():
    with TempDir() as tmp, \
            mock.patch("mlflow.projects.databricks._get_run_result_state") as run_state_mock, \
            mock.patch("mlflow.tracking.get_tracking_uri") as get_tracking_uri_mock:
        tmp_dir = tmp.path()
        get_tracking_uri_mock.return_value = tmp_dir
        run_state_mock.return_value = "SUCCESS"
        mlflow.projects.databricks.wait(databricks_run_id=-1, sleep_interval=3)
        run_state_mock.return_value = "FAILURE"
        with pytest.raises(mlflow.projects.ExecutionException):
            mlflow.projects.databricks.wait(databricks_run_id=-1, sleep_interval=3)


def test_run_databricks():
    """Test running on Databricks with mocks."""
    pass
