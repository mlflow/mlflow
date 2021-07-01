from unittest import mock

from mlflow.utils.mlflow_tags import MLFLOW_DATABRICKS_COMMAND_ID
from mlflow.tracking.context.databricks_command_context import DatabricksCommandRunContext


def test_databricks_command_run_context_in_context():
    with mock.patch("mlflow.utils.databricks_utils.get_job_group_id") as job_group_id_mock:
        assert DatabricksCommandRunContext().in_context() == (
            job_group_id_mock.return_value is not None
        )


def test_databricks_command_run_context_tags():
    with mock.patch("mlflow.utils.databricks_utils.get_job_group_id") as job_group_id_mock:
        assert DatabricksCommandRunContext().tags() == {
            MLFLOW_DATABRICKS_COMMAND_ID: job_group_id_mock.return_value
        }


def test_databricks_command_run_context_tags_nones():
    with mock.patch("mlflow.utils.databricks_utils.get_job_group_id", return_value=None) as _:
        assert DatabricksCommandRunContext().tags() == {}
