import mock

from mlflow.entities import SourceType
from mlflow.utils.mlflow_tags import (
    MLFLOW_SOURCE_NAME,
    MLFLOW_SOURCE_TYPE,
    MLFLOW_DATABRICKS_JOB_ID,
    MLFLOW_DATABRICKS_JOB_RUN_ID,
    MLFLOW_DATABRICKS_JOB_TYPE,
    MLFLOW_DATABRICKS_WEBAPP_URL,
)
from mlflow.tracking.context.databricks_job_context import DatabricksJobRunContext


def test_databricks_job_run_context_in_context():
    with mock.patch("mlflow.utils.databricks_utils.is_in_databricks_job") as in_job_mock:
        assert DatabricksJobRunContext().in_context() == in_job_mock.return_value


def test_databricks_job_run_context_tags():
    patch_job_id = mock.patch("mlflow.utils.databricks_utils.get_job_id")
    patch_job_run_id = mock.patch("mlflow.utils.databricks_utils.get_job_run_id")
    patch_job_type = mock.patch("mlflow.utils.databricks_utils.get_job_type")
    patch_webapp_url = mock.patch("mlflow.utils.databricks_utils.get_webapp_url")

    with patch_job_id as job_id_mock, patch_job_run_id as job_run_id_mock, patch_job_type as job_type_mock, patch_webapp_url as webapp_url_mock:
        assert DatabricksJobRunContext().tags() == {
            MLFLOW_SOURCE_NAME: "jobs/{job_id}/run/{job_run_id}".format(
                job_id=job_id_mock.return_value, job_run_id=job_run_id_mock.return_value
            ),
            MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.JOB),
            MLFLOW_DATABRICKS_JOB_ID: job_id_mock.return_value,
            MLFLOW_DATABRICKS_JOB_RUN_ID: job_run_id_mock.return_value,
            MLFLOW_DATABRICKS_JOB_TYPE: job_type_mock.return_value,
            MLFLOW_DATABRICKS_WEBAPP_URL: webapp_url_mock.return_value,
        }


def test_databricks_job_run_context_tags_nones():
    patch_job_id = mock.patch("mlflow.utils.databricks_utils.get_job_id", return_value=None)
    patch_job_run_id = mock.patch("mlflow.utils.databricks_utils.get_job_run_id", return_value=None)
    patch_job_type = mock.patch("mlflow.utils.databricks_utils.get_job_type", return_value=None)
    patch_webapp_url = mock.patch("mlflow.utils.databricks_utils.get_webapp_url", return_value=None)

    with patch_job_id, patch_job_run_id, patch_job_type, patch_webapp_url:
        assert DatabricksJobRunContext().tags() == {
            MLFLOW_SOURCE_NAME: None,
            MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.JOB),
        }
