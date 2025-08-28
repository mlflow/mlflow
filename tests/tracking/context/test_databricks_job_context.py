from unittest import mock

from mlflow.entities import SourceType
from mlflow.tracking.context.databricks_job_context import DatabricksJobRunContext
from mlflow.utils.mlflow_tags import (
    MLFLOW_DATABRICKS_JOB_ID,
    MLFLOW_DATABRICKS_JOB_RUN_ID,
    MLFLOW_DATABRICKS_JOB_TYPE,
    MLFLOW_DATABRICKS_WEBAPP_URL,
    MLFLOW_DATABRICKS_WORKSPACE_ID,
    MLFLOW_DATABRICKS_WORKSPACE_URL,
    MLFLOW_SOURCE_NAME,
    MLFLOW_SOURCE_TYPE,
)


def test_databricks_job_run_context_in_context():
    with mock.patch("mlflow.utils.databricks_utils.is_in_databricks_job") as in_job_mock:
        assert DatabricksJobRunContext().in_context() == in_job_mock.return_value


def test_databricks_job_run_context_tags():
    patch_job_id = mock.patch("mlflow.utils.databricks_utils.get_job_id")
    patch_job_run_id = mock.patch("mlflow.utils.databricks_utils.get_job_run_id")
    patch_job_type = mock.patch("mlflow.utils.databricks_utils.get_job_type")
    patch_webapp_url = mock.patch("mlflow.utils.databricks_utils.get_webapp_url")
    patch_workspace_url = mock.patch(
        "mlflow.utils.databricks_utils.get_workspace_url",
        return_value="https://dev.databricks.com",
    )
    patch_workspace_url_none = mock.patch(
        "mlflow.utils.databricks_utils.get_workspace_url", return_value=None
    )
    patch_workspace_info = mock.patch(
        "mlflow.utils.databricks_utils.get_workspace_info_from_dbutils",
        return_value=("https://databricks.com", "123456"),
    )

    with (
        patch_job_id as job_id_mock,
        patch_job_run_id as job_run_id_mock,
        patch_job_type as job_type_mock,
        patch_webapp_url as webapp_url_mock,
        patch_workspace_url as workspace_url_mock,
        patch_workspace_info as workspace_info_mock,
    ):
        assert DatabricksJobRunContext().tags() == {
            MLFLOW_SOURCE_NAME: (
                f"jobs/{job_id_mock.return_value}/run/{job_run_id_mock.return_value}"
            ),
            MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.JOB),
            MLFLOW_DATABRICKS_JOB_ID: job_id_mock.return_value,
            MLFLOW_DATABRICKS_JOB_RUN_ID: job_run_id_mock.return_value,
            MLFLOW_DATABRICKS_JOB_TYPE: job_type_mock.return_value,
            MLFLOW_DATABRICKS_WEBAPP_URL: webapp_url_mock.return_value,
            MLFLOW_DATABRICKS_WORKSPACE_URL: workspace_url_mock.return_value,
            MLFLOW_DATABRICKS_WORKSPACE_ID: workspace_info_mock.return_value[1],
        }

    with (
        patch_job_id as job_id_mock,
        patch_job_run_id as job_run_id_mock,
        patch_job_type as job_type_mock,
        patch_webapp_url as webapp_url_mock,
        patch_workspace_url_none as workspace_url_mock,
        patch_workspace_info as workspace_info_mock,
    ):
        assert DatabricksJobRunContext().tags() == {
            MLFLOW_SOURCE_NAME: (
                f"jobs/{job_id_mock.return_value}/run/{job_run_id_mock.return_value}"
            ),
            MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.JOB),
            MLFLOW_DATABRICKS_JOB_ID: job_id_mock.return_value,
            MLFLOW_DATABRICKS_JOB_RUN_ID: job_run_id_mock.return_value,
            MLFLOW_DATABRICKS_JOB_TYPE: job_type_mock.return_value,
            MLFLOW_DATABRICKS_WEBAPP_URL: webapp_url_mock.return_value,
            MLFLOW_DATABRICKS_WORKSPACE_URL: workspace_info_mock.return_value[0],  # fallback value
            MLFLOW_DATABRICKS_WORKSPACE_ID: workspace_info_mock.return_value[1],
        }


def test_databricks_job_run_context_tags_nones():
    patch_job_id = mock.patch("mlflow.utils.databricks_utils.get_job_id", return_value=None)
    patch_job_run_id = mock.patch("mlflow.utils.databricks_utils.get_job_run_id", return_value=None)
    patch_job_type = mock.patch("mlflow.utils.databricks_utils.get_job_type", return_value=None)
    patch_webapp_url = mock.patch("mlflow.utils.databricks_utils.get_webapp_url", return_value=None)
    patch_workspace_info = mock.patch(
        "mlflow.utils.databricks_utils.get_workspace_info_from_dbutils", return_value=(None, None)
    )

    with patch_job_id, patch_job_run_id, patch_job_type, patch_webapp_url, patch_workspace_info:
        assert DatabricksJobRunContext().tags() == {
            MLFLOW_SOURCE_NAME: None,
            MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.JOB),
        }
