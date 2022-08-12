from unittest import mock

from mlflow.entities import SourceType
from mlflow.utils.mlflow_tags import (
    MLFLOW_DATABRICKS_JOB_TYPE_INFO,
    MLFLOW_EXPERIMENT_SOURCE_TYPE,
    MLFLOW_EXPERIMENT_SOURCE_ID,
)
from mlflow.tracking.default_experiment.databricks_job_experiment_provider import (
    DatabricksJobExperimentProvider,
)
from tests.helper_functions import multi_context
from mlflow import MlflowClient


def test_databricks_job_default_experiment_in_context():
    with mock.patch(
        "mlflow.utils.databricks_utils.is_in_databricks_job"
    ) as in_job_mock, mock.patch(
        "mlflow.utils.databricks_utils.get_job_type_info"
    ) as get_job_type_info:
        in_job_mock.return_value = True
        get_job_type_info.return_value = "NORMAL"
        assert DatabricksJobExperimentProvider().in_context() == True


def test_databricks_job_default_experiment_in_context_with_not_in_databricks_job():
    with mock.patch(
        "mlflow.utils.databricks_utils.is_in_databricks_job"
    ) as in_job_mock, mock.patch(
        "mlflow.utils.databricks_utils.get_job_type_info"
    ) as get_job_type_info:
        in_job_mock.return_value = False
        get_job_type_info.return_value = "NORMAL"
        assert DatabricksJobExperimentProvider().in_context() == False


def test_databricks_job_default_experiment_in_context_with_ephemeral_job_type():
    with mock.patch(
        "mlflow.utils.databricks_utils.is_in_databricks_job"
    ) as in_job_mock, mock.patch(
        "mlflow.utils.databricks_utils.get_job_type_info"
    ) as get_job_type_info:
        in_job_mock.return_value = True
        get_job_type_info.return_value = "EPHEMERAL"
        assert DatabricksJobExperimentProvider().in_context() == False


def test_databricks_job_default_experiment_id():
    job_id = "job_id"
    exp_name = "jobs:/" + str(job_id)
    patch_job_id = mock.patch("mlflow.utils.databricks_utils.get_job_id", return_value=job_id)
    patch_job_type = mock.patch(
        "mlflow.utils.databricks_utils.get_job_type_info", return_value="NORMAL"
    )
    patch_experiment_name_from_job_id = mock.patch(
        "mlflow.utils.databricks_utils.get_experiment_name_from_job_id", return_value=exp_name
    )
    experiment_id = "experiment_id"

    create_experiment = mock.patch.object(
        MlflowClient, "create_experiment", return_value=experiment_id
    )

    with multi_context(
        patch_job_id, patch_job_type, patch_experiment_name_from_job_id, create_experiment
    ) as (
        job_id_mock,
        job_type_info_mock,
        experiment_name_from_job_id_mock,
        create_experiment_mock,
    ):
        tags = {}
        tags[MLFLOW_DATABRICKS_JOB_TYPE_INFO] = job_type_info_mock.return_value
        tags[MLFLOW_EXPERIMENT_SOURCE_TYPE] = SourceType.to_string(SourceType.JOB)
        tags[MLFLOW_EXPERIMENT_SOURCE_ID] = job_id_mock.return_value

        assert DatabricksJobExperimentProvider().get_experiment_id() == experiment_id
        create_experiment_mock.assert_called_once_with(
            experiment_name_from_job_id_mock.return_value, None, tags
        )
