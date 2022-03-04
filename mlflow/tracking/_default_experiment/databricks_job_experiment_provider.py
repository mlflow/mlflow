import logging
from mlflow.tracking._default_experiment.abstract_context import DefaultExperimentProvider
from mlflow.utils import databricks_utils
from mlflow.tracking.client import MlflowClient
from mlflow.utils.mlflow_tags import (
    MLFLOW_DATABRICKS_JOB_TYPE_INFO,
    MLFLOW_EXPERIMENT_SOURCE_TYPE,
    MLFLOW_EXPERIMENT_SOURCE_ID,
)

_logger = logging.getLogger(__name__)


class DatabricksJobExperimentProvider(DefaultExperimentProvider):
    def in_context(self):
        return (
            databricks_utils.is_in_databricks_job()
            and databricks_utils.get_job_type_info() == "NORMAL"
        )

    def get_experiment_id(self):
        job_id = databricks_utils.get_job_id()
        tags = {}
        tags[MLFLOW_DATABRICKS_JOB_TYPE_INFO] = databricks_utils.get_job_type_info()
        tags[MLFLOW_EXPERIMENT_SOURCE_TYPE] = "JOB"
        tags[MLFLOW_EXPERIMENT_SOURCE_ID] = job_id

        experiment_id = MlflowClient().create_experiment(
            databricks_utils.get_experiment_name_from_job_id(job_id), None, tags
        )
        _logger.debug(
            "Job experiment with experiment ID '%s' fetched or created",
            experiment_id,
        )

        return experiment_id
