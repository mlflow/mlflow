from mlflow.tracking.context.abstract_context import RunContextProvider
from mlflow.utils import databricks_utils
from mlflow.utils.annotations import developer_stable
from mlflow.utils.mlflow_tags import MLFLOW_DATABRICKS_NOTEBOOK_COMMAND_ID


@developer_stable
class DatabricksCommandRunContext(RunContextProvider):
    def in_context(self):
        return databricks_utils.get_job_group_id() is not None

    def tags(self):
        job_group_id = databricks_utils.get_job_group_id()
        tags = {}
        if job_group_id is not None:
            tags[MLFLOW_DATABRICKS_NOTEBOOK_COMMAND_ID] = job_group_id
        return tags
