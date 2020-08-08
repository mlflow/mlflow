from mlflow.tracking.context.abstract_context import RunContextProvider
from mlflow.utils import databricks_utils
from mlflow.utils.mlflow_tags import (
    MLFLOW_DATABRICKS_CLUSTER_ID
)


class DatabricksClusterRunContext(RunContextProvider):
    def in_context(self):
        return databricks_utils.is_in_cluster()

    def tags(self):
        cluster_id = databricks_utils.get_cluster_id()
        tags = {}
        if cluster_id is not None:
            tags[MLFLOW_DATABRICKS_CLUSTER_ID] = cluster_id
        return tags

    def execute_start_run_actions(self, run):
        pass

    def execute_end_run_actions(self, run, status):
        pass

    def execute_create_experiment_actions(self, experiment_id):
        pass

    def execute_delete_experiment_actions(self, experiment_id):
        pass
