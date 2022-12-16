from mlflow.tracking.context.abstract_context import RunContextProvider
from mlflow.utils import databricks_utils
from mlflow.utils.annotations import developer_stable
from mlflow.utils.mlflow_tags import MLFLOW_DATABRICKS_CLUSTER_ID


@developer_stable
class DatabricksClusterRunContext(RunContextProvider):
    def in_context(self):
        return databricks_utils.is_in_cluster()

    def tags(self):
        cluster_id = databricks_utils.get_cluster_id()
        tags = {}
        if cluster_id is not None:
            tags[MLFLOW_DATABRICKS_CLUSTER_ID] = cluster_id
        return tags
