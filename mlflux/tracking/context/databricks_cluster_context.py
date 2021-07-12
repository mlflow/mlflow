from mlflux.tracking.context.abstract_context import RunContextProvider
from mlflux.utils import databricks_utils
from mlflux.utils.mlflow_tags import MLFLOW_DATABRICKS_CLUSTER_ID


class DatabricksClusterRunContext(RunContextProvider):
    def in_context(self):
        return databricks_utils.is_in_cluster()

    def tags(self):
        cluster_id = databricks_utils.get_cluster_id()
        tags = {}
        if cluster_id is not None:
            tags[MLFLOW_DATABRICKS_CLUSTER_ID] = cluster_id
        return tags
