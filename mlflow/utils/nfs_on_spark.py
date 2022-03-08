from mlflow.utils.databricks_utils import is_in_databricks_runtime
from mlflow.utils._spark_utils import _get_active_spark_session

# Set spark config "spark.mlflow.nfs.rootDir" to specify a NFS (network file system) directory
# which is shared with all spark cluster nodes.
# This will help optimize routine of distributing spark driver files to remote workers.
# None represent no NFS directory available.
# Note:
#  1. If NFS directory set, you must ensure all spark cluster nodes using the same hardware and
#  installed the same OS with the same environment configured, because mlflow uses NFS directory
#  to distribute driver side virtual environment to remote workers if NFS available, heterogeneous
#  cluster nodes might cause issues under the case.
#  2. The NFS directory must be mounted before importing mlflow.
#  3. For databricks users, don't set this config, databricks product sets up internal NFS service
#  automatically.
_NFS_CACHE_ROOT_DIR = None


def get_nfs_cache_root_dir():
    if is_in_databricks_runtime():
        return "/local_disk0/.ephemeral_nfs/mlflow/cache"
    else:
        return _get_active_spark_session().conf.get("spark.mlflow.nfs.rootDir", None)
