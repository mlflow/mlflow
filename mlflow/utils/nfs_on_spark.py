import os
import shutil
import sys
import uuid

from mlflow.utils._spark_utils import _get_active_spark_session
from mlflow.utils.databricks_utils import (
    _get_dbutils,
    is_in_databricks_runtime,
    is_in_databricks_serverless,
)

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
        # Get python env for current user (i.e. the env used by %pip)
        # note the env is bound to current user, not REPL,
        # so it avoids issues in DLT runtime.
        user_python_bin_path = sys.executable
        user_env_path = os.path.dirname(os.path.dirname(user_python_bin_path))
        nfs_temp_dir = os.path.join(user_env_path, "mlflow_nfs_temp")
        os.makedirs(nfs_temp_dir, exist_ok=True)
        return nfs_temp_dir
    else:
        spark_session = _get_active_spark_session()
        if spark_session is not None:
            return spark_session.conf.get("spark.mlflow.nfs.rootDir", None)
