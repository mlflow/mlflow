import os
import shutil
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
        spark_sess = _get_active_spark_session()
        if is_in_databricks_serverless():
            nfs_enabled = True
        else:
            nfs_enabled = spark_sess and (
                spark_sess.conf.get("spark.databricks.mlflow.nfs.enabled", "true").lower() == "true"
            )
        if nfs_enabled:
            try:
                # The directory `getReplNFSTempDir` returns has read/write permissions.
                return _get_dbutils().entry_point.getReplNFSTempDir()
            except Exception:
                nfs_root_dir = "/local_disk0/.ephemeral_nfs"
                # Test whether the NFS directory is writable.
                test_path = os.path.join(nfs_root_dir, uuid.uuid4().hex)
                try:
                    os.makedirs(test_path)
                    return nfs_root_dir
                except Exception:
                    # For databricks cluster enabled Table ACL, we have no permission to access NFS
                    # directory, in this case, return None, meaning NFS is not available.
                    return None
                finally:
                    shutil.rmtree(test_path, ignore_errors=True)
        else:
            return None
    else:
        spark_session = _get_active_spark_session()
        if spark_session is not None:
            return spark_session.conf.get("spark.mlflow.nfs.rootDir", None)
