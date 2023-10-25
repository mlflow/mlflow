import os
import shutil
import tempfile
import zipfile


def _get_active_spark_session():
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        # Return None if user doesn't have PySpark installed
        return None
    try:
        # getActiveSession() only exists in Spark 3.0 and above
        return SparkSession.getActiveSession()
    except Exception:
        # Fall back to this internal field for Spark 2.x and below.
        return SparkSession._instantiatedSession


# Suppose we have a parent process already initiate a spark session that connected to a spark
# cluster, then the parent process spawns a child process, if child process directly creates
# a local spark session, it does not work correctly, because of PYSPARK_GATEWAY_PORT and
# PYSPARK_GATEWAY_SECRET are inherited from parent process and child process pyspark session
# will try to connect to the port and cause error.
# So the 2 lines here are to clear 'PYSPARK_GATEWAY_PORT' and 'PYSPARK_GATEWAY_SECRET' to
# enforce launching a new pyspark JVM gateway.
def _prepare_subprocess_environ_for_creating_local_spark_session():
    from mlflow.utils.databricks_utils import is_in_databricks_runtime

    if is_in_databricks_runtime():
        os.environ["SPARK_DIST_CLASSPATH"] = "/databricks/jars/*"

    os.environ.pop("PYSPARK_GATEWAY_PORT", None)
    os.environ.pop("PYSPARK_GATEWAY_SECRET", None)


def _create_local_spark_session_for_recipes():
    """Create a sparksession to be used within an recipe step run in a subprocess locally."""

    try:
        from pyspark.sql import SparkSession
    except ImportError:
        # Return None if user doesn't have PySpark installed
        return None
    _prepare_subprocess_environ_for_creating_local_spark_session()
    return (
        SparkSession.builder.master("local[*]")
        .config("spark.jars.packages", "io.delta:delta-core_2.12:2.4.0")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        )
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )


def _create_local_spark_session_for_loading_spark_model():
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.config("spark.python.worker.reuse", "true")
        # The config is a workaround for avoiding databricks delta cache issue when loading
        # some specific model such as ALSModel.
        .config("spark.databricks.io.cache.enabled", "false")
        # In Spark 3.1 and above, we need to set this conf explicitly to enable creating
        # a SparkSession on the workers
        .config("spark.executor.allowSparkContext", "true")
        # Binding "spark.driver.bindAddress" to 127.0.0.1 helps avoiding some local hostname
        # related issues (e.g. https://github.com/mlflow/mlflow/issues/5733).
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.executor.allowSparkContext", "true")
        .config(
            "spark.driver.extraJavaOptions",
            "-Dlog4j.configuration=file:/usr/local/spark/conf/log4j.properties",
        )
        .master("local[1]")
        .getOrCreate()
    )


_NFS_PATH_PREFIX = "nfs:"


def _get_spark_distributor_nfs_cache_dir():
    from mlflow.utils.nfs_on_spark import get_nfs_cache_root_dir  # avoid circular import

    if (nfs_root_dir := get_nfs_cache_root_dir()) is not None:
        cache_dir = os.path.join(nfs_root_dir, "mlflow_distributor_cache_dir")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
    return None


class _SparkDirectoryDistributor:
    """Distribute spark directory from driver to executors."""

    _extracted_dir_paths = {}

    def __init__(self):
        pass

    @staticmethod
    def add_dir(spark, dir_path):
        """Given a SparkSession and a model_path which refers to a pyfunc directory locally,
        we will zip the directory up, enable it to be distributed to executors, and return
        the "archive_path", which should be used as the path in get_or_load().
        """
        _, archive_basepath = tempfile.mkstemp()
        # NB: We must archive the directory as Spark.addFile does not support non-DFS
        # directories when recursive=True.
        archive_path = shutil.make_archive(archive_basepath, "zip", dir_path)

        if (nfs_cache_dir := _get_spark_distributor_nfs_cache_dir()) is not None:
            # If NFS directory (shared by all spark nodes) is available, use NFS directory
            # instead of `SparkContext.addFile` to distribute files.
            # Because `SparkContext.addFile` is not secure, so it is not allowed to be called
            # on a shared cluster.
            dest_path = os.path.join(nfs_cache_dir, os.path.basename(archive_path))
            shutil.copy(archive_path, dest_path)
            return _NFS_PATH_PREFIX + dest_path

        spark.sparkContext.addFile(archive_path)
        return archive_path

    @staticmethod
    def get_or_extract(archive_path):
        """Given a path returned by add_local_model(), this method will return a tuple of
        (loaded_model, local_model_path).
        If this Python process ever loaded the model before, we will reuse that copy.
        """
        from pyspark.files import SparkFiles

        if archive_path in _SparkDirectoryDistributor._extracted_dir_paths:
            return _SparkDirectoryDistributor._extracted_dir_paths[archive_path]

        # BUG: Despite the documentation of SparkContext.addFile() and SparkFiles.get() in Scala
        # and Python, it turns out that we actually need to use the basename as the input to
        # SparkFiles.get(), as opposed to the (absolute) path.
        if archive_path.startswith(_NFS_PATH_PREFIX):
            local_path = archive_path[len(_NFS_PATH_PREFIX) :]
        else:
            archive_path_basename = os.path.basename(archive_path)
            local_path = SparkFiles.get(archive_path_basename)
        temp_dir = tempfile.mkdtemp()
        zip_ref = zipfile.ZipFile(local_path, "r")
        zip_ref.extractall(temp_dir)
        zip_ref.close()

        _SparkDirectoryDistributor._extracted_dir_paths[archive_path] = temp_dir
        return _SparkDirectoryDistributor._extracted_dir_paths[archive_path]
