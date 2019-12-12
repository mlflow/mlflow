import logging
import os
import warnings

import mlflow
from mlflow.utils import experimental
from mlflow.utils.databricks_utils import is_in_databricks_notebook
from mlflow.tracking.client import MlflowClient


_JAVA_PACKAGE = "org.mlflow.spark.autologging"
_REPL_ID_JAVA_PACKAGE = "org.mlflow.spark.autologging.databricks"
_SPARK_TABLE_INFO_TAG_NAME = "sparkTableInfo"
_SPARK_TABLE_INFO_LISTENER = None
_logger = logging.getLogger(__name__)

def _get_java_package():
    from pyspark import SparkContext
    sc = SparkContext.getOrCreate()
    spark_version_parts = sc.version.split(".")
    spark_major_version = None
    if len(spark_version_parts) > 0:
        spark_major_version = int(spark_version_parts[0])
    # TODO: will JAR be available in ML runtime for MLflow projects? If so, should we broaden this
    # check to not just look for notebooks?
    if spark_major_version is not None and spark_major_version == 2 and \
            is_in_databricks_notebook():
        return _REPL_ID_JAVA_PACKAGE
    return _JAVA_PACKAGE

def _get_jvm_event_publisher():
    """
    Get JVM-side object implementing the following methods:
    - init() for initializing JVM state needed for autologging (e.g. attaching a SparkListener
      to watch for datasource reads)
    - register(subscriber) for registering subscribers to receive datasource events
    """
    from pyspark import SparkContext
    jvm = SparkContext._gateway.jvm
    qualified_classname = "{}.{}".format(_get_java_package(), "MlflowAutologEventPublisher")
    return getattr(jvm, qualified_classname)

def _autolog():
    """Implementation of Spark datasource autologging"""
    from pyspark import SparkContext
    from py4j.java_gateway import CallbackServerParameters

    global _SPARK_TABLE_INFO_LISTENER
    if _SPARK_TABLE_INFO_LISTENER is None:
        gw = SparkContext._gateway
        if gw is None:
            warnings.warn(
                "No active SparkContext found, refusing to enable Spark datasource "
                "autologging. Please create a SparkSession e.g. via "
                "SparkSession.builder.getOrCreate() (see API docs at "
                "https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.SparkSession) "
                "before attempting to enable autologging")
            return
        params = gw.callback_server_parameters
        callback_server_params = CallbackServerParameters(
            address=params.address, port=params.port, daemonize=True, daemonize_connections=True,
            eager_load=params.eager_load, ssl_context=params.ssl_context,
            accept_timeout=params.accept_timeout, read_timeout=params.read_timeout,
            auth_token=params.auth_token)
        gw.start_callback_server(callback_server_params)

        event_publisher = _get_jvm_event_publisher()
        event_publisher.init(1)
        _SPARK_TABLE_INFO_LISTENER = PythonSubscriber()
        _SPARK_TABLE_INFO_LISTENER.register()


@experimental
def autolog():
    """
    Enables automatic logging of Spark datasource paths, versions (if applicable), and formats
    when they are read. This method is not threadsafe and assumes a SparkSession already exists.
    It should be called on the Spark driver, not on the executors (i.e. do not call this method
    within a function parallelized by Spark).

    Datasource information is logged under the current active MLflow run, creating an active run
    if none exists. Note that autologging of Spark ML (MLlib) models is not currently supported
    via this API.

    Datasource-autologging is best-effort, meaning that if Spark is under heavy load or MLflow
    logging fails for any reason (e.g. if the MLflow server is unavailable), logging may be
    dropped.

    For any unexpected issues with autologging, check Spark driver and executor logs in addition
    to stderr & stdout generated from your MLflow code - datasource information is pulled from
    Spark, so logs relevant to debugging may show up amongst the Spark logs.
    """
    try:
        _autolog()
    except Exception as e:
        warnings.warn("Could not enable Spark datasource autologging, got error:\n%s" % e)


class PythonSubscriber(object):
    """
    Subscriber, intended to be instantiated once per Python process, that logs Spark table
    information propagated from Java to the current MLflow run, starting a run if necessary.
    class implements a Java interface (org.mlflow.spark.autologging.MlflowAutologEventSubscriber,
    defined in the mlflow-spark package) that's called-into by autologging logic in the JVM in order
    to propagate Spark datasource read events to Python.
    """
    def __init__(self):
        self.uuid = None

    def toString(self):
        # For debugging
        return "PythonSubscriber<uuid=%s>" % self.uuid

    def ping(self):
        return None

    def _get_table_info_string(self, path, version, format):
        if format == "delta":
            return "path={path},version={version},format={format}".format(
                path=path, version=version, format=format)
        return "path={path},format={format}".format(path=path, format=format)

    def notify(self, path, version, format):
        """
        This method is required by Scala Listener interface
        we defined above.
        """
        # If there's an active run, simply set the tag on it
        client = MlflowClient()
        active_run = mlflow.active_run()
        active_run_id = active_run.info.run_id if active_run is not None else os.environ.get("MLFLOW_RUN_ID")
        table_info_string = self._get_table_info_string(path, version, format)
        if active_run_id is not None:
            existing_run = client.get_run(active_run_id)
            existing_tag = existing_run.data.tags.get(_SPARK_TABLE_INFO_TAG_NAME)
            new_table_info = []
            if existing_tag is not None:
                # If we already logged the current table info, exit early
                if table_info_string in existing_tag:
                    return
                new_table_info.append(existing_tag)
            new_table_info.append(table_info_string)
            client.set_tag(active_run_id, _SPARK_TABLE_INFO_TAG_NAME, "\n".join(new_table_info))
        # Otherwise, create a run & log to it, setting the MLFLOW_RUN_ID env variable
        else:
            mlflow.start_run()
            # TODO: make this legit via try_mlflow_log
            try:
                mlflow.set_tag(_SPARK_TABLE_INFO_TAG_NAME, table_info_string)
                active_run_id = mlflow.active_run().info.run_id
            finally:
                mlflow.end_run("RUNNING")
            os.environ["MLFLOW_RUN_ID"] = active_run_id

    def register(self):
        event_publisher = _get_jvm_event_publisher()
        self.uuid = event_publisher.register(self)
        return self.uuid

    def replId(self):
        from pyspark import SparkContext
        return SparkContext.getOrCreate().getLocalProperty("spark.databricks.replId")

    class Java:
        implements = ["{}.MlflowAutologEventSubscriber".format(_JAVA_PACKAGE)]
