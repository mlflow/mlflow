import logging
import warnings
import threading

import mlflow
from mlflow.utils.databricks_utils import is_in_databricks_notebook
from mlflow.tracking.client import MlflowClient
from mlflow.tracking.context import spark_autologging_context

_JAVA_PACKAGE = "org.mlflow.spark.autologging"
_REPL_ID_JAVA_PACKAGE = "org.mlflow.spark.autologging.databricks"
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
    # def _print_thread_count():
    #     import time, threading
    #     while True:
    #         time.sleep(1)
    #         print(threading.active_count())
    #
    # import threading
    # t = threading.Thread(target=_print_thread_count)
    # t.run()

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
        # Register context provider for Spark autologging
        from mlflow.tracking.context.registry import _run_context_provider_registry
        _run_context_provider_registry.register(spark_autologging_context.SparkAutologgingContext)



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

    def _merge_tag_lines(self, existing_tag, new_table_info):
        if existing_tag is None:
            return new_table_info
        if new_table_info in existing_tag:
            return existing_tag
        return "\n".join([existing_tag, new_table_info])

    def _set_run_tag_async(self, run_id, path, version, format):
        # TODO make async
        client = MlflowClient()
        table_info_string = self._get_table_info_string(path, version, format)
        existing_run = client.get_run(run_id)
        existing_tag = existing_run.data.tags.get(spark_autologging_context._SPARK_TABLE_INFO_TAG_NAME)
        new_table_info = self._merge_tag_lines(existing_tag, table_info_string)
        client.set_tag(run_id, spark_autologging_context._SPARK_TABLE_INFO_TAG_NAME, new_table_info)

    def _add_tag_to_context_provider(self, path, version, format):
        spark_autologging_context.add_table_info(path, version, format)

    def notify(self, path, version, format):
        try:
            self._notify(path, version, format)
        except Exception as e:
            _logger.error("Unexpected exception %s while attempting to log Spark datasource "
                          "info. Exception:\n%s" % e)


    def _notify(self, path, version, format):
        """
        Method called by Scala SparkListener to propagate datasource read events to the current
        Python process
        """
        import datetime
        print("Notified in Python at %s with %s, %s, %s, active run %s, tid %s" % (datetime.datetime.now(), path, version, format, mlflow.active_run(), threading.get_ident()))
        # If there's an active run, simply set the tag on it
        # Note that there's a TOCTOU race condition here - active_run() here can actually throw
        # if the main thread happens to end the run & pop from the active run stack after we check
        # the stack size but before we peek
        active_run = mlflow.active_run()
        if active_run:
            self._set_run_tag_async(active_run.info.run_id, path, version, format)
        else:
            spark_autologging_context.add_table_info(path, version, format)


    def register(self):
        event_publisher = _get_jvm_event_publisher()
        self.uuid = event_publisher.register(self)
        return self.uuid

    def replId(self):
        from pyspark import SparkContext
        return SparkContext.getOrCreate().getLocalProperty("spark.databricks.replId")

    class Java:
        implements = ["{}.MlflowAutologEventSubscriber".format(_JAVA_PACKAGE)]
