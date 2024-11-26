import concurrent.futures
import logging
import sys
import threading
import uuid

from py4j.java_gateway import CallbackServerParameters

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.spark import FLAVOR_NAME
from mlflow.tracking.context.abstract_context import RunContextProvider
from mlflow.tracking.fluent import _get_latest_active_run
from mlflow.utils import _truncate_and_ellipsize
from mlflow.utils.autologging_utils import (
    ExceptionSafeClass,
    autologging_is_disabled,
)
from mlflow.utils.databricks_utils import get_repl_id as get_databricks_repl_id
from mlflow.utils.validation import MAX_TAG_VAL_LENGTH

_JAVA_PACKAGE = "org.mlflow.spark.autologging"
_SPARK_TABLE_INFO_TAG_NAME = "sparkDatasourceInfo"

_logger = logging.getLogger(__name__)
_lock = threading.Lock()
_table_infos = []
_spark_table_info_listener = None

# Queue & singleton consumer thread for logging Spark datasource info asynchronously
_metric_queue = []
_thread_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="MlflowSparkAutologging"
)


# Exposed for testing
def _get_current_listener():
    return _spark_table_info_listener


def _get_table_info_string(path, version, data_format):
    if data_format == "delta":
        return f"path={path},version={version},format={data_format}"
    return f"path={path},format={data_format}"


def _merge_tag_lines(existing_tag, new_table_info):
    if existing_tag is None:
        return new_table_info
    if new_table_info in existing_tag:
        return existing_tag
    return "\n".join([existing_tag, new_table_info])


def add_table_info_to_context_provider(path, version, data_format):
    with _lock:
        _table_infos.append((path, version, data_format))


def clear_table_infos():
    """Clear the table info accumulated SparkAutologgingContext.

    This is currently only used in unit tests.
    """
    with _lock:
        global _table_infos
        _table_infos = []


def _get_spark_major_version(sc):
    spark_version_parts = sc.version.split(".")
    spark_major_version = None
    if len(spark_version_parts) > 0:
        spark_major_version = int(spark_version_parts[0])
    return spark_major_version


def _get_jvm_event_publisher(spark_context):
    """
    Get JVM-side object implementing the following methods:
    - init() for initializing JVM state needed for autologging (e.g. attaching a SparkListener
      to watch for datasource reads)
    - register(subscriber) for registering subscribers to receive datasource events
    """
    jvm = spark_context._gateway.jvm
    qualified_classname = "{}.{}".format(_JAVA_PACKAGE, "MlflowAutologEventPublisher")
    return getattr(jvm, qualified_classname)


def _generate_datasource_tag_value(table_info_string):
    return _truncate_and_ellipsize(table_info_string, MAX_TAG_VAL_LENGTH)


def _set_run_tag_async(run_id, path, version, data_format):
    _thread_pool.submit(
        _set_run_tag, run_id=run_id, path=path, version=version, data_format=data_format
    )


def _set_run_tag(run_id, path, version, data_format):
    client = MlflowClient()
    table_info_string = _get_table_info_string(path, version, data_format)
    existing_run = client.get_run(run_id)
    existing_tag = existing_run.data.tags.get(_SPARK_TABLE_INFO_TAG_NAME)
    new_table_info = _merge_tag_lines(existing_tag, table_info_string)
    new_tag_value = _generate_datasource_tag_value(new_table_info)
    client.set_tag(run_id, _SPARK_TABLE_INFO_TAG_NAME, new_tag_value)


def _stop_listen_for_spark_activity(spark_context):
    gw = spark_context._gateway
    try:
        gw.shutdown_callback_server()
    except Exception as e:
        _logger.warning("Failed to shut down Spark callback server for autologging: %s", e)


def _listen_for_spark_activity(spark_context):
    global _spark_table_info_listener
    if _get_current_listener() is not None:
        return

    if _get_spark_major_version(spark_context) < 3:
        raise MlflowException("Spark autologging unsupported for Spark versions < 3")

    gw = spark_context._gateway
    params = gw.callback_server_parameters
    callback_server_params = CallbackServerParameters(
        address=params.address,
        port=params.port,
        daemonize=True,
        daemonize_connections=True,
        eager_load=params.eager_load,
        ssl_context=params.ssl_context,
        accept_timeout=params.accept_timeout,
        read_timeout=params.read_timeout,
        auth_token=params.auth_token,
    )
    callback_server_started = gw.start_callback_server(callback_server_params)

    try:
        event_publisher = _get_jvm_event_publisher(spark_context)
        event_publisher.init(1)
        _spark_table_info_listener = PythonSubscriber()
        event_publisher.register(_spark_table_info_listener)
    except Exception as e:
        if callback_server_started:
            try:
                gw.shutdown_callback_server()
            except Exception as e:
                _logger.warning(
                    "Failed to shut down Spark callback server for autologging: %s", str(e)
                )
        _spark_table_info_listener = None
        raise MlflowException(
            "Exception while attempting to initialize JVM-side state for Spark datasource "
            "autologging. Note that Spark datasource autologging only works with Spark 3.0 "
            "and above. Please create a new Spark session with required Spark version and "
            "ensure you have the mlflow-spark JAR attached to your Spark session as described "
            f"in https://mlflow.org/docs/latest/tracking/autolog.html#spark Exception:\n{e}"
        )

    # Register context provider for Spark autologging
    from mlflow.tracking.context.registry import _run_context_provider_registry

    _run_context_provider_registry.register(SparkAutologgingContext)

    _logger.info("Autologging successfully enabled for spark.")


def _get_repl_id():
    """
    Get a unique REPL ID for a PythonSubscriber instance. This is used to distinguish between
    REPLs in multitenant, REPL-aware environments where multiple Python processes may share the
    same Spark JVM (e.g. in Databricks). In such environments, we pull the REPL ID from Spark
    local properties, and expect that the PythonSubscriber for the current Python process only
    receives events for datasource reads triggered by the current process.
    """
    repl_id = get_databricks_repl_id()
    if repl_id:
        return repl_id
    main_file = sys.argv[0] if len(sys.argv) > 0 else "<console>"
    return f"PythonSubscriber[{main_file}][{uuid.uuid4().hex}]"


class PythonSubscriber(metaclass=ExceptionSafeClass):
    """
    Subscriber, intended to be instantiated once per Python process, that logs Spark table
    information propagated from Java to the current MLflow run, starting a run if necessary.
    class implements a Java interface (org.mlflow.spark.autologging.MlflowAutologEventSubscriber,
    defined in the mlflow-spark package) that's called-into by autologging logic in the JVM in order
    to propagate Spark datasource read events to Python.

    This class leverages the Py4j callback mechanism to receive callbacks from the JVM, see
    https://www.py4j.org/advanced_topics.html#implementing-java-interfaces-from-python-callback for
    more information.
    """

    def __init__(self):
        self._repl_id = _get_repl_id()

    def toString(self):
        # For debugging
        return f"PythonSubscriber<replId={self.replId()}>"

    def ping(self):
        return None

    def notify(self, path, version, data_format):
        try:
            self._notify(path, version, data_format)
        except Exception as e:
            _logger.error(
                "Unexpected exception %s while attempting to log Spark datasource "
                "info. Exception:\n",
                e,
            )

    def _notify(self, path, version, data_format):
        """
        Method called by Scala SparkListener to propagate datasource read events to the current
        Python process
        """
        if autologging_is_disabled(FLAVOR_NAME):
            return
        # If there are active runs, simply set the tag on the latest active run
        # Note that there's a TOCTOU race condition here - active_run() here can actually throw
        # if the main thread happens to end the run & pop from the active run stack after we check
        # the stack size but before we peek

        # Note Spark datasource autologging is hard to support thread-local behavior,
        # because the spark event listener callback (jvm side) does not have the python caller
        # thread information, therefore the tag is set to the latest active run, ignoring threading
        # information. This way, consistent behavior is kept with existing functionality for
        # Spark in MLflow.
        latest_active_run = _get_latest_active_run()

        if latest_active_run:
            _set_run_tag_async(latest_active_run.info.run_id, path, version, data_format)
        else:
            add_table_info_to_context_provider(path, version, data_format)

    def replId(self):
        return self._repl_id

    class Java:
        implements = [f"{_JAVA_PACKAGE}.MlflowAutologEventSubscriber"]


class SparkAutologgingContext(RunContextProvider):
    """
    Context provider used when there's no active run. Accumulates datasource read information,
    then logs that information to the next-created run. Note that this doesn't clear the accumulated
    info when logging them to the next run, so it will be logged to any successive runs as well.
    """

    def in_context(self):
        return True

    def tags(self):
        # if autologging is disabled, then short circuit `tags()` and return empty dict.
        if autologging_is_disabled(FLAVOR_NAME):
            return {}
        with _lock:
            seen = set()
            unique_infos = []
            for info in _table_infos:
                if info not in seen:
                    unique_infos.append(info)
                    seen.add(info)
            if len(unique_infos) > 0:
                tags = {
                    _SPARK_TABLE_INFO_TAG_NAME: _generate_datasource_tag_value(
                        "\n".join([_get_table_info_string(*info) for info in unique_infos])
                    )
                }
            else:
                tags = {}
            return tags
