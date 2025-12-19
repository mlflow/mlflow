import functools
import getpass
import json
import logging
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, NamedTuple, ParamSpec, TypeVar

from mlflow.utils.logging_utils import eprint
from mlflow.utils.request_utils import augmented_raise_for_status

if TYPE_CHECKING:
    from pyspark.sql.connect.session import SparkSession as SparkConnectSession


import mlflow.utils
from mlflow.environment_variables import (
    _SERVERLESS_GPU_COMPUTE_ASSOCIATED_JOB_RUN_ID,
    MLFLOW_ENABLE_DB_SDK,
    MLFLOW_TRACKING_URI,
)
from mlflow.exceptions import MlflowException
from mlflow.legacy_databricks_cli.configure.provider import (
    DatabricksConfig,
    DatabricksConfigProvider,
    DatabricksModelServingConfigProvider,
    EnvironmentVariableConfigProvider,
    ProfileConfigProvider,
    SparkTaskContextConfigProvider,
)
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils._spark_utils import _get_active_spark_session
from mlflow.utils.rest_utils import MlflowHostCreds, http_request
from mlflow.utils.uri import (
    _DATABRICKS_UNITY_CATALOG_SCHEME,
    get_db_info_from_uri,
    is_databricks_uri,
)

_logger = logging.getLogger(__name__)


_MODEL_DEPENDENCY_OAUTH_TOKEN_FILE_PATH = "/var/credentials-secret/model-dependencies-oauth-token"


def _use_repl_context_if_available(
    name: str,
    *,
    ignore_none: bool = False,
):
    """Creates a decorator to insert a short circuit that returns the specified REPL context
    attribute if it's available.

    Args:
        name: Attribute name (e.g. "apiUrl").
        ignore_none: If True, use the original function if the REPL context attribute exists but
            is None.

    Returns:
        Decorator to insert the short circuit.
    """

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            try:
                from dbruntime.databricks_repl_context import get_context

                context = get_context()
                if context is not None and hasattr(context, name):
                    attr = getattr(context, name)
                    if attr is None and ignore_none:
                        # do nothing and continue to the original function
                        pass
                    else:
                        return attr
            except Exception:
                pass
            return f(*args, **kwargs)

        return wrapper

    return decorator


def get_mlflow_credential_context_by_run_id(run_id):
    from mlflow.tracking.artifact_utils import get_artifact_uri
    from mlflow.utils.uri import get_databricks_profile_uri_from_artifact_uri

    run_root_artifact_uri = get_artifact_uri(run_id=run_id)
    profile = get_databricks_profile_uri_from_artifact_uri(run_root_artifact_uri)
    return MlflowCredentialContext(profile)


class MlflowCredentialContext:
    """Sets and clears credentials on a context using the provided profile URL."""

    def __init__(self, databricks_profile_url):
        self.databricks_profile_url = databricks_profile_url or "databricks"
        self.db_utils = _get_dbutils()

    def __enter__(self):
        db_creds = _get_databricks_creds_config(self.databricks_profile_url)
        self.db_utils.notebook.entry_point.putMlflowProperties(
            db_creds.host,
            db_creds.insecure,
            db_creds.token,
            db_creds.username,
            db_creds.password,
        )

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.db_utils.notebook.entry_point.clearMlflowProperties()


def _get_dbutils():
    try:
        import IPython

        ip_shell = IPython.get_ipython()
        if ip_shell is None:
            raise _NoDbutilsError
        return ip_shell.ns_table["user_global"]["dbutils"]
    except ImportError:
        raise _NoDbutilsError
    except KeyError:
        raise _NoDbutilsError


class _NoDbutilsError(Exception):
    pass


def _get_java_dbutils():
    dbutils = _get_dbutils()
    return dbutils.notebook.entry_point.getDbutils()


def _get_command_context():
    return _get_java_dbutils().notebook().getContext()


def _get_extra_context(context_key):
    opt = _get_command_context().extraContext().get(context_key)
    return opt.get() if opt.isDefined() else None


def _get_context_tag(context_tag_key):
    try:
        tag_opt = _get_command_context().tags().get(context_tag_key)
        if tag_opt.isDefined():
            return tag_opt.get()
    except Exception:
        pass

    return None


@_use_repl_context_if_available("aclPathOfAclRoot")
def acl_path_of_acl_root():
    try:
        return _get_command_context().aclPathOfAclRoot().get()
    except Exception:
        return _get_extra_context("aclPathOfAclRoot")


def _get_property_from_spark_context(key):
    try:
        from pyspark import TaskContext

        if task_context := TaskContext.get():
            return task_context.getLocalProperty(key)
    except Exception:
        return None


def is_databricks_default_tracking_uri(tracking_uri):
    return tracking_uri.lower().strip() == "databricks"


@_use_repl_context_if_available("isInNotebook")
def is_in_databricks_notebook():
    if _get_property_from_spark_context("spark.databricks.notebook.id") is not None:
        return True
    try:
        return path.startswith("/workspace") if (path := acl_path_of_acl_root()) else False
    except Exception:
        return False


@_use_repl_context_if_available("isInJob")
def is_in_databricks_job():
    try:
        return get_job_id() is not None and get_job_run_id() is not None
    except Exception:
        return False


def is_in_databricks_model_serving_environment():
    """
    Check if the code is running in Databricks Model Serving environment.
    The environment variable set by Databricks when starting the serving container.
    """
    val = os.environ.get("IS_IN_DB_MODEL_SERVING_ENV", "false")
    return val.lower() == "true"


def is_mlflow_tracing_enabled_in_model_serving() -> bool:
    """
    This environment variable guards tracing behaviors for models in databricks
    model serving. Tracing in serving is only enabled when this env var is true.
    """
    return os.environ.get("ENABLE_MLFLOW_TRACING", "false").lower() == "true"


# this should only be the case when we are in model serving environment
# and OAuth token file exists in specified path
def should_fetch_model_serving_environment_oauth():
    return (
        is_in_databricks_model_serving_environment()
        and os.path.exists(_MODEL_DEPENDENCY_OAUTH_TOKEN_FILE_PATH)
        and os.path.isfile(_MODEL_DEPENDENCY_OAUTH_TOKEN_FILE_PATH)
    )


def is_in_databricks_repo():
    try:
        return get_git_repo_relative_path() is not None
    except Exception:
        return False


def is_in_databricks_repo_notebook():
    try:
        path = get_notebook_path()
        return path is not None and path.startswith("/Repos")
    except Exception:
        return False


_DATABRICKS_VERSION_FILE_PATH = "/databricks/DBR_VERSION"


def get_databricks_runtime_version():
    if ver := os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        return ver
    if os.path.exists(_DATABRICKS_VERSION_FILE_PATH):
        # In Databricks DCS cluster, it doesn't have DATABRICKS_RUNTIME_VERSION
        # environment variable, we have to read version from the version file.
        with open(_DATABRICKS_VERSION_FILE_PATH) as f:
            return f.read().strip()
    return None


def is_in_databricks_runtime():
    return get_databricks_runtime_version() is not None


def is_in_databricks_serverless_runtime():
    dbr_version = get_databricks_runtime_version()
    return dbr_version and dbr_version.startswith("client.")


def is_in_databricks_shared_cluster_runtime():
    from mlflow.utils.spark_utils import is_spark_connect_mode

    return (
        is_in_databricks_runtime()
        and is_spark_connect_mode()
        and not is_in_databricks_serverless_runtime()
    )


def is_databricks_connect(spark=None):
    """
    Return True if current Spark-connect client connects to Databricks cluster.
    """
    from mlflow.utils.spark_utils import is_spark_connect_mode

    if is_in_databricks_serverless_runtime() or is_in_databricks_shared_cluster_runtime():
        return True

    spark = spark or _get_active_spark_session()
    if spark is None:
        return False

    if not is_spark_connect_mode():
        return False

    if hasattr(spark.client, "metadata"):
        metadata = spark.client.metadata()
    else:
        metadata = spark.client._builder.metadata()

    return any(k in ["x-databricks-session-id", "x-databricks-cluster-id"] for k, v in metadata)


@dataclass
class DBConnectUDFSandboxInfo:
    spark: "SparkConnectSession"
    image_version: str
    runtime_version: str
    platform_machine: str
    mlflow_version: str


_dbconnect_udf_sandbox_info_cache: DBConnectUDFSandboxInfo | None = None


def get_dbconnect_udf_sandbox_info(spark):
    """
    Get Databricks UDF sandbox info which includes the following fields:
     - image_version like
      '{major_version}.{minor_version}' or 'client.{major_version}.{minor_version}'
     - runtime_version like '{major_version}.{minor_version}'
     - platform_machine like 'x86_64' or 'aarch64'
     - mlflow_version
    """
    global _dbconnect_udf_sandbox_info_cache
    from pyspark.sql.functions import pandas_udf

    if (
        _dbconnect_udf_sandbox_info_cache is not None
        and spark is _dbconnect_udf_sandbox_info_cache.spark
    ):
        return _dbconnect_udf_sandbox_info_cache

    # version is like '15.4.x-scala2.12'
    version = spark.sql("SELECT current_version().dbr_version").collect()[0][0]
    major, minor, *_rest = version.split(".")
    runtime_version = f"{major}.{minor}"

    # For Databricks Serverless python REPL,
    # the UDF sandbox runs on client image, which has version like 'client.1.1'
    # in other cases, UDF sandbox runs on databricks runtime image with version like '15.4'
    if is_in_databricks_runtime():
        _dbconnect_udf_sandbox_info_cache = DBConnectUDFSandboxInfo(
            spark=_get_active_spark_session(),
            runtime_version=runtime_version,
            image_version=get_databricks_runtime_version(),
            platform_machine=platform.machine(),
            # In databricks runtime, driver and executor should have the
            # same version.
            mlflow_version=mlflow.__version__,
        )
    else:
        image_version = runtime_version

        @pandas_udf("string")
        def f(_):
            import pandas as pd

            platform_machine = platform.machine()

            try:
                import mlflow

                mlflow_version = mlflow.__version__
            except ImportError:
                mlflow_version = ""

            return pd.Series([f"{platform_machine}\n{mlflow_version}"])

        platform_machine, mlflow_version = (
            spark.range(1).select(f("id")).collect()[0][0].split("\n")
        )
        if mlflow_version == "":
            mlflow_version = None
        _dbconnect_udf_sandbox_info_cache = DBConnectUDFSandboxInfo(
            spark=spark,
            image_version=image_version,
            runtime_version=runtime_version,
            platform_machine=platform_machine,
            mlflow_version=mlflow_version,
        )

    return _dbconnect_udf_sandbox_info_cache


def is_databricks_serverless(spark):
    """
    Return True if running on Databricks Serverless notebook or
    on Databricks Connect client that connects to Databricks Serverless.
    """
    from mlflow.utils.spark_utils import is_spark_connect_mode

    if not is_spark_connect_mode():
        return False

    if hasattr(spark.client, "metadata"):
        metadata = spark.client.metadata()
    else:
        metadata = spark.client._builder.metadata()

    return any(k == "x-databricks-session-id" for k, v in metadata)


def is_dbfs_fuse_available():
    if not is_in_databricks_runtime():
        return False

    try:
        return (
            subprocess.call(
                ["mountpoint", "/dbfs"],
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )
            == 0
        )
    except Exception:
        return False


def is_uc_volume_fuse_available():
    try:
        return (
            subprocess.call(
                ["mountpoint", "/Volumes"],
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )
            == 0
        )
    except Exception:
        return False


@_use_repl_context_if_available("isInCluster")
def is_in_cluster():
    try:
        spark_session = _get_active_spark_session()
        return (
            spark_session is not None
            and spark_session.conf.get("spark.databricks.clusterUsageTags.clusterId", None)
            is not None
        )
    except Exception:
        return False


@_use_repl_context_if_available("notebookId")
def get_notebook_id():
    """Should only be called if is_in_databricks_notebook is true"""
    if notebook_id := _get_property_from_spark_context("spark.databricks.notebook.id"):
        return notebook_id
    if (path := acl_path_of_acl_root()) and path.startswith("/workspace"):
        return path.split("/")[-1]
    return None


@_use_repl_context_if_available("notebookPath")
def get_notebook_path():
    """Should only be called if is_in_databricks_notebook is true"""
    path = _get_property_from_spark_context("spark.databricks.notebook.path")
    if path is not None:
        return path
    try:
        return _get_command_context().notebookPath().get()
    except Exception:
        return _get_extra_context("notebook_path")


@_use_repl_context_if_available("clusterId")
def get_cluster_id():
    spark_session = _get_active_spark_session()
    if spark_session is None:
        return None
    return spark_session.conf.get("spark.databricks.clusterUsageTags.clusterId", None)


@_use_repl_context_if_available("jobGroupId")
def get_job_group_id():
    try:
        dbutils = _get_dbutils()
        job_group_id = dbutils.entry_point.getJobGroupId()
        if job_group_id is not None:
            return job_group_id
    except Exception:
        return None


@_use_repl_context_if_available("replId")
def get_repl_id():
    """
    Returns:
        The ID of the current Databricks Python REPL.
    """
    # Attempt to fetch the REPL ID from the Python REPL's entrypoint object. This REPL ID
    # is guaranteed to be set upon REPL startup in DBR / MLR 9.0
    try:
        dbutils = _get_dbutils()
        repl_id = dbutils.entry_point.getReplId()
        if repl_id is not None:
            return repl_id
    except Exception:
        pass

    # If the REPL ID entrypoint property is unavailable due to an older runtime version (< 9.0),
    # attempt to fetch the REPL ID from the Spark Context. This property may not be available
    # until several seconds after REPL startup
    try:
        from pyspark import SparkContext

        repl_id = SparkContext.getOrCreate().getLocalProperty("spark.databricks.replId")
        if repl_id is not None:
            return repl_id
    except Exception:
        pass


@_use_repl_context_if_available("jobId")
def get_job_id():
    try:
        return _get_command_context().jobId().get()
    except Exception:
        return _get_context_tag("jobId")


@_use_repl_context_if_available("idInJob")
def get_job_run_id():
    try:
        return _get_command_context().idInJob().get()
    except Exception:
        return _get_context_tag("idInJob")


@_use_repl_context_if_available("jobTaskType")
def get_job_type():
    """Should only be called if is_in_databricks_job is true"""
    try:
        return _get_command_context().jobTaskType().get()
    except Exception:
        return _get_context_tag("jobTaskType")


@_use_repl_context_if_available("jobType")
def get_job_type_info():
    try:
        return _get_context_tag("jobType")
    except Exception:
        return None


@_use_repl_context_if_available("commandRunId")
def get_command_run_id():
    try:
        return _get_command_context().commandRunId().get()
    except Exception:
        # Older runtimes may not have the commandRunId available
        return None


@_use_repl_context_if_available("workloadId")
def get_workload_id():
    try:
        return _get_command_context().workloadId().get()
    except Exception:
        return _get_context_tag("workloadId")


@_use_repl_context_if_available("workloadClass")
def get_workload_class():
    try:
        return _get_command_context().workloadClass().get()
    except Exception:
        return _get_context_tag("workloadClass")


@_use_repl_context_if_available("apiUrl")
def get_webapp_url():
    """Should only be called if is_in_databricks_notebook or is_in_databricks_jobs is true"""
    url = _get_property_from_spark_context("spark.databricks.api.url")
    if url is not None:
        return url
    try:
        return _get_command_context().apiUrl().get()
    except Exception:
        return _get_extra_context("api_url")


@_use_repl_context_if_available("workspaceId")
def get_workspace_id():
    try:
        return _get_command_context().workspaceId().get()
    except Exception:
        return _get_context_tag("orgId")


@_use_repl_context_if_available("browserHostName")
def get_browser_hostname():
    try:
        return _get_command_context().browserHostName().get()
    except Exception:
        return _get_context_tag("browserHostName")


def get_workspace_info_from_dbutils():
    try:
        dbutils = _get_dbutils()
        if dbutils:
            browser_hostname = get_browser_hostname()
            workspace_host = "https://" + browser_hostname if browser_hostname else get_webapp_url()
            workspace_id = get_workspace_id()
            return workspace_host, workspace_id
    except Exception:
        pass
    return None, None


@_use_repl_context_if_available("workspaceUrl", ignore_none=True)
def _get_workspace_url():
    try:
        if spark_session := _get_active_spark_session():
            if workspace_url := spark_session.conf.get("spark.databricks.workspaceUrl", None):
                return workspace_url
    except Exception:
        return None


def get_workspace_url():
    if url := _get_workspace_url():
        return f"https://{url}" if not url.startswith("https://") else url
    return None


def warn_on_deprecated_cross_workspace_registry_uri(registry_uri):
    workspace_host, workspace_id = get_workspace_info_from_databricks_secrets(
        tracking_uri=registry_uri
    )
    if workspace_host is not None or workspace_id is not None:
        _logger.warning(
            "Accessing remote workspace model registries using registry URIs of the form "
            "'databricks://scope:prefix', or by loading models via URIs of the form "
            "'models://scope:prefix@databricks/model-name/stage-or-version', is deprecated. "
            "Use Models in Unity Catalog instead for easy cross-workspace model access, with "
            "granular per-user audit logging and no extra setup required. See "
            "https://docs.databricks.com/machine-learning/manage-model-lifecycle/index.html "
            "for more details."
        )


def get_workspace_info_from_databricks_secrets(tracking_uri):
    profile, key_prefix = get_db_info_from_uri(tracking_uri)
    if key_prefix:
        if dbutils := _get_dbutils():
            workspace_id = dbutils.secrets.get(scope=profile, key=key_prefix + "-workspace-id")
            workspace_host = dbutils.secrets.get(scope=profile, key=key_prefix + "-host")
            return workspace_host, workspace_id
    return None, None


def _fail_malformed_databricks_auth(uri):
    if uri and uri.startswith(_DATABRICKS_UNITY_CATALOG_SCHEME):
        uri_name = "registry URI"
        uri_scheme = _DATABRICKS_UNITY_CATALOG_SCHEME
    else:
        uri_name = "tracking URI"
        uri_scheme = "databricks"
    if is_in_databricks_model_serving_environment():
        raise MlflowException(
            f"Reading Databricks credential configuration in model serving failed. "
            f"Most commonly, this happens because the model currently "
            f"being served was logged without Databricks resource dependencies "
            f"properly specified. Re-log your model, specifying resource dependencies as "
            f"described in "
            f"https://docs.databricks.com/en/generative-ai/agent-framework/log-agent.html"
            f"#specify-resources-for-pyfunc-or-langchain-agent "
            f"and then register and attempt to serve it again. Alternatively, you can explicitly "
            f"configure authentication by setting environment variables as described in "
            f"https://docs.databricks.com/en/generative-ai/agent-framework/deploy-agent.html"
            f"#manual-authentication. "
            f"Additional debug info: the MLflow {uri_name} was set to '{uri}'"
        )
    raise MlflowException(
        f"Reading Databricks credential configuration failed with MLflow {uri_name} '{uri}'. "
        "Please ensure that the 'databricks-sdk' PyPI library is installed, the tracking "
        "URI is set correctly, and Databricks authentication is properly configured. "
        f"The {uri_name} can be either '{uri_scheme}' "
        f"(using profile name specified by 'DATABRICKS_CONFIG_PROFILE' environment variable "
        f"or using 'DEFAULT' authentication profile if 'DATABRICKS_CONFIG_PROFILE' environment "
        f"variable does not exist) or '{uri_scheme}://{{profile}}'. "
        "You can configure Databricks authentication in several ways, for example by "
        "specifying environment variables (e.g. DATABRICKS_HOST + DATABRICKS_TOKEN) or "
        "logging in using 'databricks auth login'. \n"
        "For details on configuring Databricks authentication, please refer to "
        "'https://docs.databricks.com/en/dev-tools/auth/index.html#unified-auth'."
    )


# Helper function to attempt to read OAuth Token from
# mounted file in Databricks Model Serving environment
def get_model_dependency_oauth_token(should_retry=True):
    try:
        with open(_MODEL_DEPENDENCY_OAUTH_TOKEN_FILE_PATH) as f:
            oauth_dict = json.load(f)
            return oauth_dict["OAUTH_TOKEN"][0]["oauthTokenValue"]
    except Exception as e:
        # sleep and retry in case of any race conditions with OAuth refreshing
        if should_retry:
            time.sleep(0.5)
            return get_model_dependency_oauth_token(should_retry=False)
        else:
            raise MlflowException(
                "Unable to read Oauth credentials from file mount for Databricks "
                "Model Serving dependency failed"
            ) from e


class TrackingURIConfigProvider(DatabricksConfigProvider):
    """
    TrackingURIConfigProvider extracts `scope` and `key_prefix` from tracking URI
    of format like `databricks://scope:key_prefix`,
    then read host and token value from dbutils secrets by key
    "{key_prefix}-host" and "{key_prefix}-token"

    This provider only works in Databricks runtime and it is deprecated,
    in Databricks runtime you can simply use 'databricks'
    as the tracking URI and MLflow can automatically read dynamic token in
    Databricks runtime.
    """

    def __init__(self, tracking_uri):
        self.tracking_uri = tracking_uri

    def get_config(self):
        scope, key_prefix = get_db_info_from_uri(self.tracking_uri)

        if scope and key_prefix:
            if dbutils := _get_dbutils():
                # Prefix differentiates users and is provided as path information in the URI
                host = dbutils.secrets.get(scope=scope, key=key_prefix + "-host")
                token = dbutils.secrets.get(scope=scope, key=key_prefix + "-token")
                return DatabricksConfig.from_token(host=host, token=token, insecure=False)

        return None


def get_databricks_host_creds(server_uri=None):
    """
    Reads in configuration necessary to make HTTP requests to a Databricks server. This
    uses Databricks SDK workspace client API,
    If no available credential configuration is found to the server URI, this function
    will attempt to retrieve these credentials from the Databricks Secret Manager. For that to work,
    the server URI will need to be of the following format: "databricks://scope:prefix". In the
    Databricks Secret Manager, we will query for a secret in the scope "<scope>" for secrets with
    keys of the form "<prefix>-host" and "<prefix>-token". Note that this prefix *cannot* be empty
    if trying to authenticate with this method. If found, those host credentials will be used. This
    method will throw an exception if sufficient auth cannot be found.

    Args:
        server_uri: A URI that specifies the Databricks profile you want to use for making
            requests.

    Returns:
        MlflowHostCreds which includes the hostname if databricks sdk authentication is available,
        otherwise includes the hostname and authentication information necessary to
        talk to the Databricks server.

    .. Warning:: This API is deprecated. In the future it might be removed.
    """

    if MLFLOW_ENABLE_DB_SDK.get():
        from databricks.sdk import WorkspaceClient

        profile, key_prefix = get_db_info_from_uri(server_uri)
        profile = profile or os.environ.get("DATABRICKS_CONFIG_PROFILE")
        if key_prefix is not None:
            try:
                config = TrackingURIConfigProvider(server_uri).get_config()
                WorkspaceClient(host=config.host, token=config.token)
                return MlflowHostCreds(
                    config.host,
                    token=config.token,
                    use_databricks_sdk=True,
                    use_secret_scope_token=True,
                )
            except Exception as e:
                raise MlflowException(
                    f"The hostname and credentials configured by {server_uri} is invalid. "
                    "Please create valid hostname secret by command "
                    f"'databricks secrets put-secret {profile} {key_prefix}-host' and "
                    "create valid token secret by command "
                    f"'databricks secrets put-secret {profile} {key_prefix}-token'."
                ) from e
        try:
            # Using databricks-sdk to create Databricks WorkspaceClient instance,
            # If authentication is failed, MLflow falls back to legacy authentication methods,
            # see `SparkTaskContextConfigProvider`, `DatabricksModelServingConfigProvider`,
            # and `TrackingURIConfigProvider`.
            # databricks-sdk supports many kinds of authentication ways,
            # it will try to read authentication information by the following ways:
            # 1. Read dynamic generated token via databricks `dbutils`.
            # 2. parse relevant environment variables (such as DATABRICKS_HOST + DATABRICKS_TOKEN
            #    or DATABRICKS_HOST + DATABRICKS_CLIENT_ID + DATABRICKS_CLIENT_SECRET)
            #    to get authentication information
            # 3. parse ~/.databrickscfg file (generated by databricks-CLI command-line tool)
            #    to get authentication information.
            # databricks-sdk is designed to hide authentication details and
            # support various authentication ways, so that it does not provide API
            # to get credential values. Instead, we can use ``WorkspaceClient``
            # API to invoke databricks shard restful APIs.
            WorkspaceClient(profile=profile)
            use_databricks_sdk = True
            databricks_auth_profile = profile
        except Exception as e:
            _logger.debug(f"Failed to create databricks SDK workspace client, error: {e!r}")
            use_databricks_sdk = False
            databricks_auth_profile = None
    else:
        use_databricks_sdk = False
        databricks_auth_profile = None

    config = _get_databricks_creds_config(server_uri)

    if not config:
        _fail_malformed_databricks_auth(profile)

    return MlflowHostCreds(
        config.host,
        username=config.username,
        password=config.password,
        ignore_tls_verification=config.insecure,
        token=config.token,
        client_id=config.client_id,
        client_secret=config.client_secret,
        use_databricks_sdk=use_databricks_sdk,
        databricks_auth_profile=databricks_auth_profile,
    )


def get_databricks_workspace_client_config(server_uri: str):
    from databricks.sdk import WorkspaceClient

    profile, key_prefix = get_db_info_from_uri(server_uri)
    profile = profile or os.environ.get("DATABRICKS_CONFIG_PROFILE")
    if key_prefix is not None:
        config = TrackingURIConfigProvider(server_uri).get_config()
        return WorkspaceClient(host=config.host, token=config.token).config

    return WorkspaceClient(profile=profile).config


@_use_repl_context_if_available("mlflowGitRepoUrl")
def get_git_repo_url():
    try:
        return _get_command_context().mlflowGitRepoUrl().get()
    except Exception:
        return _get_extra_context("mlflowGitUrl")


@_use_repl_context_if_available("mlflowGitRepoProvider")
def get_git_repo_provider():
    try:
        return _get_command_context().mlflowGitRepoProvider().get()
    except Exception:
        return _get_extra_context("mlflowGitProvider")


@_use_repl_context_if_available("mlflowGitRepoCommit")
def get_git_repo_commit():
    try:
        return _get_command_context().mlflowGitRepoCommit().get()
    except Exception:
        return _get_extra_context("mlflowGitCommit")


@_use_repl_context_if_available("mlflowGitRelativePath")
def get_git_repo_relative_path():
    try:
        return _get_command_context().mlflowGitRelativePath().get()
    except Exception:
        return _get_extra_context("mlflowGitRelativePath")


@_use_repl_context_if_available("mlflowGitRepoReference")
def get_git_repo_reference():
    try:
        return _get_command_context().mlflowGitRepoReference().get()
    except Exception:
        return _get_extra_context("mlflowGitReference")


@_use_repl_context_if_available("mlflowGitRepoReferenceType")
def get_git_repo_reference_type():
    try:
        return _get_command_context().mlflowGitRepoReferenceType().get()
    except Exception:
        return _get_extra_context("mlflowGitReferenceType")


@_use_repl_context_if_available("mlflowGitRepoStatus")
def get_git_repo_status():
    try:
        return _get_command_context().mlflowGitRepoStatus().get()
    except Exception:
        return _get_extra_context("mlflowGitStatus")


def is_running_in_ipython_environment():
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except (ImportError, ModuleNotFoundError):
        return False


def get_databricks_run_url(tracking_uri: str, run_id: str, artifact_path=None) -> str | None:
    """
    Obtains a Databricks URL corresponding to the specified MLflow Run, optionally referring
    to an artifact within the run.

    Args:
        tracking_uri: The URI of the MLflow Tracking server containing the Run.
        run_id: The ID of the MLflow Run for which to obtain a Databricks URL.
        artifact_path: An optional relative artifact path within the Run to which the URL
            should refer.

    Returns:
        A Databricks URL corresponding to the specified MLflow Run
        (and artifact path, if specified), or None if the MLflow Run does not belong to a
        Databricks Workspace.
    """
    from mlflow.tracking.client import MlflowClient

    try:
        workspace_info = (
            DatabricksWorkspaceInfo.from_environment()
            or get_databricks_workspace_info_from_uri(tracking_uri)
        )
        if workspace_info is not None:
            experiment_id = MlflowClient(tracking_uri).get_run(run_id).info.experiment_id
            return _construct_databricks_run_url(
                host=workspace_info.host,
                experiment_id=experiment_id,
                run_id=run_id,
                workspace_id=workspace_info.workspace_id,
                artifact_path=artifact_path,
            )
    except Exception:
        return None


def get_databricks_model_version_url(registry_uri: str, name: str, version: str) -> str | None:
    """Obtains a Databricks URL corresponding to the specified Model Version.

    Args:
        registry_uri: The URI of the Model Registry server containing the Model Version.
        name: The name of the registered model containing the Model Version.
        version: Version number of the Model Version.

    Returns:
        A Databricks URL corresponding to the specified Model Version, or None if the
        Model Version does not belong to a Databricks Workspace.

    """
    try:
        workspace_info = (
            DatabricksWorkspaceInfo.from_environment()
            or get_databricks_workspace_info_from_uri(registry_uri)
        )
        if workspace_info is not None:
            return _construct_databricks_model_version_url(
                host=workspace_info.host,
                name=name,
                version=version,
                workspace_id=workspace_info.workspace_id,
            )
    except Exception:
        return None


DatabricksWorkspaceInfoType = TypeVar("DatabricksWorkspaceInfo", bound="DatabricksWorkspaceInfo")


class DatabricksWorkspaceInfo:
    WORKSPACE_HOST_ENV_VAR = "_DATABRICKS_WORKSPACE_HOST"
    WORKSPACE_ID_ENV_VAR = "_DATABRICKS_WORKSPACE_ID"

    def __init__(self, host: str, workspace_id: str | None = None):
        self.host = host
        self.workspace_id = workspace_id

    @classmethod
    def from_environment(cls) -> DatabricksWorkspaceInfoType | None:
        if DatabricksWorkspaceInfo.WORKSPACE_HOST_ENV_VAR in os.environ:
            return DatabricksWorkspaceInfo(
                host=os.environ[DatabricksWorkspaceInfo.WORKSPACE_HOST_ENV_VAR],
                workspace_id=os.environ.get(DatabricksWorkspaceInfo.WORKSPACE_ID_ENV_VAR),
            )
        else:
            return None

    def to_environment(self):
        env = {
            DatabricksWorkspaceInfo.WORKSPACE_HOST_ENV_VAR: self.host,
        }
        if self.workspace_id is not None:
            env[DatabricksWorkspaceInfo.WORKSPACE_ID_ENV_VAR] = self.workspace_id

        return env


def get_databricks_workspace_info_from_uri(tracking_uri: str) -> DatabricksWorkspaceInfo | None:
    if not is_databricks_uri(tracking_uri):
        return None

    if is_databricks_default_tracking_uri(tracking_uri) and (
        is_in_databricks_notebook() or is_in_databricks_job()
    ):
        workspace_host, workspace_id = get_workspace_info_from_dbutils()
    else:
        workspace_host, workspace_id = get_workspace_info_from_databricks_secrets(tracking_uri)
        if not workspace_id:
            _logger.info(
                "No workspace ID specified; if your Databricks workspaces share the same"
                " host URL, you may want to specify the workspace ID (along with the host"
                " information in the secret manager) for run lineage tracking. For more"
                " details on how to specify this information in the secret manager,"
                " please refer to the Databricks MLflow documentation."
            )

    if workspace_host:
        return DatabricksWorkspaceInfo(host=workspace_host, workspace_id=workspace_id)
    else:
        return None


def check_databricks_secret_scope_access(scope_name):
    if dbutils := _get_dbutils():
        try:
            dbutils.secrets.list(scope_name)
        except Exception as e:
            _logger.warning(
                f"Unable to access Databricks secret scope '{scope_name}' for OpenAI credentials "
                "that will be used to deploy the model to Databricks Model Serving. "
                "Please verify that the current Databricks user has 'READ' permission for "
                "this scope. For more information, see "
                "https://mlflow.org/docs/latest/python_api/openai/index.html#credential-management-for-openai-on-databricks. "  # noqa: E501
                f"Error: {e}"
            )


def get_sgc_job_run_id() -> str | None:
    """
    Retrieves the Serverless GPU Compute (SGC) job run ID from Databricks.

    This function is used to enable automatic run resumption for SGC jobs by fetching
    the job run ID. It first checks the Databricks widget parameter, then falls back
    to checking the environment variable if the widget is not found.

    Returns:
        str or None: The SGC job run ID if available, otherwise None. Returns None
        when neither the widget nor environment variable is set.
    """
    try:
        dbutils = _get_dbutils()
        if job_run_id := dbutils.widgets.get("SERVERLESS_GPU_COMPUTE_ASSOCIATED_JOB_RUN_ID"):
            _logger.debug(f"SGC job run ID from dbutils widget: {job_run_id}")
            return job_run_id
    except _NoDbutilsError:
        _logger.debug("dbutils not available, checking environment variable")
    except Exception as e:
        _logger.debug(f"Failed to retrieve SGC job run ID from dbutils widget: {e}", exc_info=True)

    if job_run_id := _SERVERLESS_GPU_COMPUTE_ASSOCIATED_JOB_RUN_ID.get():
        _logger.debug(f"SGC job run ID from environment variable: {job_run_id}")
        return job_run_id

    return None


def _construct_databricks_run_url(
    host: str,
    experiment_id: str,
    run_id: str,
    workspace_id: str | None = None,
    artifact_path: str | None = None,
) -> str:
    run_url = host
    if workspace_id and workspace_id != "0":
        run_url += "?o=" + str(workspace_id)

    run_url += f"#mlflow/experiments/{experiment_id}/runs/{run_id}"

    if artifact_path is not None:
        run_url += f"/artifactPath/{artifact_path.lstrip('/')}"

    return run_url


def _construct_databricks_model_version_url(
    host: str, name: str, version: str, workspace_id: str | None = None
) -> str:
    model_version_url = host
    if workspace_id and workspace_id != "0":
        model_version_url += "?o=" + str(workspace_id)

    model_version_url += f"#mlflow/models/{name}/versions/{version}"

    return model_version_url


def _construct_databricks_logged_model_url(
    workspace_url: str, experiment_id: str, model_id: str, workspace_id: str | None = None
) -> str:
    """
    Get a Databricks URL for a given registered model version in Unity Catalog.

    Args:
        workspace_url: The URL of the workspace the registered model is in.
        experiment_id: The ID of the experiment the model is logged to.
        model_id: The ID of the logged model to create the URL for.
        workspace_id: The ID of the workspace to include as a query parameter (if provided).

    Returns:
        The Databricks URL for a registered model in Unity Catalog.
    """
    query = f"?o={workspace_id}" if (workspace_id and workspace_id != "0") else ""
    return f"{workspace_url}/ml/experiments/{experiment_id}/models/{model_id}{query}"


def _construct_databricks_uc_registered_model_url(
    workspace_url: str, registered_model_name: str, version: str, workspace_id: str | None = None
) -> str:
    """
    Get a Databricks URL for a given registered model version in Unity Catalog.

    Args:
        workspace_url: The URL of the workspace the registered model is in.
        registered_model_name: The full name of the registered model containing the version.
        version: The version of the registered model to create the URL for.
        workspace_id: The ID of the workspace to include as a query parameter (if provided).

    Returns:
        The Databricks URL for a registered model in Unity Catalog.
    """
    path = registered_model_name.replace(".", "/")
    query = f"?o={workspace_id}" if (workspace_id and workspace_id != "0") else ""
    return f"{workspace_url}/explore/data/models/{path}/version/{version}{query}"


def _print_databricks_deployment_job_url(
    model_name: str,
    job_id: str,
    workspace_url: str | None = None,
    workspace_id: str | None = None,
) -> str:
    if not workspace_url:
        workspace_url = get_workspace_url()
    if not workspace_id:
        workspace_id = get_workspace_id()
    # If there is no workspace_url, we cannot print the job URL
    if not workspace_url:
        return None

    query = f"?o={workspace_id}" if (workspace_id and workspace_id != "0") else ""
    job_url = f"{workspace_url}/jobs/{job_id}{query}"
    eprint(f"🔗 Linked deployment job to '{model_name}': {job_url}")
    return job_url


def _get_databricks_creds_config(tracking_uri):
    # Note:
    # `_get_databricks_creds_config` reads credential token values or password and
    # returns a `DatabricksConfig` object
    # Databricks-SDK API doesn't support reading credential token values,
    # so that in this function we still have to use
    # configuration providers defined in legacy Databricks CLI python library to
    # read token values.
    profile, key_prefix = get_db_info_from_uri(tracking_uri)

    config = None

    if profile and key_prefix:
        # legacy way to read credentials by setting `tracking_uri` to 'databricks://scope:prefix'
        providers = [TrackingURIConfigProvider(tracking_uri)]
    elif profile:
        # If `tracking_uri` is 'databricks://<profile>'
        # MLflow should only read credentials from this profile
        providers = [ProfileConfigProvider(profile)]
    else:
        providers = [
            # `EnvironmentVariableConfigProvider` should be prioritized at the highest level,
            # to align with Databricks-SDK behavior.
            EnvironmentVariableConfigProvider(),
            _dynamic_token_config_provider,
            ProfileConfigProvider(None),
            SparkTaskContextConfigProvider(),
            DatabricksModelServingConfigProvider(),
        ]

    for provider in providers:
        if provider:
            _config = provider.get_config()
            if _config is not None and _config.is_valid:
                config = _config
                break

    if not config or not config.host:
        _fail_malformed_databricks_auth(tracking_uri)

    return config


def get_databricks_env_vars(tracking_uri):
    if not mlflow.utils.uri.is_databricks_uri(tracking_uri):
        return {}

    config = _get_databricks_creds_config(tracking_uri)

    if config.auth_type == "databricks-cli":
        raise MlflowException(
            "You configured authentication type to 'databricks-cli', in this case, MLflow cannot "
            "read credential values, so that MLflow cannot construct the databricks environment "
            "variables for child process authentication."
        )

    # We set these via environment variables so that only the current profile is exposed, rather
    # than all profiles in ~/.databrickscfg; maybe better would be to mount the necessary
    # part of ~/.databrickscfg into the container
    env_vars = {}
    env_vars[MLFLOW_TRACKING_URI.name] = "databricks"
    env_vars["DATABRICKS_HOST"] = config.host
    if config.username:
        env_vars["DATABRICKS_USERNAME"] = config.username
    if config.password:
        env_vars["DATABRICKS_PASSWORD"] = config.password
    if config.token:
        env_vars["DATABRICKS_TOKEN"] = config.token
    if config.insecure:
        env_vars["DATABRICKS_INSECURE"] = str(config.insecure)
    if config.client_id:
        env_vars["DATABRICKS_CLIENT_ID"] = config.client_id
    if config.client_secret:
        env_vars["DATABRICKS_CLIENT_SECRET"] = config.client_secret

    workspace_info = get_databricks_workspace_info_from_uri(tracking_uri)
    if workspace_info is not None:
        env_vars.update(workspace_info.to_environment())

    return env_vars


def _get_databricks_serverless_env_vars() -> dict[str, str]:
    """
    Returns the environment variables required to to initialize WorkspaceClient in a subprocess
    with serverless compute.

    Note:
        Databricks authentication related environment variables such as DATABRICKS_HOST are
        set in the are set in the _capture_imported_modules function.
    """
    envs = {}
    if "SPARK_REMOTE" in os.environ:
        envs["SPARK_LOCAL_REMOTE"] = os.environ["SPARK_REMOTE"]
    else:
        _logger.warning(
            "Missing required environment variable `SPARK_LOCAL_REMOTE` or `SPARK_REMOTE`. "
            "These are necessary to initialize the WorkspaceClient with serverless compute in "
            "a subprocess in Databricks for UC function execution. Setting the value to 'true'."
        )
        envs["SPARK_LOCAL_REMOTE"] = "true"
    return envs


class DatabricksRuntimeVersion(NamedTuple):
    is_client_image: bool
    major: int
    minor: int

    @classmethod
    def parse(cls, databricks_runtime: str | None = None):
        dbr_version = databricks_runtime or get_databricks_runtime_version()
        try:
            dbr_version_splits = dbr_version.split(".", maxsplit=2)
            if dbr_version_splits[0] == "client":
                is_client_image = True
                major = int(dbr_version_splits[1])
                minor = int(dbr_version_splits[2]) if len(dbr_version_splits) > 2 else 0
            else:
                is_client_image = False
                major = int(dbr_version_splits[0])
                minor = int(dbr_version_splits[1])
            return cls(is_client_image, major, minor)
        except Exception:
            raise MlflowException(f"Failed to parse databricks runtime version '{dbr_version}'.")


def get_databricks_runtime_major_minor_version():
    return DatabricksRuntimeVersion.parse()


_dynamic_token_config_provider = None


def _init_databricks_dynamic_token_config_provider(entry_point):
    """
    set a custom DatabricksConfigProvider with the hostname and token of the
    user running the current command (achieved by looking at
    PythonAccessibleThreadLocals.commandContext, via the already-exposed
    NotebookUtils.getContext API)
    """
    global _dynamic_token_config_provider

    notebook_utils = entry_point.getDbutils().notebook()

    dbr_version = get_databricks_runtime_major_minor_version()
    dbr_major_minor_version = (dbr_version.major, dbr_version.minor)

    # the CLI code in client-branch-1.0 is the same as in the 15.0 runtime branch
    if dbr_version.is_client_image or dbr_major_minor_version >= (13, 2):

        class DynamicConfigProvider(DatabricksConfigProvider):
            def get_config(self):
                logger = entry_point.getLogger()
                try:
                    from dbruntime.databricks_repl_context import get_context

                    ctx = get_context()
                    if ctx and ctx.apiUrl and ctx.apiToken:
                        return DatabricksConfig.from_token(
                            host=ctx.apiUrl, token=ctx.apiToken, insecure=ctx.sslTrustAll
                        )
                except Exception as e:
                    _logger.debug(
                        "Unexpected internal error while constructing `DatabricksConfig` "
                        f"from REPL context: {e}",
                    )
                # Invoking getContext() will attempt to find the credentials related to the
                # current command execution, so it's critical that we execute it on every
                # get_config().
                api_url_option = notebook_utils.getContext().apiUrl()
                api_url = api_url_option.get() if api_url_option.isDefined() else None
                # Invoking getNonUcApiToken() will attempt to find the current credentials related
                # to the current command execution and refresh it if its expired automatically,
                # so it's critical that we execute it on every get_config().
                api_token = None
                try:
                    api_token = entry_point.getNonUcApiToken()
                except Exception:
                    # Using apiToken from command context would return back the token which is not
                    # refreshed.
                    fallback_api_token_option = notebook_utils.getContext().apiToken()
                    logger.logUsage(
                        "refreshableTokenNotFound",
                        {"api_url": api_url},
                        None,
                    )
                    if fallback_api_token_option.isDefined():
                        api_token = fallback_api_token_option.get()

                ssl_trust_all = entry_point.getDriverConf().workflowSslTrustAll()

                if api_token is None or api_url is None:
                    return None

                return DatabricksConfig.from_token(
                    host=api_url, token=api_token, insecure=ssl_trust_all
                )
    elif dbr_major_minor_version >= (10, 3):

        class DynamicConfigProvider(DatabricksConfigProvider):
            def get_config(self):
                try:
                    from dbruntime.databricks_repl_context import get_context

                    ctx = get_context()
                    if ctx and ctx.apiUrl and ctx.apiToken:
                        return DatabricksConfig.from_token(
                            host=ctx.apiUrl, token=ctx.apiToken, insecure=ctx.sslTrustAll
                        )
                except Exception as e:
                    _logger.debug(
                        "Unexpected internal error while constructing `DatabricksConfig` "
                        f"from REPL context: {e}",
                    )
                # Invoking getContext() will attempt to find the credentials related to the
                # current command execution, so it's critical that we execute it on every
                # get_config().
                api_token_option = notebook_utils.getContext().apiToken()
                api_url_option = notebook_utils.getContext().apiUrl()
                ssl_trust_all = entry_point.getDriverConf().workflowSslTrustAll()

                if not api_token_option.isDefined() or not api_url_option.isDefined():
                    return None

                return DatabricksConfig.from_token(
                    host=api_url_option.get(), token=api_token_option.get(), insecure=ssl_trust_all
                )
    else:

        class DynamicConfigProvider(DatabricksConfigProvider):
            def get_config(self):
                # Invoking getContext() will attempt to find the credentials related to the
                # current command execution, so it's critical that we execute it on every
                # get_config().
                api_token_option = notebook_utils.getContext().apiToken()
                api_url_option = notebook_utils.getContext().apiUrl()
                ssl_trust_all = entry_point.getDriverConf().workflowSslTrustAll()

                if not api_token_option.isDefined() or not api_url_option.isDefined():
                    return None

                return DatabricksConfig.from_token(
                    host=api_url_option.get(), token=api_token_option.get(), insecure=ssl_trust_all
                )

    _dynamic_token_config_provider = DynamicConfigProvider()


if is_in_databricks_runtime():
    try:
        dbutils = _get_dbutils()
        _init_databricks_dynamic_token_config_provider(dbutils.entry_point)
    except _NoDbutilsError:
        # If there is no dbutils available, it means it is run in databricks driver local suite,
        # in this case, we don't need to initialize databricks token because
        # there is no backend mlflow service available.
        pass


def get_databricks_nfs_temp_dir():
    entry_point = _get_dbutils().entry_point
    if getpass.getuser().lower() == "root":
        return entry_point.getReplNFSTempDir()
    else:
        try:
            # If it is not ROOT user, it means the code is running in Safe-spark.
            # In this case, we should get temporary directory of current user.
            # and `getReplNFSTempDir` will be deprecated for this case.
            return entry_point.getUserNFSTempDir()
        except Exception:
            # fallback
            return entry_point.getReplNFSTempDir()


def get_databricks_local_temp_dir():
    entry_point = _get_dbutils().entry_point
    if getpass.getuser().lower() == "root":
        return entry_point.getReplLocalTempDir()
    else:
        try:
            # If it is not ROOT user, it means the code is running in Safe-spark.
            # In this case, we should get temporary directory of current user.
            # and `getReplLocalTempDir` will be deprecated for this case.
            return entry_point.getUserLocalTempDir()
        except Exception:
            # fallback
            return entry_point.getReplLocalTempDir()


def stage_model_for_databricks_model_serving(model_name: str, model_version: str):
    response = http_request(
        host_creds=get_databricks_host_creds(),
        endpoint="/api/2.0/serving-endpoints:stageDeployment",
        method="POST",
        raise_on_status=False,
        json={
            "model_name": model_name,
            "model_version": model_version,
        },
    )
    augmented_raise_for_status(response)


P = ParamSpec("P")
T = TypeVar("T")


def databricks_api_disabled(api_name: str = "This API", alternative: str | None = None):
    """
    Decorator that disables an API method when used with Databricks.

    This decorator checks if the tracking URI is a Databricks URI and raises an error if so.

    Args:
        api_name: Name of the API for the error message.
        alternative: Optional alternative solution to suggest in the error message.

    Returns:
        Decorator function that wraps the method to check for Databricks.
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            from mlflow.tracking import get_tracking_uri
            from mlflow.utils.uri import is_databricks_uri

            tracking_uri = get_tracking_uri()
            if not is_databricks_uri(tracking_uri):
                return func(*args, **kwargs)

            error_msg = f"{api_name} is not supported in Databricks environments."
            if alternative:
                error_msg += f" {alternative}"

            raise MlflowException(
                error_msg,
                error_code=INVALID_PARAMETER_VALUE,
            )

        return wrapper

    return decorator
