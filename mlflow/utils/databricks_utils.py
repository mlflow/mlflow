import functools
import getpass
import json
import logging
import os
import subprocess
import time
from sys import stderr
from typing import NamedTuple, Optional, TypeVar

import mlflow.utils
from mlflow.environment_variables import MLFLOW_TRACKING_URI
from mlflow.exceptions import MlflowException
from mlflow.legacy_databricks_cli.configure.provider import (
    DatabricksConfig,
    DatabricksConfigProvider,
    DefaultConfigProvider,
    ProfileConfigProvider,
    get_config,
    set_config_provider,
)
from mlflow.utils._spark_utils import _get_active_spark_session
from mlflow.utils.rest_utils import MlflowHostCreds
from mlflow.utils.uri import get_db_info_from_uri, is_databricks_uri

_logger = logging.getLogger(__name__)


_MODEL_DEPENDENCY_OAUTH_TOKEN_FILE_PATH = "/var/credentials-secret/model-dependencies-oauth-token"


def _use_repl_context_if_available(name):
    """Creates a decorator to insert a short circuit that returns the specified REPL context
    attribute if it's available.

    Args:
        name: Attribute name (e.g. "apiUrl").

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
                    return getattr(context, name)
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
        db_creds = get_databricks_host_creds(self.databricks_profile_url)
        self.db_utils.notebook.entry_point.putMlflowProperties(
            db_creds.host,
            db_creds.ignore_tls_verification,
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
    return _get_command_context().extraContext().get(context_key).get()


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

        task_context = TaskContext.get()
        if task_context:
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
        return acl_path_of_acl_root().startswith("/workspace")
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
    val = (
        os.environ.get("IS_IN_DB_MODEL_SERVING_ENV")
        # Checking the old env var name for backward compatibility. The env var was renamed once
        # to fix a model loading issue, but we still need to support it for a while.
        # TODO: Remove this once the new env var is fully rolled out.
        or os.environ.get("IS_IN_DATABRICKS_MODEL_SERVING_ENV")
        or "false"
    )
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


def is_in_databricks_serverless():
    dbr_version = get_databricks_runtime_version()
    return dbr_version and dbr_version.startswith("client.")


def is_dbfs_fuse_available():
    with open(os.devnull, "w") as devnull_stderr, open(os.devnull, "w") as devnull_stdout:
        try:
            return (
                subprocess.call(
                    ["mountpoint", "/dbfs"], stderr=devnull_stderr, stdout=devnull_stdout
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
    notebook_id = _get_property_from_spark_context("spark.databricks.notebook.id")
    if notebook_id is not None:
        return notebook_id
    acl_path = acl_path_of_acl_root()
    if acl_path.startswith("/workspace"):
        return acl_path.split("/")[-1]
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


@_use_repl_context_if_available("workspaceUrl")
def get_workspace_url():
    try:
        spark_session = _get_active_spark_session()
        if spark_session is not None:
            if workspace_url := spark_session.conf.get("spark.databricks.workspaceUrl", None):
                return f"https://{workspace_url}"
    except Exception:
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
        dbutils = _get_dbutils()
        if dbutils:
            workspace_id = dbutils.secrets.get(scope=profile, key=key_prefix + "-workspace-id")
            workspace_host = dbutils.secrets.get(scope=profile, key=key_prefix + "-host")
            return workspace_host, workspace_id
    return None, None


def _fail_malformed_databricks_auth(profile):
    raise MlflowException(
        "Got malformed Databricks CLI profile '%s'. Please make sure the "
        "Databricks CLI is properly configured as described at "
        "https://github.com/databricks/databricks-cli." % profile
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


def get_databricks_host_creds(server_uri=None):
    """
    Reads in configuration necessary to make HTTP requests to a Databricks server. This
    uses the Databricks CLI's ConfigProvider interface to load the DatabricksConfig object.
    If no Databricks CLI profile is found corresponding to the server URI, this function
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
        MlflowHostCreds which includes the hostname and authentication information necessary to
        talk to the Databricks server.
    """
    profile, path = get_db_info_from_uri(server_uri)
    config = ProfileConfigProvider(profile).get_config() if profile else get_config()
    # if a path is specified, that implies a Databricks tracking URI of the form:
    # databricks://profile-name/path-specifier
    if (not config or not config.host) and path:
        dbutils = _get_dbutils()
        if dbutils:
            # Prefix differentiates users and is provided as path information in the URI
            key_prefix = path
            host = dbutils.secrets.get(scope=profile, key=key_prefix + "-host")
            token = dbutils.secrets.get(scope=profile, key=key_prefix + "-token")
            if host and token:
                config = DatabricksConfig.from_token(host=host, token=token, insecure=False)
    if not config or not config.host:
        _fail_malformed_databricks_auth(profile)

    insecure = hasattr(config, "insecure") and config.insecure

    if config.username is not None and config.password is not None:
        return MlflowHostCreds(
            config.host,
            username=config.username,
            password=config.password,
            ignore_tls_verification=insecure,
        )
    elif config.token:
        return MlflowHostCreds(config.host, token=config.token, ignore_tls_verification=insecure)
    _fail_malformed_databricks_auth(profile)


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


def get_databricks_run_url(tracking_uri: str, run_id: str, artifact_path=None) -> Optional[str]:
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


def get_databricks_model_version_url(registry_uri: str, name: str, version: str) -> Optional[str]:
    """Obtains a Databricks URL corresponding to the specified Model Version.

    Args:
        tracking_uri: The URI of the Model Registry server containing the Model Version.
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

    def __init__(self, host: str, workspace_id: Optional[str] = None):
        self.host = host
        self.workspace_id = workspace_id

    @classmethod
    def from_environment(cls) -> Optional[DatabricksWorkspaceInfoType]:
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


def get_databricks_workspace_info_from_uri(tracking_uri: str) -> Optional[DatabricksWorkspaceInfo]:
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
    dbutils = _get_dbutils()
    if dbutils:
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


def _construct_databricks_run_url(
    host: str,
    experiment_id: str,
    run_id: str,
    workspace_id: Optional[str] = None,
    artifact_path: Optional[str] = None,
) -> str:
    run_url = host
    if workspace_id and workspace_id != "0":
        run_url += "?o=" + str(workspace_id)

    run_url += f"#mlflow/experiments/{experiment_id}/runs/{run_id}"

    if artifact_path is not None:
        run_url += f"/artifactPath/{artifact_path.lstrip('/')}"

    return run_url


def _construct_databricks_model_version_url(
    host: str, name: str, version: str, workspace_id: Optional[str] = None
) -> str:
    model_version_url = host
    if workspace_id and workspace_id != "0":
        model_version_url += "?o=" + str(workspace_id)

    model_version_url += f"#mlflow/models/{name}/versions/{version}"

    return model_version_url


def get_databricks_env_vars(tracking_uri):
    if not mlflow.utils.uri.is_databricks_uri(tracking_uri):
        return {}

    config = get_databricks_host_creds(tracking_uri)
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
    if config.ignore_tls_verification:
        env_vars["DATABRICKS_INSECURE"] = str(config.ignore_tls_verification)

    workspace_info = get_databricks_workspace_info_from_uri(tracking_uri)
    if workspace_info is not None:
        env_vars.update(workspace_info.to_environment())

    return env_vars


class DatabricksRuntimeVersion(NamedTuple):
    is_client_image: bool
    major: int
    minor: int

    @classmethod
    def parse(cls):
        dbr_version = get_databricks_runtime_version()
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


def _init_databricks_cli_config_provider(entry_point):
    """
    set a custom DatabricksConfigProvider with the hostname and token of the
    user running the current command (achieved by looking at
    PythonAccessibleThreadLocals.commandContext, via the already-exposed
    NotebookUtils.getContext API)
    """
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
                    print(  # noqa
                        "Unexpected internal error while constructing `DatabricksConfig` "
                        f"from REPL context: {e}",
                        file=stderr,
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

                # Fallback to the default behavior if we were unable to find a token.
                if api_token is None or api_url is None:
                    return DefaultConfigProvider().get_config()

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
                    print(  # noqa
                        "Unexpected internal error while constructing `DatabricksConfig` "
                        f"from REPL context: {e}",
                        file=stderr,
                    )
                # Invoking getContext() will attempt to find the credentials related to the
                # current command execution, so it's critical that we execute it on every
                # get_config().
                api_token_option = notebook_utils.getContext().apiToken()
                api_url_option = notebook_utils.getContext().apiUrl()
                ssl_trust_all = entry_point.getDriverConf().workflowSslTrustAll()

                # Fallback to the default behavior if we were unable to find a token.
                if not api_token_option.isDefined() or not api_url_option.isDefined():
                    return DefaultConfigProvider().get_config()

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

                # Fallback to the default behavior if we were unable to find a token.
                if not api_token_option.isDefined() or not api_url_option.isDefined():
                    return DefaultConfigProvider().get_config()

                return DatabricksConfig.from_token(
                    host=api_url_option.get(), token=api_token_option.get(), insecure=ssl_trust_all
                )

    set_config_provider(DynamicConfigProvider())


if is_in_databricks_runtime():
    try:
        dbutils = _get_dbutils()
        _init_databricks_cli_config_provider(dbutils.entry_point)
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
