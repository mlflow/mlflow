"""
This module defines environment variables used in MLflow.
"""
import os
from pathlib import Path


class _EnvironmentVariable:
    """
    Represents an environment variable.
    """

    def __init__(self, name, type_, default):
        self.name = name
        self.type = type_
        self.default = default

    @property
    def defined(self):
        return self.name in os.environ

    def get_raw(self):
        return os.getenv(self.name)

    def set(self, value):
        os.environ[self.name] = str(value)

    def unset(self):
        os.environ.pop(self.name, None)

    def get(self):
        """
        Reads the value of the environment variable if it exists and converts it to the desired
        type. Otherwise, returns the default value.
        """
        if (val := self.get_raw()) is not None:
            try:
                return self.type(val)
            except Exception as e:
                raise ValueError(f"Failed to convert {val!r} to {self.type} for {self.name}: {e}")
        return self.default

    def __str__(self):
        return f"{self.name} (default: {self.default}, type: {self.type.__name__})"

    def __repr__(self):
        return repr(self.name)

    def __format__(self, format_spec: str) -> str:
        return self.name.__format__(format_spec)


class _BooleanEnvironmentVariable(_EnvironmentVariable):
    """
    Represents a boolean environment variable.
    """

    def __init__(self, name, default):
        # `default not in [True, False, None]` doesn't work because `1 in [True]`
        # (or `0 in [False]`) returns True.
        if not (default is True or default is False or default is None):
            raise ValueError(f"{name} default value must be one of [True, False, None]")
        super().__init__(name, bool, default)

    def get(self):
        if not self.defined:
            return self.default

        val = os.getenv(self.name)
        lowercased = val.lower()
        if lowercased not in ["true", "false", "1", "0"]:
            raise ValueError(
                f"{self.name} value must be one of ['true', 'false', '1', '0'] (case-insensitive), "
                f"but got {val}"
            )
        return lowercased in ["true", "1"]


#: Specifies the tracking URI.
#: (default: ``None``)
MLFLOW_TRACKING_URI = _EnvironmentVariable("MLFLOW_TRACKING_URI", str, None)

#: Specifies the registry URI.
#: (default: ``None``)
MLFLOW_REGISTRY_URI = _EnvironmentVariable("MLFLOW_REGISTRY_URI", str, None)

#: Specifies the ``dfs_tmpdir`` parameter to use for ``mlflow.spark.save_model``,
#: ``mlflow.spark.log_model`` and ``mlflow.spark.load_model``. See
#: https://www.mlflow.org/docs/latest/python_api/mlflow.spark.html#mlflow.spark.save_model
#: for more information.
#: (default: ``/tmp/mlflow``)
MLFLOW_DFS_TMP = _EnvironmentVariable("MLFLOW_DFS_TMP", str, "/tmp/mlflow")

#: Specifies the maximum number of retries for MLflow HTTP requests
#: (default: ``5``)
MLFLOW_HTTP_REQUEST_MAX_RETRIES = _EnvironmentVariable("MLFLOW_HTTP_REQUEST_MAX_RETRIES", int, 5)

#: Specifies the backoff increase factor between MLflow HTTP request failures
#: (default: ``2``)
MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR = _EnvironmentVariable(
    "MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", int, 2
)

#: Specifies the backoff jitter between MLflow HTTP request failures
#: (default: ``1.0``)
MLFLOW_HTTP_REQUEST_BACKOFF_JITTER = _EnvironmentVariable(
    "MLFLOW_HTTP_REQUEST_BACKOFF_JITTER", float, 1.0
)

#: Specifies the timeout in seconds for MLflow HTTP requests
#: (default: ``120``)
MLFLOW_HTTP_REQUEST_TIMEOUT = _EnvironmentVariable("MLFLOW_HTTP_REQUEST_TIMEOUT", int, 120)

#: Specifies whether MLflow HTTP requests should be signed using AWS signature V4. It will overwrite
#: (default: ``False``). When set, it will overwrite the "Authorization" HTTP header.
#: See https://docs.aws.amazon.com/general/latest/gr/signature-version-4.html for more information.
MLFLOW_TRACKING_AWS_SIGV4 = _BooleanEnvironmentVariable("MLFLOW_TRACKING_AWS_SIGV4", False)

#: Specifies the auth provider to sign the MLflow HTTP request
#: (default: ``None``). When set, it will overwrite the "Authorization" HTTP header.
MLFLOW_TRACKING_AUTH = _EnvironmentVariable("MLFLOW_TRACKING_AUTH", str, None)

#: Specifies the chunk size to use when downloading a file from GCS
#: (default: ``None``). If None, the chunk size is automatically determined by the
#: ``google-cloud-storage`` package.
MLFLOW_GCS_DOWNLOAD_CHUNK_SIZE = _EnvironmentVariable("MLFLOW_GCS_DOWNLOAD_CHUNK_SIZE", int, None)

#: Specifies the chunk size to use when uploading a file to GCS.
#: (default: ``None``). If None, the chunk size is automatically determined by the
#: ``google-cloud-storage`` package.
MLFLOW_GCS_UPLOAD_CHUNK_SIZE = _EnvironmentVariable("MLFLOW_GCS_UPLOAD_CHUNK_SIZE", int, None)

#: (Deprecated, please use ``MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT``)
#: Specifies the default timeout to use when downloading/uploading a file from/to GCS
#: (default: ``None``). If None, ``google.cloud.storage.constants._DEFAULT_TIMEOUT`` is used.
MLFLOW_GCS_DEFAULT_TIMEOUT = _EnvironmentVariable("MLFLOW_GCS_DEFAULT_TIMEOUT", int, None)

#: Specifies whether to disable model logging and loading via mlflowdbfs.
#: (default: ``None``)
_DISABLE_MLFLOWDBFS = _EnvironmentVariable("DISABLE_MLFLOWDBFS", str, None)

#: Specifies the S3 endpoint URL to use for S3 artifact operations.
#: (default: ``None``)
MLFLOW_S3_ENDPOINT_URL = _EnvironmentVariable("MLFLOW_S3_ENDPOINT_URL", str, None)

#: Specifies whether or not to skip TLS certificate verification for S3 artifact operations.
#: (default: ``False``)
MLFLOW_S3_IGNORE_TLS = _BooleanEnvironmentVariable("MLFLOW_S3_IGNORE_TLS", False)

#: Specifies extra arguments for S3 artifact uploads.
#: (default: ``None``)
MLFLOW_S3_UPLOAD_EXTRA_ARGS = _EnvironmentVariable("MLFLOW_S3_UPLOAD_EXTRA_ARGS", str, None)

#: Specifies the location of a Kerberos ticket cache to use for HDFS artifact operations.
#: (default: ``None``)
MLFLOW_KERBEROS_TICKET_CACHE = _EnvironmentVariable("MLFLOW_KERBEROS_TICKET_CACHE", str, None)

#: Specifies a Kerberos user for HDFS artifact operations.
#: (default: ``None``)
MLFLOW_KERBEROS_USER = _EnvironmentVariable("MLFLOW_KERBEROS_USER", str, None)

#: Specifies extra pyarrow configurations for HDFS artifact operations.
#: (default: ``None``)
MLFLOW_PYARROW_EXTRA_CONF = _EnvironmentVariable("MLFLOW_PYARROW_EXTRA_CONF", str, None)

#: Specifies the ``pool_size`` parameter to use for ``sqlalchemy.create_engine`` in the SQLAlchemy
#: tracking store. See https://docs.sqlalchemy.org/en/14/core/engines.html#sqlalchemy.create_engine.params.pool_size
#: for more information.
#: (default: ``None``)
MLFLOW_SQLALCHEMYSTORE_POOL_SIZE = _EnvironmentVariable(
    "MLFLOW_SQLALCHEMYSTORE_POOL_SIZE", int, None
)

#: Specifies the ``pool_recycle`` parameter to use for ``sqlalchemy.create_engine`` in the
#: SQLAlchemy tracking store. See https://docs.sqlalchemy.org/en/14/core/engines.html#sqlalchemy.create_engine.params.pool_recycle
#: for more information.
#: (default: ``None``)
MLFLOW_SQLALCHEMYSTORE_POOL_RECYCLE = _EnvironmentVariable(
    "MLFLOW_SQLALCHEMYSTORE_POOL_RECYCLE", int, None
)

#: Specifies the ``max_overflow`` parameter to use for ``sqlalchemy.create_engine`` in the
#: SQLAlchemy tracking store. See https://docs.sqlalchemy.org/en/14/core/engines.html#sqlalchemy.create_engine.params.max_overflow
#: for more information.
#: (default: ``None``)
MLFLOW_SQLALCHEMYSTORE_MAX_OVERFLOW = _EnvironmentVariable(
    "MLFLOW_SQLALCHEMYSTORE_MAX_OVERFLOW", int, None
)

#: Specifies the ``echo`` parameter to use for ``sqlalchemy.create_engine`` in the
#: SQLAlchemy tracking store. See https://docs.sqlalchemy.org/en/14/core/engines.html#sqlalchemy.create_engine.params.echo
#: for more information.
#: (default: ``False``)
MLFLOW_SQLALCHEMYSTORE_ECHO = _BooleanEnvironmentVariable("MLFLOW_SQLALCHEMYSTORE_ECHO", False)

#: Specifies whether or not to print a warning when `--env-manager=conda` is specified.
#: (default: ``False``)
MLFLOW_DISABLE_ENV_MANAGER_CONDA_WARNING = _BooleanEnvironmentVariable(
    "MLFLOW_DISABLE_ENV_MANAGER_CONDA_WARNING", False
)
#: Specifies the ``poolclass`` parameter to use for ``sqlalchemy.create_engine`` in the
#: SQLAlchemy tracking store. See https://docs.sqlalchemy.org/en/14/core/engines.html#sqlalchemy.create_engine.params.poolclass
#: for more information.
#: (default: ``None``)
MLFLOW_SQLALCHEMYSTORE_POOLCLASS = _EnvironmentVariable(
    "MLFLOW_SQLALCHEMYSTORE_POOLCLASS", str, None
)

#: Specifies the ``timeout_seconds`` for MLflow Model dependency inference operations.
#: (default: ``120``)
MLFLOW_REQUIREMENTS_INFERENCE_TIMEOUT = _EnvironmentVariable(
    "MLFLOW_REQUIREMENTS_INFERENCE_TIMEOUT", int, 120
)

#: Specifies the MLflow Model Scoring server request timeout in seconds
#: (default: ``60``)
MLFLOW_SCORING_SERVER_REQUEST_TIMEOUT = _EnvironmentVariable(
    "MLFLOW_SCORING_SERVER_REQUEST_TIMEOUT", int, 60
)

#: (Experimental, may be changed or removed)
#: Specifies the timeout to use when uploading or downloading a file
#: (default: ``None``). If None, individual artifact stores will choose defaults.
MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT = _EnvironmentVariable(
    "MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT", int, None
)

#: Specifies the device intended for use in the predict function - can be used
#: to override behavior where the GPU is used by default when available by
#: setting this environment variable to be ``cpu``. Currently, this
#: variable is only supported for the MLflow PyTorch and HuggingFace flavors.
#: For the HuggingFace flavor, note that device must be parseable as an integer.
MLFLOW_DEFAULT_PREDICTION_DEVICE = _EnvironmentVariable(
    "MLFLOW_DEFAULT_PREDICTION_DEVICE", str, None
)

#: Specifies to Huggingface whether to use the automatic device placement logic of
# HuggingFace accelerate. If it's set to false, the low_cpu_mem_usage flag will not be
# set to True and device_map will not be set to "auto".
MLFLOW_HUGGINGFACE_DISABLE_ACCELERATE_FEATURES = _BooleanEnvironmentVariable(
    "MLFLOW_DISABLE_HUGGINGFACE_ACCELERATE_FEATURES", False
)

#: Specifies to Huggingface whether to use the automatic device placement logic of
# HuggingFace accelerate. If it's set to false, the low_cpu_mem_usage flag will not be
# set to True and device_map will not be set to "auto".
MLFLOW_HUGGINGFACE_USE_DEVICE_MAP = _BooleanEnvironmentVariable(
    "MLFLOW_HUGGINGFACE_USE_DEVICE_MAP", True
)

#: Specifies to Huggingface to use the automatic device placement logic of HuggingFace accelerate.
#: This can be set to values supported by the version of HuggingFace Accelerate being installed.
MLFLOW_HUGGINGFACE_DEVICE_MAP_STRATEGY = _EnvironmentVariable(
    "MLFLOW_HUGGINGFACE_DEVICE_MAP_STRATEGY", str, "auto"
)

#: Specifies to Huggingface to use the low_cpu_mem_usage flag powered by HuggingFace accelerate.
#: If it's set to false, the low_cpu_mem_usage flag will be set to False.
MLFLOW_HUGGINGFACE_USE_LOW_CPU_MEM_USAGE = _BooleanEnvironmentVariable(
    "MLFLOW_HUGGINGFACE_USE_LOW_CPU_MEM_USAGE", True
)

#: Specifies the max_shard_size to use when mlflow transformers flavor saves the model checkpoint.
#: This can be set to override the 500MB default.
MLFLOW_HUGGINGFACE_MODEL_MAX_SHARD_SIZE = _EnvironmentVariable(
    "MLFLOW_HUGGINGFACE_MODEL_MAX_SHARD_SIZE", str, "500MB"
)

#: Specifies whether or not to allow using a file URI as a model version source.
#: Please be aware that setting this environment variable to True is potentially risky
#: because it can allow access to arbitrary files on the specified filesystem
#: (default: ``False``).
MLFLOW_ALLOW_FILE_URI_AS_MODEL_VERSION_SOURCE = _BooleanEnvironmentVariable(
    "MLFLOW_ALLOW_FILE_URI_AS_MODEL_VERSION_SOURCE", False
)


#: Specifies the name of the Databricks secret scope to use for storing OpenAI API keys.
MLFLOW_OPENAI_SECRET_SCOPE = _EnvironmentVariable("MLFLOW_OPENAI_SECRET_SCOPE", str, None)

#: Specifier whether or not to retry OpenAI API calls.
MLFLOW_OPENAI_RETRIES_ENABLED = _BooleanEnvironmentVariable("MLFLOW_OPENAI_RETRIES_ENABLED", True)

#: (Experimental, may be changed or removed)
#: Specifies the download options to be used by pip wheel when `add_libraries_to_model` is used to
#: create and log model dependencies as model artifacts. The default behavior only uses dependency
#: binaries and no source packages.
#: (default: ``--only-binary=:all:``).
MLFLOW_WHEELED_MODEL_PIP_DOWNLOAD_OPTIONS = _EnvironmentVariable(
    "MLFLOW_WHEELED_MODEL_PIP_DOWNLOAD_OPTIONS", str, "--only-binary=:all:"
)

# Specifies whether or not to use multipart download when downloading a large file on Databricks.
MLFLOW_ENABLE_MULTIPART_DOWNLOAD = _BooleanEnvironmentVariable(
    "MLFLOW_ENABLE_MULTIPART_DOWNLOAD", True
)

# Specifies whether or not to use multipart upload when uploading large artifacts.
MLFLOW_ENABLE_MULTIPART_UPLOAD = _BooleanEnvironmentVariable("MLFLOW_ENABLE_MULTIPART_UPLOAD", True)

#: Specifies whether or not to use multipart upload for proxied artifact access.
#: (default: ``False``)
MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD = _BooleanEnvironmentVariable(
    "MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD", False
)

#: Private environment variable that's set to ``True`` while running tests.
_MLFLOW_TESTING = _BooleanEnvironmentVariable("MLFLOW_TESTING", False)

#: Specifies the username used to authenticate with a tracking server.
#: (default: ``None``)
MLFLOW_TRACKING_USERNAME = _EnvironmentVariable("MLFLOW_TRACKING_USERNAME", str, None)

#: Specifies the password used to authenticate with a tracking server.
#: (default: ``None``)
MLFLOW_TRACKING_PASSWORD = _EnvironmentVariable("MLFLOW_TRACKING_PASSWORD", str, None)

#: Specifies and takes precedence for setting the basic/bearer auth on http requests.
#: (default: ``None``)
MLFLOW_TRACKING_TOKEN = _EnvironmentVariable("MLFLOW_TRACKING_TOKEN", str, None)

#: Specifies whether to verify TLS connection in ``requests.request`` function,
#: see https://requests.readthedocs.io/en/master/api/
#: (default: ``False``).
MLFLOW_TRACKING_INSECURE_TLS = _BooleanEnvironmentVariable("MLFLOW_TRACKING_INSECURE_TLS", False)

#: Sets the ``verify`` param in ``requests.request`` function,
#: see https://requests.readthedocs.io/en/master/api/
#: (default: ``None``)
MLFLOW_TRACKING_SERVER_CERT_PATH = _EnvironmentVariable(
    "MLFLOW_TRACKING_SERVER_CERT_PATH", str, None
)

#: Sets the ``cert`` param in ``requests.request`` function,
#: see https://requests.readthedocs.io/en/master/api/
#: (default: ``None``)
MLFLOW_TRACKING_CLIENT_CERT_PATH = _EnvironmentVariable(
    "MLFLOW_TRACKING_CLIENT_CERT_PATH", str, None
)

#: Specified the ID of the run to log data to.
#: (default: ``None``)
MLFLOW_RUN_ID = _EnvironmentVariable("MLFLOW_RUN_ID", str, None)

#: Specifies the default root directory for tracking `FileStore`.
#: (default: ``None``)
MLFLOW_TRACKING_DIR = _EnvironmentVariable("MLFLOW_TRACKING_DIR", str, None)

#: Specifies the default root directory for registry `FileStore`.
#: (default: ``None``)
MLFLOW_REGISTRY_DIR = _EnvironmentVariable("MLFLOW_REGISTRY_DIR", str, None)

#: Specifies the default experiment ID to create run to.
#: (default: ``None``)
MLFLOW_EXPERIMENT_ID = _EnvironmentVariable("MLFLOW_EXPERIMENT_ID", str, None)

#: Specifies the default experiment name to create run to.
#: (default: ``None``)
MLFLOW_EXPERIMENT_NAME = _EnvironmentVariable("MLFLOW_EXPERIMENT_NAME", str, None)

#: Specified the path to the configuration file for MLflow Authentication.
#: (default: ``None``)
MLFLOW_AUTH_CONFIG_PATH = _EnvironmentVariable("MLFLOW_AUTH_CONFIG_PATH", str, None)

#: Specifies the root directory to create Python virtual environments in.
#: (default: ``~/.mlflow/envs``)
MLFLOW_ENV_ROOT = _EnvironmentVariable(
    "MLFLOW_ENV_ROOT", str, str(Path.home().joinpath(".mlflow", "envs"))
)

#: Specifies whether or not to use DBFS FUSE mount to store artifacts on Databricks
#: (default: ``False``)
MLFLOW_ENABLE_DBFS_FUSE_ARTIFACT_REPO = _BooleanEnvironmentVariable(
    "MLFLOW_ENABLE_DBFS_FUSE_ARTIFACT_REPO", True
)

#: Private environment variable that should be set to ``True`` when running autologging tests.
#: (default: ``False``)
_MLFLOW_AUTOLOGGING_TESTING = _BooleanEnvironmentVariable("MLFLOW_AUTOLOGGING_TESTING", False)

#: (Experimental, may be changed or removed)
#: Specifies the uri of a MLflow Gateway Server instance to be used with the Gateway Client APIs
#: (default: ``None``)
MLFLOW_GATEWAY_URI = _EnvironmentVariable("MLFLOW_GATEWAY_URI", str, None)

#: (Experimental, may be changed or removed)
#: Specifies the uri of a MLflow Deployments Server instance to be used with the Deployments
#: Client APIs
#: (default: ``None``)
MLFLOW_DEPLOYMENTS_TARGET = _EnvironmentVariable("MLFLOW_DEPLOYMENTS_TARGET", str, None)

#: Specifies the path of the config file for MLflow AI Gateway.
#: (default: ``None``)
MLFLOW_GATEWAY_CONFIG = _EnvironmentVariable("MLFLOW_GATEWAY_CONFIG", str, None)

#: Specifies the path of the config file for the MLflow Deployments server.
#: (default: ``None``)
MLFLOW_DEPLOYMENTS_CONFIG = _EnvironmentVariable("MLFLOW_DEPLOYMENTS_CONFIG", str, None)

#: Specifies whether to display the progress bar when uploading/downloading artifacts.
#: (default: ``True``)
MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR = _BooleanEnvironmentVariable(
    "MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", True
)

#: Specifies the conda home directory to use.
#: (default: ``conda``)
MLFLOW_CONDA_HOME = _EnvironmentVariable("MLFLOW_CONDA_HOME", str, None)

#: Specifies the name of the command to use when creating the environments.
#: For example, let's say we want to use mamba (https://github.com/mamba-org/mamba)
#: instead of conda to create environments.
#: Then: > conda install mamba -n base -c conda-forge
#: If not set, use the same as conda_path
#: (default: ``conda``)
MLFLOW_CONDA_CREATE_ENV_CMD = _EnvironmentVariable("MLFLOW_CONDA_CREATE_ENV_CMD", str, "conda")

#: Specifies the execution directory for recipes.
#: (default: ``None``)
MLFLOW_RECIPES_EXECUTION_DIRECTORY = _EnvironmentVariable(
    "MLFLOW_RECIPES_EXECUTION_DIRECTORY", str, None
)

#: Specifies the target step to execute for recipes.
#: (default: ``None``)
MLFLOW_RECIPES_EXECUTION_TARGET_STEP_NAME = _EnvironmentVariable(
    "MLFLOW_RECIPES_EXECUTION_TARGET_STEP_NAME", str, None
)

#: Specifies the flavor to serve in the scoring server.
#: (default ``None``)
MLFLOW_DEPLOYMENT_FLAVOR_NAME = _EnvironmentVariable("MLFLOW_DEPLOYMENT_FLAVOR_NAME", str, None)

#: Specifies the profile to use for recipes.
#: (default: ``None``)
MLFLOW_RECIPES_PROFILE = _EnvironmentVariable("MLFLOW_RECIPES_PROFILE", str, None)

#: Specifies the MLflow Run context
#: (default: ``None``)
MLFLOW_RUN_CONTEXT = _EnvironmentVariable("MLFLOW_RUN_CONTEXT", str, None)

#: Specifies the URL of the ECR-hosted Docker image a model is deployed into for SageMaker.
# (default: ``None``)
MLFLOW_SAGEMAKER_DEPLOY_IMG_URL = _EnvironmentVariable("MLFLOW_SAGEMAKER_DEPLOY_IMG_URL", str, None)

#: Specifies whether to disable creating a new conda environment for `mlflow models build-docker`.
#: (default: ``False``)
MLFLOW_DISABLE_ENV_CREATION = _BooleanEnvironmentVariable("MLFLOW_DISABLE_ENV_CREATION", False)

#: Specifies the timeout value for downloading chunks of mlflow artifacts.
#: (default: ``300``)
MLFLOW_DOWNLOAD_CHUNK_TIMEOUT = _EnvironmentVariable("MLFLOW_DOWNLOAD_CHUNK_TIMEOUT", int, 300)

#: Specifies if system metrics logging should be enabled.
MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING = _BooleanEnvironmentVariable(
    "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", False
)

#: Specifies the sampling interval for system metrics logging.
MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL = _EnvironmentVariable(
    "MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL", float, None
)

#: Specifies the number of samples before logging system metrics.
MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING = _EnvironmentVariable(
    "MLFLOW_SYSTEM_METRICS_SAMPLES_BEFORE_LOGGING", int, None
)

# Private environment variable to specify the number of chunk download retries for multipart
# download.
_MLFLOW_MPD_NUM_RETRIES = _EnvironmentVariable("_MLFLOW_MPD_NUM_RETRIES", int, 3)

# Private environment variable to specify the interval between chunk download retries for multipart
# download.
_MLFLOW_MPD_RETRY_INTERVAL_SECONDS = _EnvironmentVariable(
    "_MLFLOW_MPD_RETRY_INTERVAL_SECONDS", int, 1
)

#: Specifies the minimum file size in bytes to use multipart upload when logging artifacts
#: (default: ``524_288_000`` (500 MB))
MLFLOW_MULTIPART_UPLOAD_MINIMUM_FILE_SIZE = _EnvironmentVariable(
    "MLFLOW_MULTIPART_UPLOAD_MINIMUM_FILE_SIZE", int, 500 * 1024**2
)

#: Specifies the chunk size in bytes to use when performing multipart upload
#: (default: ``104_857_60`` (10 MB))
MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE = _EnvironmentVariable(
    "MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE", int, 10 * 1024**2
)

#: Specifies the chunk size in bytes to use when performing multipart download
#: (default: ``104_857_600`` (100 MB))
MLFLOW_MULTIPART_DOWNLOAD_CHUNK_SIZE = _EnvironmentVariable(
    "MLFLOW_MULTIPART_DOWNLOAD_CHUNK_SIZE", int, 100 * 1024**2
)

#: Specifies whether or not to allow the MLflow server to follow redirects when
#: making HTTP requests. If set to False, the server will throw an exception if it
#: encounters a redirect response.
#: (default: ``True``)
MLFLOW_ALLOW_HTTP_REDIRECTS = _BooleanEnvironmentVariable("MLFLOW_ALLOW_HTTP_REDIRECTS", True)
