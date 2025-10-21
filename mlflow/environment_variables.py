"""
This module defines environment variables used in MLflow.
MLflow's environment variables adhere to the following naming conventions:
- Public variables: environment variable names begin with `MLFLOW_`
- Internal-use variables: For variables used only internally, names start with `_MLFLOW_`
"""

import os
import warnings
from pathlib import Path


class _EnvironmentVariable:
    """
    Represents an environment variable.
    """

    def __init__(self, name, type_, default):
        if type_ == bool and not isinstance(self, _BooleanEnvironmentVariable):
            raise ValueError("Use _BooleanEnvironmentVariable instead for boolean variables")
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

    def is_set(self):
        return self.name in os.environ

    def get(self):
        """
        Reads the value of the environment variable if it exists and converts it to the desired
        type. Otherwise, returns the default value.
        """
        if (val := self.get_raw()) is not None:
            try:
                return self.type(val)
            except Exception as e:
                raise ValueError(f"Failed to convert {val!r} for {self.name}: {e}")
        return self.default

    def __str__(self):
        return f"{self.name} (default: {self.default})"

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
        # TODO: Remove this block in MLflow 3.2.0
        if self.name == MLFLOW_CONFIGURE_LOGGING.name and (
            val := os.getenv("MLFLOW_LOGGING_CONFIGURE_LOGGING")
        ):
            warnings.warn(
                "Environment variable MLFLOW_LOGGING_CONFIGURE_LOGGING is deprecated and will be "
                f"removed in a future release. Please use {MLFLOW_CONFIGURE_LOGGING.name} instead.",
                FutureWarning,
                stacklevel=2,
            )
            return val.lower() in ["true", "1"]

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

#: Specifies the maximum number of retries with exponential backoff for MLflow HTTP requests
#: (default: ``7``)
MLFLOW_HTTP_REQUEST_MAX_RETRIES = _EnvironmentVariable(
    "MLFLOW_HTTP_REQUEST_MAX_RETRIES",
    int,
    # Important: It's common for MLflow backends to rate limit requests for more than 1 minute.
    # To remain resilient to rate limiting, the MLflow client needs to retry for more than 1
    # minute. Assuming 2 seconds per retry, 7 retries with backoff will take ~ 4 minutes,
    # which is appropriate for most rate limiting scenarios
    7,
)

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

#: Specifies the timeout in seconds for MLflow deployment client HTTP requests
#: (non-predict operations). This is separate from MLFLOW_HTTP_REQUEST_TIMEOUT to allow
#: longer timeouts for LLM calls (default: ``300``)
MLFLOW_DEPLOYMENT_CLIENT_HTTP_REQUEST_TIMEOUT = _EnvironmentVariable(
    "MLFLOW_DEPLOYMENT_CLIENT_HTTP_REQUEST_TIMEOUT", int, 300
)

#: Specifies whether to respect Retry-After header on status codes defined as
#: Retry.RETRY_AFTER_STATUS_CODES or not for MLflow HTTP request
#: (default: ``True``)
MLFLOW_HTTP_RESPECT_RETRY_AFTER_HEADER = _BooleanEnvironmentVariable(
    "MLFLOW_HTTP_RESPECT_RETRY_AFTER_HEADER", True
)

#: Internal-only configuration that sets an upper bound to the allowable maximum
#: retries for HTTP requests
#: (default: ``10``)
_MLFLOW_HTTP_REQUEST_MAX_RETRIES_LIMIT = _EnvironmentVariable(
    "_MLFLOW_HTTP_REQUEST_MAX_RETRIES_LIMIT", int, 10
)

#: Internal-only configuration that sets the upper bound for an HTTP backoff_factor
#: (default: ``120``)
_MLFLOW_HTTP_REQUEST_MAX_BACKOFF_FACTOR_LIMIT = _EnvironmentVariable(
    "_MLFLOW_HTTP_REQUEST_MAX_BACKOFF_FACTOR_LIMIT", int, 120
)

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

#: Specifies the timeout for model inference with input example(s) when logging/saving a model.
#: MLflow runs a few inference requests against the model to infer model signature and pip
#: requirements. Sometimes the prediction hangs for a long time, especially for a large model.
#: This timeout limits the allowable time for performing a prediction for signature inference
#: and will abort the prediction, falling back to the default signature and pip requirements.
MLFLOW_INPUT_EXAMPLE_INFERENCE_TIMEOUT = _EnvironmentVariable(
    "MLFLOW_INPUT_EXAMPLE_INFERENCE_TIMEOUT", int, 180
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
# set to True and device_map will not be set to "auto". Default to False.
MLFLOW_HUGGINGFACE_USE_DEVICE_MAP = _BooleanEnvironmentVariable(
    "MLFLOW_HUGGINGFACE_USE_DEVICE_MAP", False
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

#: Specifies the name of the Databricks secret scope to use for storing OpenAI API keys.
MLFLOW_OPENAI_SECRET_SCOPE = _EnvironmentVariable("MLFLOW_OPENAI_SECRET_SCOPE", str, None)

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

#: Specifies and takes precedence for setting the UC OSS basic/bearer auth on http requests.
#: (default: ``None``)
MLFLOW_UC_OSS_TOKEN = _EnvironmentVariable("MLFLOW_UC_OSS_TOKEN", str, None)

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

#: Specifies whether or not to use UC Volume FUSE mount to store artifacts on Databricks
#: (default: ``True``)
MLFLOW_ENABLE_UC_VOLUME_FUSE_ARTIFACT_REPO = _BooleanEnvironmentVariable(
    "MLFLOW_ENABLE_UC_VOLUME_FUSE_ARTIFACT_REPO", True
)

#: Private environment variable that should be set to ``True`` when running autologging tests.
#: (default: ``False``)
_MLFLOW_AUTOLOGGING_TESTING = _BooleanEnvironmentVariable("MLFLOW_AUTOLOGGING_TESTING", False)

#: (Experimental, may be changed or removed)
#: Specifies the uri of a MLflow Gateway Server instance to be used with the Gateway Client APIs
#: (default: ``None``)
MLFLOW_GATEWAY_URI = _EnvironmentVariable("MLFLOW_GATEWAY_URI", str, None)

#: (Experimental, may be changed or removed)
#: Specifies the uri of an MLflow AI Gateway instance to be used with the Deployments
#: Client APIs
#: (default: ``None``)
MLFLOW_DEPLOYMENTS_TARGET = _EnvironmentVariable("MLFLOW_DEPLOYMENTS_TARGET", str, None)

#: Specifies the path of the config file for MLflow AI Gateway.
#: (default: ``None``)
MLFLOW_GATEWAY_CONFIG = _EnvironmentVariable("MLFLOW_GATEWAY_CONFIG", str, None)

#: Specifies the path of the config file for MLflow AI Gateway.
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

#: Specifies the flavor to serve in the scoring server.
#: (default ``None``)
MLFLOW_DEPLOYMENT_FLAVOR_NAME = _EnvironmentVariable("MLFLOW_DEPLOYMENT_FLAVOR_NAME", str, None)

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

#: Specifies the node id of system metrics logging. This is useful in multi-node (distributed
#: training) setup.
MLFLOW_SYSTEM_METRICS_NODE_ID = _EnvironmentVariable("MLFLOW_SYSTEM_METRICS_NODE_ID", str, None)


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

#: Specifies the minimum file size in bytes to use multipart download when downloading artifacts
#: (default: ``524_288_000`` (500 MB))
MLFLOW_MULTIPART_DOWNLOAD_MINIMUM_FILE_SIZE = _EnvironmentVariable(
    "MLFLOW_MULTIPART_DOWNLOAD_MINIMUM_FILE_SIZE", int, 500 * 1024**2
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

#: Timeout for a SINGLE HTTP request to a deployment endpoint (in seconds).
#: This controls how long ONE individual predict/predict_stream request can take before timing out.
#: If your model inference takes longer than this (e.g., long-running agent queries that take
#: several minutes), you MUST increase this value to allow the single request to complete.
#: For example, if your longest query takes 5 minutes, set this to at least 300 seconds.
#: Used within the `predict` and `predict_stream` APIs.
#: (default: ``120``)
MLFLOW_DEPLOYMENT_PREDICT_TIMEOUT = _EnvironmentVariable(
    "MLFLOW_DEPLOYMENT_PREDICT_TIMEOUT", int, 120
)

#: TOTAL time limit for ALL retry attempts combined (in seconds).
#: This controls how long the client will keep retrying failed requests across ALL attempts
#: before giving up entirely. This is SEPARATE from MLFLOW_DEPLOYMENT_PREDICT_TIMEOUT, which
#: controls how long a SINGLE request can run, while this variable controls the TOTAL time
#: for ALL retries. For long-running operations that may also experience transient failures,
#: ensure BOTH timeouts are set appropriately. This value should be greater than or equal to
#: MLFLOW_DEPLOYMENT_PREDICT_TIMEOUT.
#: (default: ``600``)
MLFLOW_DEPLOYMENT_PREDICT_TOTAL_TIMEOUT = _EnvironmentVariable(
    "MLFLOW_DEPLOYMENT_PREDICT_TOTAL_TIMEOUT", int, 600
)

MLFLOW_GATEWAY_RATE_LIMITS_STORAGE_URI = _EnvironmentVariable(
    "MLFLOW_GATEWAY_RATE_LIMITS_STORAGE_URI", str, None
)

#: If True, MLflow fluent logging APIs, e.g., `mlflow.log_metric` will log asynchronously.
MLFLOW_ENABLE_ASYNC_LOGGING = _BooleanEnvironmentVariable("MLFLOW_ENABLE_ASYNC_LOGGING", False)

#: Number of workers in the thread pool used for asynchronous logging, defaults to 10.
MLFLOW_ASYNC_LOGGING_THREADPOOL_SIZE = _EnvironmentVariable(
    "MLFLOW_ASYNC_LOGGING_THREADPOOL_SIZE", int, 10
)

#: Specifies whether or not to have mlflow configure logging on import.
#: If set to True, mlflow will configure ``mlflow.<module_name>`` loggers with
#: logging handlers and formatters.
#: (default: ``True``)
MLFLOW_CONFIGURE_LOGGING = _BooleanEnvironmentVariable("MLFLOW_CONFIGURE_LOGGING", True)

#: If set to True, the following entities will be truncated to their maximum length:
#: - Param value
#: - Tag value
#: If set to False, an exception will be raised if the length of the entity exceeds the maximum
#: length.
#: (default: ``True``)
MLFLOW_TRUNCATE_LONG_VALUES = _BooleanEnvironmentVariable("MLFLOW_TRUNCATE_LONG_VALUES", True)

# Whether to run slow tests with pytest. Default to False in normal runs,
# but set to True in the weekly slow test jobs.
_MLFLOW_RUN_SLOW_TESTS = _BooleanEnvironmentVariable("MLFLOW_RUN_SLOW_TESTS", False)

#: The OpenJDK version to install in the Docker image used for MLflow models.
#: (default: ``11``)
MLFLOW_DOCKER_OPENJDK_VERSION = _EnvironmentVariable("MLFLOW_DOCKER_OPENJDK_VERSION", str, "11")


#: How long a trace can be "in-progress". When this is set to a positive value and a trace is
#: not completed within this time, it will be automatically halted and exported to the specified
#: backend destination with status "ERROR".
MLFLOW_TRACE_TIMEOUT_SECONDS = _EnvironmentVariable("MLFLOW_TRACE_TIMEOUT_SECONDS", int, None)

#: How frequently to check for timed-out traces. For example, if this is set to 10, MLflow will
#: check for timed-out traces every 10 seconds (in a background worker) and halt any traces that
#: have exceeded the timeout. This is only effective if MLFLOW_TRACE_TIMEOUT_SECONDS is set to a
#: positive value.
MLFLOW_TRACE_TIMEOUT_CHECK_INTERVAL_SECONDS = _EnvironmentVariable(
    "MLFLOW_TRACE_TIMEOUT_CHECK_INTERVAL_SECONDS", int, 1
)

# How long a trace can be buffered in-memory at client side before being abandoned.
MLFLOW_TRACE_BUFFER_TTL_SECONDS = _EnvironmentVariable("MLFLOW_TRACE_BUFFER_TTL_SECONDS", int, 3600)

# How many traces to be buffered in-memory at client side before being abandoned.
MLFLOW_TRACE_BUFFER_MAX_SIZE = _EnvironmentVariable("MLFLOW_TRACE_BUFFER_MAX_SIZE", int, 1000)

#: Maximum number of prompt versions to cache in the LRU cache for _load_prompt_version_cached.
#: This cache improves performance by avoiding repeated network calls for the same prompt version.
#: (default: ``128``)
MLFLOW_PROMPT_CACHE_MAX_SIZE = _EnvironmentVariable("MLFLOW_PROMPT_CACHE_MAX_SIZE", int, 128)

#: Private configuration option.
#: Enables the ability to catch exceptions within MLflow evaluate for classification models
#: where a class imbalance due to a missing target class would raise an error in the
#: underlying metrology modules (scikit-learn). If set to True, specific exceptions will be
#: caught, alerted via the warnings module, and evaluation will resume.
#: (default: ``False``)
_MLFLOW_EVALUATE_SUPPRESS_CLASSIFICATION_ERRORS = _BooleanEnvironmentVariable(
    "_MLFLOW_EVALUATE_SUPPRESS_CLASSIFICATION_ERRORS", False
)

#: Maximum number of workers to use for running model prediction and scoring during
#: for each row in the dataset passed to the `mlflow.genai.evaluate` function.
#: (default: ``10``)
MLFLOW_GENAI_EVAL_MAX_WORKERS = _EnvironmentVariable("MLFLOW_GENAI_EVAL_MAX_WORKERS", int, 10)


#: Skip trace validation during GenAI evaluation. By default (False), MLflow will validate if
#: the given predict function generates a valid trace, and otherwise wraps it with @mlflow.trace
#: decorator to make sure a trace is generated. This validation requires running a single
#: prediction. When you are sure that the predict function generates a trace, set this to True
#: to skip the validation and save the time of running a single prediction.
MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION = _BooleanEnvironmentVariable(
    "MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION", False
)

#: Whether to warn (default) or raise (opt-in) for unresolvable requirements inference for
#: a model's dependency inference. If set to True, an exception will be raised if requirements
#: inference or the process of capturing imported modules encounters any errors.
MLFLOW_REQUIREMENTS_INFERENCE_RAISE_ERRORS = _BooleanEnvironmentVariable(
    "MLFLOW_REQUIREMENTS_INFERENCE_RAISE_ERRORS", False
)

# How many traces to display in Databricks Notebooks
MLFLOW_MAX_TRACES_TO_DISPLAY_IN_NOTEBOOK = _EnvironmentVariable(
    "MLFLOW_MAX_TRACES_TO_DISPLAY_IN_NOTEBOOK", int, 10
)

#: Specifies the sampling ratio for traces. Value should be between 0.0 and 1.0.
#: A value of 1.0 means all traces are sampled (default behavior).
#: A value of 0.5 means 50% of traces are sampled.
#: A value of 0.0 means no traces are sampled.
#: (default: ``1.0``)
MLFLOW_TRACE_SAMPLING_RATIO = _EnvironmentVariable("MLFLOW_TRACE_SAMPLING_RATIO", float, 1.0)

#: When OTel export is configured and this is set to true, MLflow will write spans to BOTH
#: MLflow Tracking Server and OpenTelemetry Collector. When false (default), OTel export
#: replaces MLflow export.
#: (default: ``False``)
MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT = _BooleanEnvironmentVariable(
    "MLFLOW_TRACE_ENABLE_OTLP_DUAL_EXPORT", False
)

#: Controls whether MLflow should export traces to OTLP endpoint when
#: OTEL_EXPORTER_OTLP_TRACES_ENDPOINT is set. This allows users to disable MLflow's OTLP
#: export even when the OTEL endpoint is configured for other telemetry clients.
#: (default: ``True``)
MLFLOW_ENABLE_OTLP_EXPORTER = _BooleanEnvironmentVariable("MLFLOW_ENABLE_OTLP_EXPORTER", True)


# Default addressing style to use for boto client
MLFLOW_BOTO_CLIENT_ADDRESSING_STYLE = _EnvironmentVariable(
    "MLFLOW_BOTO_CLIENT_ADDRESSING_STYLE", str, "auto"
)

#: Specify the timeout in seconds for Databricks endpoint HTTP request retries.
MLFLOW_DATABRICKS_ENDPOINT_HTTP_RETRY_TIMEOUT = _EnvironmentVariable(
    "MLFLOW_DATABRICKS_ENDPOINT_HTTP_RETRY_TIMEOUT", int, 500
)

#: Specifies the number of connection pools to cache in urllib3. This environment variable sets the
#: `pool_connections` parameter in the `requests.adapters.HTTPAdapter` constructor. By adjusting
#: this variable, users can enhance the concurrency of HTTP requests made by MLflow.
MLFLOW_HTTP_POOL_CONNECTIONS = _EnvironmentVariable("MLFLOW_HTTP_POOL_CONNECTIONS", int, 10)

#: Specifies the maximum number of connections to keep in the HTTP connection pool. This environment
#: variable sets the `pool_maxsize` parameter in the `requests.adapters.HTTPAdapter` constructor.
#: By adjusting this variable, users can enhance the concurrency of HTTP requests made by MLflow.
MLFLOW_HTTP_POOL_MAXSIZE = _EnvironmentVariable("MLFLOW_HTTP_POOL_MAXSIZE", int, 10)

#: Enable Unity Catalog integration for MLflow AI Gateway.
#: (default: ``False``)
MLFLOW_ENABLE_UC_FUNCTIONS = _BooleanEnvironmentVariable("MLFLOW_ENABLE_UC_FUNCTIONS", False)

#: Specifies the length of time in seconds for the asynchronous logging thread to wait before
#: logging a batch.
MLFLOW_ASYNC_LOGGING_BUFFERING_SECONDS = _EnvironmentVariable(
    "MLFLOW_ASYNC_LOGGING_BUFFERING_SECONDS", int, None
)

#: Whether to enable Databricks SDK. If true, MLflow uses databricks-sdk to send HTTP requests
#: to Databricks endpoint, otherwise MLflow uses ``requests`` library to send HTTP requests
#: to Databricks endpoint. Note that if you want to use OAuth authentication, you have to
#: set this environment variable to true.
#: (default: ``True``)
MLFLOW_ENABLE_DB_SDK = _BooleanEnvironmentVariable("MLFLOW_ENABLE_DB_SDK", True)

#: A flag that's set to 'true' in the child process for capturing modules.
_MLFLOW_IN_CAPTURE_MODULE_PROCESS = _BooleanEnvironmentVariable(
    "MLFLOW_IN_CAPTURE_MODULE_PROCESS", False
)

#: Use DatabricksSDKModelsArtifactRepository when registering and loading models to and from
#: Databricks UC. This is required for SEG(Secure Egress Gateway) enabled workspaces and helps
#: eliminate models exfiltration risk associated with temporary scoped token generation used in
#: existing model artifact repo classes.
MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC = _BooleanEnvironmentVariable(
    "MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC", False
)

#: Disable Databricks SDK for run artifacts. We enable this by default since we want to
#: use Databricks SDK for run artifacts in most cases, but this gives us a way to disable
#: it for certain cases if needed.
MLFLOW_DISABLE_DATABRICKS_SDK_FOR_RUN_ARTIFACTS = _BooleanEnvironmentVariable(
    "MLFLOW_DISABLE_DATABRICKS_SDK_FOR_RUN_ARTIFACTS", False
)

#: Skip signature validation check when migrating model versions from Databricks Workspace
#: Model Registry to Databricks Unity Catalog Model Registry.
#: (default: ``False``)
MLFLOW_SKIP_SIGNATURE_CHECK_FOR_UC_REGISTRY_MIGRATION = _BooleanEnvironmentVariable(
    "MLFLOW_SKIP_SIGNATURE_CHECK_FOR_UC_REGISTRY_MIGRATION", False
)

# Specifies the model environment archive file downloading path when using
# ``mlflow.pyfunc.spark_udf``. (default: ``None``)
MLFLOW_MODEL_ENV_DOWNLOADING_TEMP_DIR = _EnvironmentVariable(
    "MLFLOW_MODEL_ENV_DOWNLOADING_TEMP_DIR", str, None
)

# Specifies whether to log environment variable names used during model logging.
MLFLOW_RECORD_ENV_VARS_IN_MODEL_LOGGING = _BooleanEnvironmentVariable(
    "MLFLOW_RECORD_ENV_VARS_IN_MODEL_LOGGING", True
)

#: Specifies the artifact compression method used when logging a model
#: allowed values are "lzma", "bzip2" and "gzip"
#: (default: ``None``, indicating no compression)
MLFLOW_LOG_MODEL_COMPRESSION = _EnvironmentVariable("MLFLOW_LOG_MODEL_COMPRESSION", str, None)


# Specifies whether to convert a {"messages": [{"role": "...", "content": "..."}]} input
# to a List[BaseMessage] object when invoking a PyFunc model saved with langchain flavor.
# This takes precedence over the default behavior of trying such conversion if the model
# is not an AgentExecutor and the input schema doesn't contain a 'messages' field.
MLFLOW_CONVERT_MESSAGES_DICT_FOR_LANGCHAIN = _BooleanEnvironmentVariable(
    "MLFLOW_CONVERT_MESSAGES_DICT_FOR_LANGCHAIN", None
)

#: A boolean flag which enables additional functionality in Python tests for GO backend.
_MLFLOW_GO_STORE_TESTING = _BooleanEnvironmentVariable("MLFLOW_GO_STORE_TESTING", False)

# Specifies whether the current environment is a serving environment.
# This should only be used internally by MLflow to add some additional logic when running in a
# serving environment.
_MLFLOW_IS_IN_SERVING_ENVIRONMENT = _BooleanEnvironmentVariable(
    "_MLFLOW_IS_IN_SERVING_ENVIRONMENT", None
)

#: Secret key for the Flask app. This is necessary for enabling CSRF protection
#: in the UI signup page when running the app with basic authentication enabled
MLFLOW_FLASK_SERVER_SECRET_KEY = _EnvironmentVariable("MLFLOW_FLASK_SERVER_SECRET_KEY", str, None)

#: (MLflow 3.5.0+) Comma-separated list of allowed CORS origins for the MLflow server.
#: Example: "http://localhost:3000,https://app.example.com"
#: Use "*" to allow ALL origins (DANGEROUS - only use for development!).
#: (default: ``None`` - localhost origins only)
MLFLOW_SERVER_CORS_ALLOWED_ORIGINS = _EnvironmentVariable(
    "MLFLOW_SERVER_CORS_ALLOWED_ORIGINS", str, None
)

#: (MLflow 3.5.0+) Comma-separated list of allowed Host headers for the MLflow server.
#: Example: "mlflow.company.com,mlflow.internal:5000"
#: Use "*" to allow ALL hosts (not recommended for production).
#: If not set, defaults to localhost variants and private IP ranges.
#: (default: ``None`` - localhost and private IP ranges)
MLFLOW_SERVER_ALLOWED_HOSTS = _EnvironmentVariable("MLFLOW_SERVER_ALLOWED_HOSTS", str, None)

#: (MLflow 3.5.0+) Disable all security middleware (DANGEROUS - only use for testing!).
#: Set to "true" to disable security headers, CORS protection, and host validation.
#: (default: ``"false"``)
MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE = _EnvironmentVariable(
    "MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE", str, "false"
)

#: (MLflow 3.5.0+) X-Frame-Options header value for clickjacking protection.
#: Options: "SAMEORIGIN" (default), "DENY", or "NONE" (disable).
#: Set to "NONE" to allow embedding MLflow UI in iframes from different origins.
#: (default: ``"SAMEORIGIN"``)
MLFLOW_SERVER_X_FRAME_OPTIONS = _EnvironmentVariable(
    "MLFLOW_SERVER_X_FRAME_OPTIONS", str, "SAMEORIGIN"
)

#: Specifies the max length (in chars) of an experiment's artifact location.
#: The default is 2048.
MLFLOW_ARTIFACT_LOCATION_MAX_LENGTH = _EnvironmentVariable(
    "MLFLOW_ARTIFACT_LOCATION_MAX_LENGTH", int, 2048
)

#: Path to SSL CA certificate file for MySQL connections
#: Used when creating a SQLAlchemy engine for MySQL
#: (default: ``None``)
MLFLOW_MYSQL_SSL_CA = _EnvironmentVariable("MLFLOW_MYSQL_SSL_CA", str, None)

#: Path to SSL certificate file for MySQL connections
#: Used when creating a SQLAlchemy engine for MySQL
#: (default: ``None``)
MLFLOW_MYSQL_SSL_CERT = _EnvironmentVariable("MLFLOW_MYSQL_SSL_CERT", str, None)

#: Path to SSL key file for MySQL connections
#: Used when creating a SQLAlchemy engine for MySQL
#: (default: ``None``)
MLFLOW_MYSQL_SSL_KEY = _EnvironmentVariable("MLFLOW_MYSQL_SSL_KEY", str, None)

#######################################################################################
# Tracing
#######################################################################################

#: Specifies whether to enable async trace logging to Databricks Tracing Server.
#: TODO: Update OSS MLflow Server to logging async by default
#: Default: ``True``.
MLFLOW_ENABLE_ASYNC_TRACE_LOGGING = _BooleanEnvironmentVariable(
    "MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", True
)

#: Maximum number of worker threads to use for async trace logging.
#: (default: ``10``)
MLFLOW_ASYNC_TRACE_LOGGING_MAX_WORKERS = _EnvironmentVariable(
    "MLFLOW_ASYNC_TRACE_LOGGING_MAX_WORKERS", int, 10
)

#: Maximum number of export tasks to queue for async trace logging.
#: When the queue is full, new export tasks will be dropped.
#: (default: ``1000``)
MLFLOW_ASYNC_TRACE_LOGGING_MAX_QUEUE_SIZE = _EnvironmentVariable(
    "MLFLOW_ASYNC_TRACE_LOGGING_MAX_QUEUE_SIZE", int, 1000
)


#: Timeout seconds for retrying trace logging.
#: (default: ``500``)
MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT = _EnvironmentVariable(
    "MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT", int, 500
)

#: Specifies the SQL warehouse ID to use for tracing with Databricks backend.
#: (default: ``None``)
MLFLOW_TRACING_SQL_WAREHOUSE_ID = _EnvironmentVariable("MLFLOW_TRACING_SQL_WAREHOUSE_ID", str, None)


#: Specifies the location to send traces to. This can be either an MLflow experiment ID or a
#: Databricks Unity Catalog (UC) schema (format: `<catalog_name>.<schema_name>`).
#: (default: ``None`` (an active MLflow experiment will be used))
MLFLOW_TRACING_DESTINATION = _EnvironmentVariable("MLFLOW_TRACING_DESTINATION", str, None)


#######################################################################################
# Model Logging
#######################################################################################

#: The default active LoggedModel ID. Traces created while this variable is set (unless overridden,
#: e.g., by the `set_active_model()` API) will be associated with this LoggedModel ID.
#: (default: ``None``)
MLFLOW_ACTIVE_MODEL_ID = _EnvironmentVariable("MLFLOW_ACTIVE_MODEL_ID", str, None)

#: Legacy environment variable for setting the default active LoggedModel ID.
#: This should only by used by MLflow internally. Users should use the
#: public `MLFLOW_ACTIVE_MODEL_ID` environment variable or the `set_active_model`
#: API to set the active LoggedModel, and should not set this environment variable directly.
#: (default: ``None``)
_MLFLOW_ACTIVE_MODEL_ID = _EnvironmentVariable("_MLFLOW_ACTIVE_MODEL_ID", str, None)

#: Maximum number of parameters to include in the initial CreateLoggedModel request.
#: Additional parameters will be logged in separate requests.
#: (default: ``100``)
_MLFLOW_CREATE_LOGGED_MODEL_PARAMS_BATCH_SIZE = _EnvironmentVariable(
    "_MLFLOW_CREATE_LOGGED_MODEL_PARAMS_BATCH_SIZE", int, 100
)


#: Maximum number of parameters to include in each batch when logging parameters
#: for a logged model.
#: (default: ``100``)
_MLFLOW_LOG_LOGGED_MODEL_PARAMS_BATCH_SIZE = _EnvironmentVariable(
    "_MLFLOW_LOG_LOGGED_MODEL_PARAMS_BATCH_SIZE", int, 100
)

#: A boolean flag that enables printing URLs for logged and registered models when
#: they are created.
#: (default: ``True``)
MLFLOW_PRINT_MODEL_URLS_ON_CREATION = _BooleanEnvironmentVariable(
    "MLFLOW_PRINT_MODEL_URLS_ON_CREATION", True
)

#: Maximum number of threads to use when downloading traces during search operations.
#: (default: ``max(32, (# of system CPUs * 4)``)
MLFLOW_SEARCH_TRACES_MAX_THREADS = _EnvironmentVariable(
    # Threads used to download traces during search are network IO-bound (waiting for downloads)
    # rather than CPU-bound, so we want more threads than CPU cores
    "MLFLOW_SEARCH_TRACES_MAX_THREADS",
    int,
    max(32, (os.cpu_count() or 1) * 4),
)

#: Maximum number of traces to fetch in a single BatchGetTraces request during search operations.
#: (default: ``10``)
_MLFLOW_SEARCH_TRACES_MAX_BATCH_SIZE = _EnvironmentVariable(
    "MLFLOW_SEARCH_TRACES_MAX_BATCH_SIZE", int, 10
)

#: Specifies the logging level for MLflow. This can be set to any valid logging level
#: (e.g., "DEBUG", "INFO"). This environment must be set before importing mlflow to take
#: effect. To modify the logging level after importing mlflow, use `importlib.reload(mlflow)`.
#: (default: ``None``).
MLFLOW_LOGGING_LEVEL = _EnvironmentVariable("MLFLOW_LOGGING_LEVEL", str, None)

#: Avoid printing experiment and run url to stdout at run termination
#: (default: ``False``)
MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT = _BooleanEnvironmentVariable(
    "MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", False
)

#: If True, MLflow locks both direct and transitive model dependencies when logging a model.
#: (default: ``False``).
MLFLOW_LOCK_MODEL_DEPENDENCIES = _BooleanEnvironmentVariable(
    "MLFLOW_LOCK_MODEL_DEPENDENCIES", False
)

#: If specified, tracking server rejects model `/mlflow/model-versions/create` requests with
#: a source that does not match the specified regular expression.
#: (default: ``None``).
MLFLOW_CREATE_MODEL_VERSION_SOURCE_VALIDATION_REGEX = _EnvironmentVariable(
    "MLFLOW_CREATE_MODEL_VERSION_SOURCE_VALIDATION_REGEX", str, None
)

#: Maximum number of root fields to include in the MLflow server GraphQL request.
#: (default: ``10``)
MLFLOW_SERVER_GRAPHQL_MAX_ROOT_FIELDS = _EnvironmentVariable(
    "MLFLOW_SERVER_GRAPHQL_MAX_ROOT_FIELDS", int, 10
)

#: Maximum number of aliases to include in the MLflow server GraphQL request.
#: (default: ``10``)
MLFLOW_SERVER_GRAPHQL_MAX_ALIASES = _EnvironmentVariable(
    "MLFLOW_SERVER_GRAPHQL_MAX_ALIASES", int, 10
)


#: Whether to disable schema details in error messages for MLflow schema enforcement.
#: (default: ``False``)
MLFLOW_DISABLE_SCHEMA_DETAILS = _BooleanEnvironmentVariable("MLFLOW_DISABLE_SCHEMA_DETAILS", False)


def _split_strip(s: str) -> list[str]:
    return [s.strip() for s in s.split(",")]


# Specifies the allowed schemes for MLflow webhook URLs.
# This environment variable is not intended for production use.
_MLFLOW_WEBHOOK_ALLOWED_SCHEMES = _EnvironmentVariable(
    "MLFLOW_WEBHOOK_ALLOWED_SCHEMES", _split_strip, ["https"]
)


#: Specifies the secret key used to encrypt webhook secrets in MLflow.
MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY = _EnvironmentVariable(
    "MLFLOW_WEBHOOK_SECRET_ENCRYPTION_KEY", str, None
)

#: Specifies the timeout in seconds for webhook HTTP requests
#: (default: ``30``)
MLFLOW_WEBHOOK_REQUEST_TIMEOUT = _EnvironmentVariable("MLFLOW_WEBHOOK_REQUEST_TIMEOUT", int, 30)

#: Specifies the maximum number of threads for webhook delivery thread pool
#: (default: ``10``)
MLFLOW_WEBHOOK_DELIVERY_MAX_WORKERS = _EnvironmentVariable(
    "MLFLOW_WEBHOOK_DELIVERY_MAX_WORKERS", int, 10
)

#: Specifies the maximum number of retries for webhook HTTP requests
#: (default: ``3``)
MLFLOW_WEBHOOK_REQUEST_MAX_RETRIES = _EnvironmentVariable(
    "MLFLOW_WEBHOOK_REQUEST_MAX_RETRIES", int, 3
)

#: Specifies the TTL in seconds for webhook list cache
#: (default: ``60``)
MLFLOW_WEBHOOK_CACHE_TTL = _EnvironmentVariable("MLFLOW_WEBHOOK_CACHE_TTL", int, 60)


#: Whether to disable telemetry collection in MLflow. If set to True, no telemetry
#: data will be collected. (default: ``False``)
MLFLOW_DISABLE_TELEMETRY = _BooleanEnvironmentVariable("MLFLOW_DISABLE_TELEMETRY", False)


#: Internal flag to enable telemetry in mlflow tests.
#: (default: ``False``)
_MLFLOW_TESTING_TELEMETRY = _BooleanEnvironmentVariable("_MLFLOW_TESTING_TELEMETRY", False)


#: Internal environment variable to set the telemetry session id when TelemetryClient is initialized
#: This should never be set by users or explicitly.
#: (default: ``None``)
_MLFLOW_TELEMETRY_SESSION_ID = _EnvironmentVariable("_MLFLOW_TELEMETRY_SESSION_ID", str, None)


#: Internal flag to enable telemetry logging
#: (default: ``False``)
_MLFLOW_TELEMETRY_LOGGING = _BooleanEnvironmentVariable("_MLFLOW_TELEMETRY_LOGGING", False)

#: Internal environment variable to indicate which SGI is being used,
#: e.g. "uvicorn" or "gunicorn".
#: This should never be set by users or explicitly.
#: (default: ``None``)
_MLFLOW_SGI_NAME = _EnvironmentVariable("_MLFLOW_SGI_NAME", str, None)

#: Specifies whether to enforce using stdin scoring server in Spark udf.
#: (default: ``True``)
MLFLOW_ENFORCE_STDIN_SCORING_SERVER_FOR_SPARK_UDF = _BooleanEnvironmentVariable(
    "MLFLOW_ENFORCE_STDIN_SCORING_SERVER_FOR_SPARK_UDF", True
)

#: Specifies whether to enable job execution feature for MLflow server.
#: This feature requires "huey" package dependency, and requires MLflow server to configure
#: --backend-store-uri to database URI.
#: (default: ``False``)
MLFLOW_SERVER_ENABLE_JOB_EXECUTION = _BooleanEnvironmentVariable(
    "MLFLOW_SERVER_ENABLE_JOB_EXECUTION", False
)

#: Specifies MLflow server job maximum allowed retries for transient errors.
#: (default: ``3``)
MLFLOW_SERVER_JOB_TRANSIENT_ERROR_MAX_RETRIES = _EnvironmentVariable(
    "MLFLOW_SERVER_JOB_TRANSIENT_ERROR_MAX_RETRIES", int, 3
)

#: Specifies MLflow server job retry base delay in seconds for transient errors.
#: The retry uses exponential backoff strategy, retry delay is computed by
#: `delay = min(base_delay * (2 ** (retry_count - 1)), max_delay)`
#: (default: ``15``)
MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_BASE_DELAY = _EnvironmentVariable(
    "MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_BASE_DELAY", int, 15
)

#: Specifies MLflow server job retry maximum delay in seconds for transient errors.
#: The retry uses exponential backoff strategy, retry delay is computed by
#: `delay = min(base_delay * (2 ** (retry_count - 1)), max_delay)`
#: (default: ``60``)
MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_MAX_DELAY = _EnvironmentVariable(
    "MLFLOW_SERVER_JOB_TRANSIENT_ERROR_RETRY_MAX_DELAY", int, 60
)


#: Specifies the maximum number of completion iterations allowed when invoking
#: judge models. This prevents infinite loops in case of complex traces or
#: issues with the judge's reasoning.
#: (default: ``30``)
MLFLOW_JUDGE_MAX_ITERATIONS = _EnvironmentVariable("MLFLOW_JUDGE_MAX_ITERATIONS", int, 30)
