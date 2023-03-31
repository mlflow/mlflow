"""
This module defines environment variables used in MLflow.
"""
import os


class _EnvironmentVariable:
    """
    Represents an environment variable.
    """

    def __init__(self, name, type_, default):
        self.name = name
        self.type = type_
        self.default = default

    @property
    def is_defined(self):
        return self.name in os.environ

    def get(self):
        """
        Reads the value of the environment variable if it exists and converts it to the desired
        type. Otherwise, returns the default value.
        """
        val = os.getenv(self.name)
        if val:
            try:
                return self.type(val)
            except Exception as e:
                raise ValueError(f"Failed to convert {val} to {self.type} for {self.name}: {e}")
        return self.default

    def __str__(self):
        return f"{self.name} (default: {self.default}, type: {self.type.__name__})"

    def __repr__(self):
        return repr(self.name)


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
        if not self.is_defined:
            return self.default

        val = os.getenv(self.name)
        lowercased = val.lower()
        if lowercased not in ["true", "false", "1", "0"]:
            raise ValueError(
                f"{self.name} value must be one of ['true', 'false', '1', '0'] (case-insensitive), "
                f"but got {val}"
            )
        return lowercased in ["true", "1"]


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

#: Specifies the timeout in seconds for MLflow HTTP requests
#: (default: ``120``)
MLFLOW_HTTP_REQUEST_TIMEOUT = _EnvironmentVariable("MLFLOW_HTTP_REQUEST_TIMEOUT", int, 120)

#: Specifies whether MLFlow HTTP requests should be signed using AWS signature V4. It will overwrite
#: (default: ``False``). When set, it will overwrite the "Authorization" HTTP header.
#: See https://docs.aws.amazon.com/general/latest/gr/signature-version-4.html for more information.
MLFLOW_TRACKING_AWS_SIGV4 = _BooleanEnvironmentVariable("MLFLOW_TRACKING_AWS_SIGV4", False)

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
#: variable is only supported for the MLflow PyTorch flavor.
MLFLOW_DEFAULT_PREDICTION_DEVICE = _EnvironmentVariable(
    "MLFLOW_DEFAULT_PREDICTION_DEVICE", str, None
)

#: Specifies whether or not to allow using a file URI as a model version source.
#: Please be aware that setting this environment variable to True is potentially risky
#: because it can allow access to arbitrary files on the specified filesystem
#: (default: ``False``).
MLFLOW_ALLOW_FILE_URI_AS_MODEL_VERSION_SOURCE = _BooleanEnvironmentVariable(
    "MLFLOW_ALLOW_FILE_URI_AS_MODEL_VERSION_SOURCE", False
)
