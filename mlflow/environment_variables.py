"""
This module defines environment variable names used in MLflow.
"""
from mlflow.utils.env import EnvironmentVariable

#: Specify maximum retries for mlflow http request, default value is 5.
MLFLOW_HTTP_REQUEST_MAX_RETRIES = EnvironmentVariable("MLFLOW_HTTP_REQUEST_MAX_RETRIES", int, 5)

#: Specify backoff factor for mlflow http request, default value is 2.
MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR = EnvironmentVariable(
    "MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", int, 2
)

#: Specify timeout in seconds for mlflow http request, default value is 120.
MLFLOW_HTTP_REQUEST_TIMEOUT = EnvironmentVariable("MLFLOW_HTTP_REQUEST_TIMEOUT", int, 120)
