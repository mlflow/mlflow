"""
This module defines environment variable names used in MLflow.
"""
from mlflow.utils.env import EnvironmentVariable

#: The environment variable to specify maximum retries for mlflow http request.
MLFLOW_HTTP_REQUEST_MAX_RETRIES = EnvironmentVariable("MLFLOW_HTTP_REQUEST_MAX_RETRIES", int, 5)

#: The environment variable to specify backoff factor for mlflow http request.
MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR = EnvironmentVariable(
    "MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", int, 2
)

#: The environment variable to specify timeout in seconds for mlflow http request.
MLFLOW_HTTP_REQUEST_TIMEOUT = EnvironmentVariable("MLFLOW_HTTP_REQUEST_TIMEOUT", int, 120)
