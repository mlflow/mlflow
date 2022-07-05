"""
This module defines environment variable names used in MLflow.
"""

#: Specify maximum retries for mlflow rest API http request.
MLFLOW_HTTP_REQUEST_MAX_RETRIES = "MLFLOW_HTTP_REQUEST_MAX_RETRIES"

#: Specify backoff factor for mlflow rest API http request.
MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR = "MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR"

#: Specify timeout in seconds for mlflow rest API http request.
MLFLOW_HTTP_REQUEST_TIMEOUT = "MLFLOW_HTTP_REQUEST_TIMEOUT"
