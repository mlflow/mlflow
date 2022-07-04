"""
This module defines environment variable names used in MLflow.
"""

#: Specify maximum retries for mlflow rest API http request.
MLFLOW_RESTAPI_REQUEST_MAX_RETRIES = "MLFLOW_RESTAPI_REQUEST_MAX_RETRIES"

#: Specify backoff factor for mlflow rest API http request.
MLFLOW_RESTAPI_REQUEST_BACKOFF_FACTOR = "MLFLOW_RESTAPI_REQUEST_BACKOFF_FACTOR"

#: Specify timeout in seconds for mlflow rest API http request.
MLFLOW_RESTAPI_REQUEST_TIMEOUT = "MLFLOW_RESTAPI_REQUEST_TIMEOUT"
