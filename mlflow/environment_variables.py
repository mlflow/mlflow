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
        return f"EnvironmentVariable(name={self.name}, type={self.type}, default={self.default})"

    def __repr__(self):
        return repr(self.name)


#: Specify the maximum number of retries for MLflow http request
#: (default: ``5``)
MLFLOW_HTTP_REQUEST_MAX_RETRIES = _EnvironmentVariable("MLFLOW_HTTP_REQUEST_MAX_RETRIES", int, 5)

#: Specify backoff factor for MLflow http request
#: (default: ``2``)
MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR = _EnvironmentVariable(
    "MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", int, 2
)

#: Specify timeout in seconds for MLflow http request
#: (default: ``120``)
MLFLOW_HTTP_REQUEST_TIMEOUT = _EnvironmentVariable("MLFLOW_HTTP_REQUEST_TIMEOUT", int, 120)
