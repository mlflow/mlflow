from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

LOCAL = "local"
CONDA = "conda"
VIRTUALENV = "virtualenv"
UV = "uv"


def validate(env_manager):
    allowed_values = [LOCAL, CONDA, VIRTUALENV, UV]
    if env_manager not in allowed_values:
        raise MlflowException(
            f"Invalid value for `env_manager`: {env_manager}. Must be one of {allowed_values}",
            error_code=INVALID_PARAMETER_VALUE,
        )
