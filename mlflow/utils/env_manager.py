from mlflow.environment_variables import MLFLOW_ENV_MANAGER
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

LOCAL = "local"
CONDA = "conda"
VIRTUALENV = "virtualenv"


def validate_and_set(env_manager):
    allowed_values = [LOCAL, CONDA, VIRTUALENV]
    if env_manager not in allowed_values:
        raise MlflowException(
            f"Invalid value for `env_manager`: {env_manager}. Must be one of {allowed_values}",
            error_code=INVALID_PARAMETER_VALUE,
        )
    MLFLOW_ENV_MANAGER.set(env_manager)
