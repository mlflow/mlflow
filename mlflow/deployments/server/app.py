"""
TODO: Remove this module once after Deployments Server deprecation window elapses
"""

from mlflow.environment_variables import (
    MLFLOW_DEPLOYMENTS_CONFIG,
)
from mlflow.exceptions import MlflowException
from mlflow.gateway.app import GatewayAPI
from mlflow.gateway.app import (
    create_app_from_config as gateway_create_app_from_config,
)
from mlflow.gateway.app import (
    create_app_from_path as gateway_create_app_from_path,
)

create_app_from_config = gateway_create_app_from_config
create_app_from_path = gateway_create_app_from_path


def create_app_from_env() -> GatewayAPI:
    """
    Load the path from the environment variable and generate the GatewayAPI app instance.
    """
    if config_path := MLFLOW_DEPLOYMENTS_CONFIG.get():
        return create_app_from_path(config_path)

    raise MlflowException(
        f"Environment variable {MLFLOW_DEPLOYMENTS_CONFIG!r} is not set. "
        "Please set it to the path of the gateway configuration file."
    )
