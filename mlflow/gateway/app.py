"""
TODO: Remove this module once after AI Gateway deprecation window elapses
"""

from mlflow.deployments.server.app import GatewayAPI
from mlflow.deployments.server.app import (
    create_app_from_config as deployments_create_app_from_config,
)
from mlflow.deployments.server.app import create_app_from_path as deployments_create_app_from_path
from mlflow.environment_variables import MLFLOW_GATEWAY_CONFIG
from mlflow.exceptions import MlflowException
from mlflow.gateway.config import GatewayConfig


def create_app_from_config(config: GatewayConfig) -> GatewayAPI:
    return deployments_create_app_from_config(config)


def create_app_from_env() -> GatewayAPI:
    """
    Load the path from the environment variable and generate the GatewayAPI app instance.
    """
    if config_path := MLFLOW_GATEWAY_CONFIG.get():
        return deployments_create_app_from_path(config_path)

    raise MlflowException(
        f"Environment variable {MLFLOW_GATEWAY_CONFIG!r} is not set. "
        "Please set it to the path of the gateway configuration file."
    )
