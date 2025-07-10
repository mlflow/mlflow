"""Azure authentication support for MLflow."""

from mlflow.azure.config import AzureAuthConfig, AuthMethod
from mlflow.azure.auth_handler import AzureAuthHandler
from mlflow.azure.connection_factory import ConnectionFactory
from mlflow.azure.stores import create_store, get_azure_tracking_store, test_azure_connection
from mlflow.azure.exceptions import (
    AzureAuthError,
    TokenAcquisitionError,
    ConnectionError,
    ConfigurationError,
)

__all__ = [
    "AzureAuthConfig",
    "AuthMethod", 
    "AzureAuthHandler",
    "ConnectionFactory",
    "create_store",
    "get_azure_tracking_store",
    "test_azure_connection",
    "AzureAuthError",
    "TokenAcquisitionError",
    "ConnectionError",
    "ConfigurationError",
]