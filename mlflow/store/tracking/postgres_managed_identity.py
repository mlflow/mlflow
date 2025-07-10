"""
PostgreSQL with Azure Managed Identity Support for Genesis-Flow

This module provides enhanced PostgreSQL connectivity with Azure Managed Identity,
allowing passwordless authentication for secure production deployments.

This is a compatibility layer that uses the new Azure authentication architecture.
"""

import os
import logging
from typing import Optional

from mlflow.azure.config import AzureAuthConfig
from mlflow.azure.stores import create_store

logger = logging.getLogger(__name__)


def get_postgres_store_with_managed_identity(store_uri: str, artifact_uri: Optional[str] = None):
    """
    Get a PostgreSQL store with Azure Managed Identity support.
    
    This is the main entry point for the postgres managed identity functionality.
    It uses the new Azure authentication architecture.
    
    Args:
        store_uri: PostgreSQL connection URI
        artifact_uri: Artifact storage URI
        
    Returns:
        SqlAlchemyStore configured with Managed Identity authentication
    """
    logger.info("Creating PostgreSQL store with Azure Managed Identity support")
    
    # Use the new architecture
    return create_store(store_uri, artifact_uri)


# Legacy compatibility functions
def create_postgres_engine_with_managed_identity(db_uri: str, **kwargs):
    """
    Legacy compatibility function for creating engines.
    
    This now uses the new ConnectionFactory architecture.
    """
    from mlflow.azure.connection_factory import ConnectionFactory
    
    config = AzureAuthConfig()
    factory = ConnectionFactory(config)
    
    return factory.create_engine(db_uri, **kwargs)


class PostgresManagedIdentityAuth:
    """
    Legacy compatibility class.
    
    The actual authentication is now handled by AzureAuthHandler.
    """
    
    @staticmethod
    def get_azure_ad_token() -> Optional[str]:
        """Get Azure AD access token."""
        from mlflow.azure.auth_handler import AzureAuthHandler
        
        config = AzureAuthConfig()
        if not config.should_use_azure_auth:
            return None
            
        handler = AzureAuthHandler(config)
        try:
            return handler.get_access_token()
        except Exception as e:
            logger.error("Failed to get Azure AD token: %s", str(e))
            return None


# Configuration helper for easy setup  
class PostgresConfig:
    """Helper class for PostgreSQL configuration with Managed Identity."""
    
    @staticmethod
    def get_connection_string(
        host: str,
        database: str,
        username: str,
        port: int = 5432,
        use_managed_identity: bool = True,
        **params
    ) -> str:
        """
        Get a connection string for PostgreSQL with optional Managed Identity.
        
        Args:
            host: Database host
            database: Database name  
            username: Database username
            port: Database port
            use_managed_identity: Whether to use Managed Identity
            **params: Additional connection parameters
            
        Returns:
            Connection string
        """
        from mlflow.azure.connection_factory import ConnectionFactory
        
        config = AzureAuthConfig()
        factory = ConnectionFactory(config)
        
        return factory.create_connection_url(
            host=host,
            database=database, 
            username=username,
            port=port,
            **params
        )
    
    @staticmethod
    def validate_azure_environment() -> bool:
        """
        Validate that the Azure environment is properly configured.
        
        Returns:
            True if environment is valid for Azure auth
        """
        config = AzureAuthConfig()
        return config.should_use_azure_auth
    
    @staticmethod 
    def configure_from_environment() -> dict:
        """
        Configure PostgreSQL connection from environment variables.
        
        Returns:
            Configuration dictionary
        """
        config = AzureAuthConfig()
        
        return {
            "auth_enabled": config.auth_enabled,
            "auth_method": config.auth_method.value,
            "client_id": config.client_id,
            "tenant_id": config.tenant_id,
            "should_use_azure_auth": config.should_use_azure_auth,
        }