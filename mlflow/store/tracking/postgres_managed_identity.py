"""
PostgreSQL with Azure Managed Identity Support for Genesis-Flow

This module provides enhanced PostgreSQL connectivity with Azure Managed Identity,
allowing passwordless authentication for secure production deployments.
"""

import os
import logging
from typing import Optional, Dict, Any
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
import json

from sqlalchemy import create_engine, event
from sqlalchemy.pool import NullPool

logger = logging.getLogger(__name__)


class PostgresManagedIdentityAuth:
    """
    Handles PostgreSQL authentication with Azure Managed Identity support.
    
    This class provides seamless integration with Azure AD authentication,
    allowing applications to connect to Azure Database for PostgreSQL using
    managed identities instead of passwords.
    """
    
    @staticmethod
    def get_azure_ad_token() -> Optional[str]:
        """
        Retrieve Azure AD access token for PostgreSQL authentication.
        
        Returns:
            Access token string or None if not available
        """
        try:
            # Check if running in Azure environment with Managed Identity
            if os.getenv("IDENTITY_ENDPOINT") and os.getenv("IDENTITY_HEADER"):
                # Azure App Service / Function App Managed Identity
                return PostgresManagedIdentityAuth._get_app_service_token()
            elif os.getenv("MSI_ENDPOINT"):
                # Azure VM Managed Identity (legacy)
                return PostgresManagedIdentityAuth._get_vm_token_legacy()
            elif os.getenv("IMDS_ENDPOINT"):
                # Azure VM Managed Identity (newer)
                return PostgresManagedIdentityAuth._get_vm_token()
            else:
                # Try Azure CLI for local development
                return PostgresManagedIdentityAuth._get_cli_token()
        except Exception as e:
            logger.warning(f"Failed to get Azure AD token: {e}")
            return None
    
    @staticmethod
    def _get_app_service_token() -> Optional[str]:
        """Get token from App Service Managed Identity."""
        import requests
        
        endpoint = os.getenv("IDENTITY_ENDPOINT")
        header = os.getenv("IDENTITY_HEADER")
        
        params = {
            "api-version": "2019-08-01",
            "resource": "https://ossrdbms-aad.database.windows.net"
        }
        
        headers = {"X-IDENTITY-HEADER": header}
        
        try:
            response = requests.get(endpoint, params=params, headers=headers, timeout=5)
            if response.status_code == 200:
                return response.json()["access_token"]
        except Exception as e:
            logger.error(f"Failed to get App Service token: {e}")
        
        return None
    
    @staticmethod
    def _get_vm_token() -> Optional[str]:
        """Get token from VM Managed Identity (IMDS)."""
        import requests
        
        endpoint = "http://169.254.169.254/metadata/identity/oauth2/token"
        
        params = {
            "api-version": "2018-02-01",
            "resource": "https://ossrdbms-aad.database.windows.net"
        }
        
        headers = {"Metadata": "true"}
        
        try:
            response = requests.get(endpoint, params=params, headers=headers, timeout=5)
            if response.status_code == 200:
                return response.json()["access_token"]
        except Exception as e:
            logger.error(f"Failed to get VM token: {e}")
        
        return None
    
    @staticmethod
    def _get_vm_token_legacy() -> Optional[str]:
        """Get token from VM Managed Identity (legacy MSI endpoint)."""
        import requests
        
        endpoint = os.getenv("MSI_ENDPOINT")
        
        params = {
            "api-version": "2017-09-01",
            "resource": "https://ossrdbms-aad.database.windows.net"
        }
        
        headers = {"secret": os.getenv("MSI_SECRET", "")}
        
        try:
            response = requests.get(endpoint, params=params, headers=headers, timeout=5)
            if response.status_code == 200:
                return response.json()["access_token"]
        except Exception as e:
            logger.error(f"Failed to get legacy VM token: {e}")
        
        return None
    
    @staticmethod
    def _get_cli_token() -> Optional[str]:
        """Get token using Azure CLI for local development."""
        try:
            import subprocess
            import json
            
            result = subprocess.run(
                ["az", "account", "get-access-token", "--resource", 
                 "https://ossrdbms-aad.database.windows.net"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                token_data = json.loads(result.stdout)
                return token_data["accessToken"]
        except Exception as e:
            logger.debug(f"Azure CLI not available or failed: {e}")
        
        return None
    
    @staticmethod
    def configure_engine(engine, connection_record):
        """
        Configure SQLAlchemy engine to use Azure AD token for authentication.
        
        This is used as an event listener for SQLAlchemy's 'connect' event.
        """
        token = PostgresManagedIdentityAuth.get_azure_ad_token()
        if token:
            # Set the password to the Azure AD token
            connection_record.info['password'] = token
            logger.debug("Configured connection with Azure AD token")


def create_postgres_engine_with_managed_identity(db_uri: str, **kwargs) -> Any:
    """
    Create a SQLAlchemy engine with Azure Managed Identity support.
    
    Args:
        db_uri: PostgreSQL connection URI
        **kwargs: Additional arguments for create_engine
        
    Returns:
        SQLAlchemy engine configured for Managed Identity authentication
    """
    # Parse the URI to check if we should use Managed Identity
    parsed = urlparse(db_uri)
    query_params = parse_qs(parsed.query)
    
    # Check if Managed Identity is explicitly requested or if no password is provided
    use_managed_identity = (
        query_params.get("auth_method", [""])[0] == "managed_identity" or
        os.getenv("MLFLOW_POSTGRES_USE_MANAGED_IDENTITY", "").lower() == "true" or
        (not parsed.password and os.getenv("IDENTITY_ENDPOINT"))  # Auto-detect Azure environment
    )
    
    if use_managed_identity:
        logger.info("Configuring PostgreSQL connection with Azure Managed Identity")
        
        # Remove password from URI if present
        if parsed.password:
            netloc = f"{parsed.username}@{parsed.hostname}"
            if parsed.port:
                netloc += f":{parsed.port}"
            parsed = parsed._replace(netloc=netloc)
            db_uri = urlunparse(parsed)
        
        # Create engine
        engine_kwargs = {
            "poolclass": NullPool,  # Disable connection pooling for token refresh
            "connect_args": {
                "connect_timeout": 30,
                "sslmode": query_params.get("sslmode", ["require"])[0]
            }
        }
        engine_kwargs.update(kwargs)
        
        engine = create_engine(db_uri, **engine_kwargs)
        
        # Add event listener for token injection
        @event.listens_for(engine, "do_connect")
        def provide_token(dialect, conn_rec, cargs, cparams):
            token = PostgresManagedIdentityAuth.get_azure_ad_token()
            if token:
                # For psycopg2, set password in connection parameters
                cparams['password'] = token
            else:
                logger.warning("Failed to get Azure AD token, attempting connection anyway")
        
        return engine
    else:
        # Standard PostgreSQL connection
        return create_engine(db_uri, **kwargs)


def get_postgres_store_with_managed_identity(store_uri: str, artifact_uri: str):
    """
    Get a PostgreSQL-backed MLflow store with Managed Identity support.
    
    Args:
        store_uri: PostgreSQL connection URI
        artifact_uri: Artifact storage URI
        
    Returns:
        SqlAlchemyStore configured with Managed Identity authentication
    """
    from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
    
    # Create a custom SqlAlchemyStore with Managed Identity support
    class PostgresManagedIdentityStore(SqlAlchemyStore):
        def __init__(self, db_uri, default_artifact_root):
            # Override engine creation to use Managed Identity
            self.db_uri = db_uri
            self.default_artifact_root = default_artifact_root
            self.engine = create_postgres_engine_with_managed_identity(db_uri)
            
            # Initialize the rest of the store
            self.SessionMaker = sqlalchemy.orm.sessionmaker(bind=self.engine)
            self._initialize_models_and_lookup_tables()
    
    return PostgresManagedIdentityStore(store_uri, artifact_uri)


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
        ssl_mode: str = "require",
        **kwargs
    ) -> str:
        """
        Build PostgreSQL connection string with proper parameters.
        
        Args:
            host: PostgreSQL server hostname
            database: Database name
            username: Username (often the Azure AD username for Managed Identity)
            port: PostgreSQL port (default: 5432)
            use_managed_identity: Whether to use Managed Identity authentication
            ssl_mode: SSL mode (default: require)
            **kwargs: Additional connection parameters
            
        Returns:
            PostgreSQL connection URI
        """
        # Build base URI
        uri = f"postgresql://{username}@{host}:{port}/{database}"
        
        # Add query parameters
        params = {"sslmode": ssl_mode}
        if use_managed_identity:
            params["auth_method"] = "managed_identity"
        
        params.update(kwargs)
        
        if params:
            uri += "?" + urlencode(params)
        
        return uri
    
    @staticmethod
    def validate_azure_environment() -> Dict[str, Any]:
        """
        Validate Azure environment for Managed Identity usage.
        
        Returns:
            Dictionary with validation results
        """
        results = {
            "is_azure_environment": False,
            "managed_identity_available": False,
            "environment_type": "unknown",
            "issues": []
        }
        
        # Check for Azure environment indicators
        if os.getenv("IDENTITY_ENDPOINT"):
            results["is_azure_environment"] = True
            results["environment_type"] = "app_service"
        elif os.getenv("IMDS_ENDPOINT") or os.getenv("MSI_ENDPOINT"):
            results["is_azure_environment"] = True
            results["environment_type"] = "virtual_machine"
        else:
            # Check if Azure CLI is available
            try:
                import subprocess
                result = subprocess.run(
                    ["az", "account", "show"],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    results["environment_type"] = "local_development"
                    results["managed_identity_available"] = True
            except:
                results["issues"].append("Not running in Azure and Azure CLI not available")
        
        # Test token acquisition
        if results["is_azure_environment"] or results["environment_type"] == "local_development":
            token = PostgresManagedIdentityAuth.get_azure_ad_token()
            if token:
                results["managed_identity_available"] = True
            else:
                results["issues"].append("Failed to acquire Azure AD token")
        
        return results


# Environment variable configuration support
def configure_from_environment():
    """
    Configure PostgreSQL connection from environment variables.
    
    Environment variables:
    - MLFLOW_POSTGRES_HOST: PostgreSQL server hostname
    - MLFLOW_POSTGRES_DATABASE: Database name
    - MLFLOW_POSTGRES_USERNAME: Username
    - MLFLOW_POSTGRES_PORT: Port (default: 5432)
    - MLFLOW_POSTGRES_USE_MANAGED_IDENTITY: Use Managed Identity (true/false)
    - MLFLOW_POSTGRES_SSL_MODE: SSL mode (default: require)
    
    Returns:
        PostgreSQL connection URI or None if not configured
    """
    host = os.getenv("MLFLOW_POSTGRES_HOST")
    database = os.getenv("MLFLOW_POSTGRES_DATABASE")
    username = os.getenv("MLFLOW_POSTGRES_USERNAME")
    
    if not all([host, database, username]):
        return None
    
    return PostgresConfig.get_connection_string(
        host=host,
        database=database,
        username=username,
        port=int(os.getenv("MLFLOW_POSTGRES_PORT", "5432")),
        use_managed_identity=os.getenv("MLFLOW_POSTGRES_USE_MANAGED_IDENTITY", "true").lower() == "true",
        ssl_mode=os.getenv("MLFLOW_POSTGRES_SSL_MODE", "require")
    )