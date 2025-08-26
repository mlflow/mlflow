"""Azure-authenticated PostgreSQL stores for MLflow."""

from __future__ import annotations

import os
import logging
from typing import Optional, Any, Dict

from sqlalchemy import Engine

from mlflow.azure.config import AzureAuthConfig, AuthMethod
from mlflow.azure.connection_factory import ConnectionFactory
from mlflow.azure.exceptions import ConnectionError, ConfigurationError

logger = logging.getLogger(__name__)


class AzurePostgreSQLStore:
    """MLflow tracking store with Azure Managed Identity authentication for PostgreSQL."""

    def __init__(
        self,
        db_uri: str,
        default_artifact_root: Optional[str] = None,
        config: Optional[AzureAuthConfig] = None,
    ):
        """Initialize the Azure PostgreSQL tracking store.

        Args:
            db_uri: Database connection URI
            default_artifact_root: Default artifact root path
            config: Azure authentication configuration
        """
        # Import MLflow classes only when needed to avoid circular imports
        from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

        self.config = config or AzureAuthConfig()
        self.connection_factory = ConnectionFactory(self.config)

        logger.info(
            "Initializing Azure PostgreSQL tracking store: azure_auth_enabled=%s, auth_method=%s",
            self.config.should_use_azure_auth,
            self.config.auth_method.value,
        )

        # Create engine with Azure authentication
        try:
            engine = self._create_engine(db_uri)
        except Exception as e:
            logger.error("Failed to create database engine: %s", str(e))
            raise ConnectionError(f"Failed to initialize tracking store: {e}") from e

        # Initialize internal SQLAlchemy store
        try:
            # Convert azure-postgres:// scheme to postgresql:// for MLflow compatibility
            # MLflow only recognizes standard database schemes
            standard_uri = db_uri.replace("azure-postgres://", "postgresql://", 1)
            
            # Remove auth_method parameter from URI for MLflow
            from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
            parsed = urlparse(standard_uri)
            query_params = parse_qs(parsed.query)
            clean_query_params = {k: v for k, v in query_params.items() if k != "auth_method"}
            clean_query = urlencode(clean_query_params, doseq=True)
            clean_parsed = parsed._replace(query=clean_query)
            clean_uri = urlunparse(clean_parsed)

            # Add our Azure-enabled engine to the engine map BEFORE initializing SqlAlchemyStore
            # This prevents MLflow from creating its own engine and ensures migrations work correctly
            SqlAlchemyStore._db_uri_sql_alchemy_engine_map[clean_uri] = engine

            # Initialize with the clean PostgreSQL URI so MLflow's validation passes
            self._store = SqlAlchemyStore(clean_uri, default_artifact_root)

            logger.info("Azure PostgreSQL tracking store initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize SQLAlchemy store: %s", str(e))
            raise

    def __getattr__(self, name):
        """Delegate all method calls to the internal SQLAlchemy store."""
        return getattr(self._store, name)

    def _create_engine(self, db_uri: str) -> Engine:
        """Create database engine with Azure authentication.

        Args:
            db_uri: Database connection URI

        Returns:
            Configured SQLAlchemy engine
        """
        # Use our connection factory to create the engine
        return self.connection_factory.create_engine(
            db_uri,
            echo=self.config.enable_debug_logging,
        )

    @classmethod
    def from_config(
        cls,
        db_uri: Optional[str] = None,
        default_artifact_root: Optional[str] = None,
        config: Optional[AzureAuthConfig] = None,
    ) -> "AzurePostgreSQLStore":
        """Create store from configuration.

        Args:
            db_uri: Database URI (defaults to MLFLOW_BACKEND_STORE_URI)
            default_artifact_root: Artifact root (defaults to MLFLOW_DEFAULT_ARTIFACT_ROOT)
            config: Azure authentication configuration

        Returns:
            Configured store instance
        """
        # Get URI from environment if not provided
        if not db_uri:
            db_uri = os.getenv("MLFLOW_BACKEND_STORE_URI")
            if not db_uri:
                raise ConfigurationError(
                    "Database URI must be provided or set in MLFLOW_BACKEND_STORE_URI"
                )

        # Get artifact root from environment if not provided
        if not default_artifact_root:
            default_artifact_root = os.getenv("MLFLOW_DEFAULT_ARTIFACT_ROOT")

        # Create config if not provided
        if not config:
            config = AzureAuthConfig()

        return cls(db_uri, default_artifact_root, config)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        try:
            if hasattr(self, "engine") and self.engine:
                self.engine.dispose()
                logger.debug("Database engine disposed")
        except Exception as e:
            logger.warning("Error during cleanup: %s", str(e))


def create_store(store_uri: str, artifact_uri: Optional[str] = None):
    """Factory function for creating tracking store with Azure authentication.

    This function is called by MLflow's plugin system and automatically detects
    if Azure authentication should be used based on environment variables.

    Args:
        store_uri: Database connection URI
        artifact_uri: Artifact storage URI

    Returns:
        Azure-enabled tracking store if Azure auth is enabled, standard store otherwise
    """
    # Check if Azure authentication should be used
    config = AzureAuthConfig()
    
    # IMPORTANT: Do NOT auto-detect Azure PostgreSQL based on hostname
    # Only use Azure auth if explicitly enabled through configuration
    # The MLFLOW_AZURE_AUTH_ENABLED flag should be the single source of truth
    
    # Check if URI explicitly requests Azure auth (via scheme or parameter)
    explicit_azure_request = (
        store_uri.startswith("azure-postgres://") or
        "auth_method=managed_identity" in store_uri
    )
    
    # If auth is not enabled via config and not explicitly requested in URI, skip Azure auth
    if not config.auth_enabled and not explicit_azure_request:
        # Log for debugging
        logger.debug(
            "Azure auth not enabled: config.auth_enabled=%s, explicit_request=%s",
            config.auth_enabled,
            explicit_azure_request
        )
        return None
    
    # If we reach here, Azure auth is either enabled or explicitly requested
    # Make sure config reflects this
    if explicit_azure_request and not config.auth_enabled:
        config.auth_enabled = True
        if not config.auth_method or config.auth_method == AuthMethod.SQL_AUTH:
            config.auth_method = AuthMethod.MANAGED_IDENTITY
    
    if config.should_use_azure_auth:
        logger.info("Creating Azure-enabled tracking store via plugin: auth_method=%s", config.auth_method.value)
        
        try:
            # Import MLflow store
            from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

            # Ensure we have a postgresql:// scheme
            standard_uri = store_uri.replace("azure-postgres://", "postgresql://", 1)
            
            # Clean the URI
            from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
            parsed = urlparse(standard_uri)
            query_params = parse_qs(parsed.query)
            clean_query_params = {k: v for k, v in query_params.items() if k != "auth_method"}
            clean_query = urlencode(clean_query_params, doseq=True)
            clean_parsed = parsed._replace(query=clean_query)
            clean_uri = urlunparse(clean_parsed)

            # Check if we already have this engine cached
            if clean_uri not in SqlAlchemyStore._db_uri_sql_alchemy_engine_map:
                # Create our Azure-enabled engine and cache it
                connection_factory = ConnectionFactory(config)
                engine = connection_factory.create_engine(standard_uri)
                SqlAlchemyStore._db_uri_sql_alchemy_engine_map[clean_uri] = engine
                logger.debug("Cached new Azure-enabled engine")
            else:
                logger.debug("Reusing cached Azure-enabled engine")

            # Create the standard MLflow store - it will use our cached engine
            # Ensure artifact_uri is not None or empty to avoid FileNotFoundError
            logger.debug(f"Raw artifact_uri: {repr(artifact_uri)}")
            safe_artifact_uri = artifact_uri if (artifact_uri and artifact_uri.strip()) else "./mlflow-artifacts"
            logger.debug(f"Safe artifact_uri: {repr(safe_artifact_uri)}")
            return SqlAlchemyStore(clean_uri, safe_artifact_uri)

        except Exception as e:
            logger.error("Failed to create Azure-enabled store via plugin: %s", str(e))
            raise
    else:
        # Azure auth not enabled, fall back to standard MLflow behavior
        logger.info("Azure authentication not enabled, using standard tracking store")
        from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
        # Ensure artifact_uri is not None or empty to avoid FileNotFoundError
        logger.debug(f"Fallback raw artifact_uri: {repr(artifact_uri)}")
        safe_artifact_uri = artifact_uri if (artifact_uri and artifact_uri.strip()) else "./mlflow-artifacts"
        logger.debug(f"Fallback safe artifact_uri: {repr(safe_artifact_uri)}")
        return SqlAlchemyStore(store_uri, safe_artifact_uri)


# For backward compatibility and testing
def get_store(store_uri: str, artifact_uri: Optional[str] = None) -> AzurePostgreSQLStore:
    """Get a configured Azure PostgreSQL tracking store.

    Args:
        store_uri: Database connection URI
        artifact_uri: Artifact storage URI

    Returns:
        Configured store instance
    """
    return create_store(store_uri, artifact_uri)


# Utility functions for manual store creation
def get_azure_tracking_store(
    db_uri: str, artifact_uri: Optional[str] = None, config: Optional[AzureAuthConfig] = None
) -> AzurePostgreSQLStore:
    """
    Create a tracking store with specific configuration.

    Args:
        db_uri: Database connection URI
        artifact_uri: Artifact storage URI
        config: Authentication configuration

    Returns:
        Configured AzurePostgreSQLStore
    """
    if config:
        # Temporarily override global config
        import os

        original_env = {}

        try:
            # Save original environment
            for key in [
                "MLFLOW_AZURE_AUTH_ENABLED",
                "MLFLOW_AZURE_AUTH_METHOD",
                "AZURE_CLIENT_ID",
                "AZURE_TENANT_ID",
            ]:
                if key in os.environ:
                    original_env[key] = os.environ[key]

            # Set new environment
            os.environ["MLFLOW_AZURE_AUTH_ENABLED"] = str(config.auth_enabled).lower()
            os.environ["MLFLOW_AZURE_AUTH_METHOD"] = config.auth_method.value
            if config.client_id:
                os.environ["AZURE_CLIENT_ID"] = config.client_id
            if config.tenant_id:
                os.environ["AZURE_TENANT_ID"] = config.tenant_id

            # Create store with new config
            store = AzurePostgreSQLStore(db_uri, artifact_uri)

        finally:
            # Restore original environment
            for key, value in original_env.items():
                os.environ[key] = value

            # Remove keys that weren't originally set
            for key in [
                "MLFLOW_AZURE_AUTH_ENABLED",
                "MLFLOW_AZURE_AUTH_METHOD",
                "AZURE_CLIENT_ID",
                "AZURE_TENANT_ID",
            ]:
                if key not in original_env and key in os.environ:
                    del os.environ[key]

        return store

    else:
        return AzurePostgreSQLStore(db_uri, artifact_uri)


def test_azure_connection(db_uri: str, config: Optional[AzureAuthConfig] = None) -> bool:
    """
    Test Azure database connection.

    Args:
        db_uri: Database connection URI
        config: Authentication configuration

    Returns:
        True if connection successful, False otherwise
    """
    test_config = config or AzureAuthConfig()
    connection_factory = ConnectionFactory(test_config)

    return connection_factory.test_connection(db_uri)