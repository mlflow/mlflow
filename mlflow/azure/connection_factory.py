"""Database connection factory with Azure authentication support."""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any
from urllib.parse import urlparse, urlencode, parse_qs

from sqlalchemy import create_engine, event, Engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.pool import QueuePool

from mlflow.azure.auth_handler import AzureAuthHandler
from mlflow.azure.config import AzureAuthConfig
from mlflow.azure.exceptions import ConnectionError, ConfigurationError
from mlflow.azure.utils import sanitize_connection_string_for_logging

logger = logging.getLogger(__name__)


class ConnectionFactory:
    """Factory for creating database connections with Azure authentication."""

    def __init__(self, config: Optional[AzureAuthConfig] = None):
        """Initialize the connection factory.

        Args:
            config: Azure authentication configuration
        """
        self.config = config or AzureAuthConfig()
        self.auth_handler = (
            AzureAuthHandler(self.config) if self.config.should_use_azure_auth else None
        )

        logger.info(
            "Initialized connection factory: azure_auth_enabled=%s, auth_method=%s, auth_flag=%s",
            self.config.should_use_azure_auth,
            self.config.auth_method.value if self.config.should_use_azure_auth else "sql_auth",
            self.config.auth_enabled,
        )

    def create_engine(
        self,
        database_uri: str,
        **engine_kwargs: Any,
    ) -> Engine:
        """Create a SQLAlchemy engine with Azure authentication support.

        Args:
            database_uri: Database connection URI
            **engine_kwargs: Additional engine configuration

        Returns:
            Configured SQLAlchemy engine

        Raises:
            ConnectionError: If connection configuration fails
            ConfigurationError: If the URI is invalid
        """
        # Check if auth_method is in URI
        has_auth_method_param = "auth_method=" in database_uri
        
        logger.info(
            "Creating database engine: connection_info=%s, azure_auth=%s, has_auth_param=%s, auth_enabled_env=%s",
            sanitize_connection_string_for_logging(database_uri),
            self.config.should_use_azure_auth,
            has_auth_method_param,
            self.config.auth_enabled,
        )
        
        # Warn if there's a mismatch between URI parameter and configuration
        if has_auth_method_param and not self.config.should_use_azure_auth:
            logger.warning(
                "Database URI contains auth_method parameter but Azure authentication is not enabled. "
                "The auth_method parameter will be ignored. Enable Azure authentication in Helm values to use Managed Identity."
            )

        try:
            # Parse and validate the URI
            parsed_uri = self._parse_and_validate_uri(database_uri)

            # Prepare engine configuration
            engine_config = self._prepare_engine_config(parsed_uri, **engine_kwargs)

            # Create the engine
            engine = create_engine(**engine_config)

            # Set up Azure authentication if enabled
            if self.config.should_use_azure_auth and self.auth_handler:
                self._setup_azure_auth_events(engine)

            logger.info("Database engine created successfully")
            return engine

        except Exception as e:
            logger.error("Failed to create database engine: %s", str(e))
            raise ConnectionError(f"Failed to create database engine: {e}") from e

    def _parse_and_validate_uri(self, database_uri: str) -> Dict[str, Any]:
        """Parse and validate database URI.

        Args:
            database_uri: Database connection URI

        Returns:
            Parsed URI components

        Raises:
            ConfigurationError: If URI is invalid
        """
        try:
            # Handle azure-postgres:// scheme by converting to postgresql://
            if database_uri.startswith("azure-postgres://"):
                database_uri = database_uri.replace("azure-postgres://", "postgresql://", 1)

            url = make_url(database_uri)

            if url.drivername not in ["postgresql", "postgresql+psycopg2"]:
                raise ConfigurationError(f"Unsupported database driver: {url.drivername}")

            return {
                "drivername": url.drivername,
                "username": url.username,
                "password": url.password,
                "host": url.host,
                "port": url.port or 5432,
                "database": url.database,
                "query": dict(url.query),
            }

        except Exception as e:
            raise ConfigurationError(f"Invalid database URI: {e}") from e

    def _prepare_engine_config(
        self,
        parsed_uri: Dict[str, Any],
        **engine_kwargs: Any,
    ) -> Dict[str, Any]:
        """Prepare SQLAlchemy engine configuration.

        Args:
            parsed_uri: Parsed database URI components
            **engine_kwargs: Additional engine configuration

        Returns:
            Engine configuration dictionary
        """
        # Build connection URL
        url_parts = {
            "drivername": parsed_uri["drivername"],
            "username": parsed_uri["username"],
            "host": parsed_uri["host"],
            "port": parsed_uri["port"],
            "database": parsed_uri["database"],
        }

        # Handle password for non-Azure auth
        if not self.config.should_use_azure_auth:
            url_parts["password"] = parsed_uri["password"]

        # Prepare query parameters - remove custom auth parameters
        query_params = parsed_uri["query"].copy()
        
        # Remove custom parameters that psycopg2 doesn't understand
        query_params.pop("auth_method", None)

        # Ensure SSL for Azure connections
        if self.config.should_use_azure_auth:
            query_params.setdefault("sslmode", "require")

        url_parts["query"] = query_params

        # Create URL
        from sqlalchemy.engine.url import URL

        url = URL.create(**url_parts)

        # Default engine configuration
        config = {
            "url": url,
            "echo": engine_kwargs.pop("echo", False),
            "echo_pool": engine_kwargs.pop("echo_pool", False),
            "poolclass": engine_kwargs.pop("poolclass", QueuePool),
            "pool_size": engine_kwargs.pop("pool_size", self.config.pool_size),
            "max_overflow": engine_kwargs.pop("max_overflow", self.config.max_overflow),
            "pool_recycle": engine_kwargs.pop("pool_recycle", self.config.pool_recycle),
            "pool_pre_ping": engine_kwargs.pop("pool_pre_ping", self.config.pool_pre_ping),
            "connect_args": engine_kwargs.pop("connect_args", {}),
        }

        # Add connection timeout
        config["connect_args"].setdefault("connect_timeout", self.config.connection_timeout)

        # Add any remaining engine kwargs
        config.update(engine_kwargs)

        return config

    def _setup_azure_auth_events(self, engine: Engine) -> None:
        """Set up SQLAlchemy events for Azure authentication.

        Args:
            engine: SQLAlchemy engine to configure
        """
        if not self.auth_handler:
            return

        @event.listens_for(engine, "do_connect")
        def do_connect(dialect, conn_rec, cargs, cparams):
            """Inject Azure token before connecting."""
            try:
                # Get fresh token
                token = self.auth_handler.get_access_token()

                # Inject token as password
                cparams["password"] = token

                logger.debug("Injected Azure token for database connection")

            except Exception as e:
                logger.error("Failed to inject Azure token: %s", str(e))
                raise ConnectionError(f"Failed to inject Azure token: {e}") from e

        @event.listens_for(engine, "connect")
        def connect(dbapi_connection, connection_record):
            """Handle post-connection setup."""
            logger.debug("Database connection established")

        @event.listens_for(engine, "checkout")
        def checkout(dbapi_connection, connection_record, connection_proxy):
            """Handle connection checkout from pool."""
            # Refresh token if needed before using connection
            if self.auth_handler:
                try:
                    self.auth_handler.refresh_token_if_needed()
                except Exception as e:
                    logger.warning("Token refresh failed during checkout: %s", str(e))

    def test_connection(self, database_uri: str) -> bool:
        """Test database connection.

        Args:
            database_uri: Database connection URI

        Returns:
            True if connection successful, False otherwise
        """
        try:
            engine = self.create_engine(database_uri)

            with engine.connect() as conn:
                # Simple test query
                result = conn.execute(text("SELECT 1"))
                result.fetchone()

            logger.info("Database connection test successful")
            return True

        except Exception as e:
            logger.error("Database connection test failed: %s", str(e))
            return False

    def create_connection_url(
        self,
        host: str,
        database: str,
        username: str,
        port: int = 5432,
        password: Optional[str] = None,
        **params: Any,
    ) -> str:
        """Create a PostgreSQL connection URL.

        Args:
            host: Database host
            database: Database name
            username: Database username
            port: Database port
            password: Database password (optional for Azure auth)
            **params: Additional connection parameters

        Returns:
            PostgreSQL connection URL
        """
        # Ensure SSL for Azure connections
        if self.config.should_use_azure_auth:
            params.setdefault("sslmode", "require")

        # Build URL components
        url_parts = ["postgresql://"]

        # Add credentials
        if username:
            url_parts.append(username)
            if password and not self.config.should_use_azure_auth:
                url_parts.append(f":{password}")
            url_parts.append("@")

        # Add host and port
        url_parts.append(f"{host}:{port}")

        # Add database
        if database:
            url_parts.append(f"/{database}")

        # Add query parameters
        if params:
            query_string = urlencode(params)
            url_parts.append(f"?{query_string}")

        url = "".join(url_parts)
        logger.debug("Created connection URL: %s", sanitize_connection_string_for_logging(url))

        return url


class PoolManager:
    """Manages connection pools with token refresh capabilities."""

    def __init__(self, connection_factory: ConnectionFactory):
        self.connection_factory = connection_factory
        self._engines: Dict[str, Engine] = {}

    def get_engine(self, connection_string: str, **kwargs) -> Engine:
        """Get or create engine for connection string."""
        # Use connection string as key (could be enhanced with hashing)
        key = connection_string

        if key not in self._engines:
            self._engines[key] = self.connection_factory.create_engine(connection_string, **kwargs)

        return self._engines[key]

    def refresh_connections(self):
        """Refresh all connection pools to get new tokens."""
        for engine in self._engines.values():
            try:
                # Dispose current connections to force new ones with fresh tokens
                engine.dispose()
                logger.info("Refreshed connection pool")
            except Exception as e:
                logger.error("Failed to refresh connection pool: %s", str(e))

    def close_all(self):
        """Close all connection pools."""
        for engine in self._engines.values():
            try:
                engine.dispose()
            except Exception as e:
                logger.error("Failed to dispose engine: %s", str(e))

        self._engines.clear()
        logger.info("Closed all connection pools")