"""Configuration management for Azure authentication."""

from __future__ import annotations

from enum import Enum
import os
from typing import Optional


class AuthMethod(str, Enum):
    """Supported authentication methods."""

    MANAGED_IDENTITY = "managed_identity"
    SERVICE_PRINCIPAL = "service_principal"
    SQL_AUTH = "sql_auth"
    DEFAULT_AZURE_CREDENTIAL = "default_azure_credential"


class AzureAuthConfig:
    """Configuration for Azure authentication.

    This configuration can be loaded from environment variables or
    provided programmatically.
    """

    def __init__(
        self,
        auth_enabled: Optional[bool] = None,
        auth_method: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        tenant_id: Optional[str] = None,
        token_scope: str = "https://ossrdbms-aad.database.windows.net/.default",
        connection_timeout: int = 30,
        token_refresh_buffer: int = 300,
        pool_size: int = 20,
        max_overflow: int = 10,
        pool_recycle: int = 3600,
        pool_pre_ping: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_backoff: float = 2.0,
        enable_debug_logging: bool = False,
    ):
        """Initialize Azure authentication configuration."""
        
        # Core settings - use provided values or environment variables
        self.auth_enabled = (
            auth_enabled 
            if auth_enabled is not None 
            else os.getenv("MLFLOW_AZURE_AUTH_ENABLED", "false").lower() == "true"
        )
        
        self.auth_method = AuthMethod(
            auth_method.lower() 
            if auth_method 
            else os.getenv("MLFLOW_AZURE_AUTH_METHOD", "sql_auth").lower()
        )

        # Azure specific settings - check both provided values and environment
        self.client_id = client_id or os.getenv("AZURE_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("AZURE_CLIENT_SECRET") 
        self.tenant_id = tenant_id or os.getenv("AZURE_TENANT_ID")

        # Connection settings
        self.token_scope = token_scope
        self.connection_timeout = connection_timeout
        self.token_refresh_buffer = token_refresh_buffer

        # Pool settings
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_recycle = pool_recycle
        self.pool_pre_ping = pool_pre_ping

        # Retry settings
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff = retry_backoff

        # Logging
        self.enable_debug_logging = (
            enable_debug_logging
            or os.getenv("MLFLOW_AZURE_DEBUG", "false").lower() == "true"
        )

        # IMPORTANT: Do not automatically enable Managed Identity based on other env vars
        # The auth_enabled flag should be the single source of truth
        # This prevents unintended authentication attempts when Helm hasn't enabled it
        
        # Log configuration decision for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(
            "AzureAuthConfig initialized: auth_enabled=%s, auth_method=%s, source=%s",
            self.auth_enabled,
            self.auth_method.value,
            "explicit" if auth_enabled is not None else "env_var"
        )

        # Validate configuration
        self._validate()

    def _validate(self):
        """Validate configuration settings."""
        if self.auth_method == AuthMethod.SERVICE_PRINCIPAL:
            if not self.client_secret:
                raise ValueError("client_secret is required for service principal authentication")
            if not self.client_id:
                raise ValueError("client_id is required for service principal authentication")
            if not self.tenant_id:
                raise ValueError("tenant_id is required for service principal authentication")

    @property
    def should_use_azure_auth(self) -> bool:
        """Check if Azure authentication should be used."""
        return self.auth_enabled and self.auth_method != AuthMethod.SQL_AUTH

    @property
    def requires_token_refresh(self) -> bool:
        """Check if the auth method requires token refresh."""
        return self.auth_method in [
            AuthMethod.MANAGED_IDENTITY,
            AuthMethod.SERVICE_PRINCIPAL,
            AuthMethod.DEFAULT_AZURE_CREDENTIAL,
        ]