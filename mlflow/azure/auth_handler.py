"""Azure authentication handler with advanced token management."""

from __future__ import annotations

import threading
import logging
from datetime import datetime, timedelta
from typing import Optional

from azure.core.credentials import AccessToken
from azure.core.exceptions import ClientAuthenticationError
from azure.identity import (
    DefaultAzureCredential,
    ManagedIdentityCredential,
    ClientSecretCredential,
)

from mlflow.azure.config import AzureAuthConfig, AuthMethod
from mlflow.azure.exceptions import TokenAcquisitionError
from mlflow.azure.utils import is_token_expired

logger = logging.getLogger(__name__)


class TokenCache:
    """Thread-safe token cache with expiration handling."""

    def __init__(self):
        self._lock = threading.RLock()
        self._token: Optional[AccessToken] = None
        self._expiry: Optional[datetime] = None

    def get_token(self, buffer_seconds: int = 300) -> Optional[AccessToken]:
        """Get cached token if it's still valid.

        Args:
            buffer_seconds: Buffer time before expiry to consider token invalid

        Returns:
            Cached token if valid, None otherwise
        """
        with self._lock:
            if self._token and not is_token_expired(self._expiry, buffer_seconds):
                logger.debug("Using cached token")
                return self._token

            if self._token:
                logger.debug("Cached token expired or expiring soon")

            return None

    def set_token(self, token: AccessToken) -> None:
        """Cache a new token.

        Args:
            token: Access token to cache
        """
        with self._lock:
            self._token = token
            # Convert timestamp to datetime
            self._expiry = datetime.fromtimestamp(token.expires_on)

            logger.debug(
                "Token cached, expires at %s (in %d seconds)",
                self._expiry.isoformat(),
                int((self._expiry - datetime.utcnow()).total_seconds()),
            )

    def clear(self) -> None:
        """Clear the cached token."""
        with self._lock:
            self._token = None
            self._expiry = None
            logger.debug("Token cache cleared")


class AzureAuthHandler:
    """Handles Azure authentication and token management for MLflow PostgreSQL connections."""

    def __init__(self, config: AzureAuthConfig):
        """Initialize the authentication handler.

        Args:
            config: Azure authentication configuration
        """
        self.config = config
        self._token_cache = TokenCache()
        self._credential = None

        if config.enable_debug_logging:
            logging.getLogger().setLevel(logging.DEBUG)

        logger.info(
            "Initialized Azure auth handler: auth_method=%s, client_id=%s",
            config.auth_method.value,
            config.client_id[:8] + "..." if config.client_id else None,
        )

    @property
    def credential(self):
        """Get or create Azure credential based on configuration."""
        if self._credential is None:
            self._credential = self._create_credential()
        return self._credential

    def _create_credential(self):
        """Create appropriate Azure credential based on auth method."""
        auth_method = self.config.auth_method

        logger.info("Creating Azure credential: auth_method=%s", auth_method.value)

        try:
            if auth_method == AuthMethod.MANAGED_IDENTITY:
                if self.config.client_id:
                    logger.info(
                        "Using user-assigned managed identity: client_id=%s", self.config.client_id
                    )
                    return ManagedIdentityCredential(client_id=self.config.client_id)
                else:
                    logger.info("Using system-assigned managed identity")
                    return ManagedIdentityCredential()

            elif auth_method == AuthMethod.SERVICE_PRINCIPAL:
                if not all(
                    [self.config.client_id, self.config.client_secret, self.config.tenant_id]
                ):
                    raise TokenAcquisitionError(
                        "Service principal authentication requires client_id, client_secret, and tenant_id"
                    )

                logger.info(
                    "Using service principal authentication: client_id=%s, tenant_id=%s",
                    self.config.client_id,
                    self.config.tenant_id,
                )
                return ClientSecretCredential(
                    tenant_id=self.config.tenant_id,
                    client_id=self.config.client_id,
                    client_secret=self.config.client_secret,
                )

            elif auth_method == AuthMethod.DEFAULT_AZURE_CREDENTIAL:
                logger.info("Using DefaultAzureCredential")
                return DefaultAzureCredential()

            else:
                raise TokenAcquisitionError(f"Unsupported authentication method: {auth_method}")

        except Exception as e:
            logger.error(
                "Failed to create Azure credential: error=%s, auth_method=%s", str(e), auth_method.value
            )
            raise TokenAcquisitionError(f"Failed to create credential: {e}") from e

    def _acquire_token_with_retry(self) -> AccessToken:
        """Acquire token with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_retries):
            try:
                logger.debug("Acquiring new token from Azure (attempt %d)", attempt + 1)
                token = self.credential.get_token(self.config.token_scope)

                logger.info(
                    "Successfully acquired token, expires in %d seconds",
                    int(token.expires_on - datetime.utcnow().timestamp()),
                )

                return token

            except ClientAuthenticationError as e:
                logger.error("Authentication failed: %s", str(e))
                raise TokenAcquisitionError(f"Authentication failed: {e}") from e

            except Exception as e:
                last_exception = e
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (self.config.retry_backoff ** attempt)
                    logger.warning(
                        "Token acquisition failed (attempt %d), retrying in %.1f seconds: %s",
                        attempt + 1, delay, str(e)
                    )
                    import time
                    time.sleep(delay)
                else:
                    logger.error("Token acquisition failed after all retries: %s", str(e))

        raise TokenAcquisitionError(f"Token acquisition failed: {last_exception}") from last_exception

    def get_access_token(self, force_refresh: bool = False) -> str:
        """Get a valid access token for PostgreSQL authentication.

        Args:
            force_refresh: Force token refresh even if cached token is valid

        Returns:
            Valid access token string

        Raises:
            TokenAcquisitionError: If token acquisition fails
        """
        if not self.config.should_use_azure_auth:
            raise TokenAcquisitionError("Azure authentication is not enabled")

        # Try to get cached token first (unless force refresh)
        if not force_refresh:
            cached_token = self._token_cache.get_token(self.config.token_refresh_buffer)
            if cached_token:
                return cached_token.token

        # Acquire new token
        try:
            new_token = self._acquire_token_with_retry()
            self._token_cache.set_token(new_token)
            return new_token.token

        except Exception as e:
            logger.error("Failed to acquire access token: %s", str(e))
            raise

    def refresh_token_if_needed(self) -> bool:
        """Refresh token if it's expiring soon.

        Returns:
            True if token was refreshed, False otherwise
        """
        if not self.config.should_use_azure_auth:
            return False

        cached_token = self._token_cache.get_token(self.config.token_refresh_buffer)
        if cached_token:
            return False  # Token is still valid

        try:
            self.get_access_token(force_refresh=True)
            logger.info("Token refreshed proactively")
            return True
        except Exception as e:
            logger.warning("Proactive token refresh failed: %s", str(e))
            return False

    def clear_token_cache(self) -> None:
        """Clear the token cache."""
        self._token_cache.clear()
        logger.info("Token cache cleared")

    def is_token_valid(self) -> bool:
        """Check if current cached token is valid.

        Returns:
            True if token is valid, False otherwise
        """
        if not self.config.should_use_azure_auth:
            return True  # SQL auth doesn't use tokens

        cached_token = self._token_cache.get_token(self.config.token_refresh_buffer)
        return cached_token is not None