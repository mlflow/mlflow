"""Custom exceptions for Azure authentication."""

from __future__ import annotations


class AzureAuthError(Exception):
    """Base exception for Azure authentication errors."""

    pass


class TokenAcquisitionError(AzureAuthError):
    """Raised when token acquisition fails."""

    pass


class ConnectionError(AzureAuthError):
    """Raised when database connection fails."""

    pass


class ConfigurationError(AzureAuthError):
    """Raised when configuration is invalid."""

    pass