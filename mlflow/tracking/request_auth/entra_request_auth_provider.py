"""Request auth provider for Microsoft Entra ID (formerly Azure Active Directory).

This module provides an auth plugin activated via ``MLFLOW_TRACKING_AUTH=entra``.
It adds an ``Authorization: Bearer <token>`` header to outgoing requests, acquiring
tokens with :class:`azure.identity.DefaultAzureCredential`. Tokens are cached and
refreshed in memory by ``azure-identity``, so requests keep working after the
initial token expires.

The token scope is read from the ``MLFLOW_ENTRA_ID_SCOPE`` environment variable,
e.g. ``api://<client-id>/.default`` for a tracking server registered as an Entra ID
application.

Requires the ``azure-identity`` package. Install it with:
    pip install azure-identity
"""

import logging
import threading

from mlflow.environment_variables import MLFLOW_ENTRA_ID_SCOPE
from mlflow.exceptions import MlflowException
from mlflow.tracking.request_auth.abstract_request_auth_provider import RequestAuthProvider

_logger = logging.getLogger(__name__)

AUTHORIZATION_HEADER_NAME = "Authorization"

_credential = None
_credential_lock = threading.Lock()


def _check_azure_identity_installed():
    try:
        from azure.identity import DefaultAzureCredential  # noqa: F401
    except ImportError:
        raise MlflowException(
            "The 'azure-identity' package is required to use Microsoft Entra ID "
            "authentication (MLFLOW_TRACKING_AUTH=entra), but it is not installed. "
            "Install it with: pip install azure-identity"
        )


def _get_entra_scope() -> str:
    scope = MLFLOW_ENTRA_ID_SCOPE.get()
    if not scope:
        raise MlflowException(
            "No Microsoft Entra ID token scope is configured. Set the "
            f"{MLFLOW_ENTRA_ID_SCOPE.name} environment variable to the scope of your "
            "MLflow tracking server's app registration, e.g. 'api://<client-id>/.default'."
        )
    return scope


def _get_credential():
    """Return a process-wide ``DefaultAzureCredential``, creating it lazily.

    ``DefaultAzureCredential`` caches access tokens in memory and refreshes them
    shortly before they expire, so reusing a single instance avoids
    re-authenticating on every request.
    """
    global _credential
    with _credential_lock:
        if _credential is None:
            from azure.identity import DefaultAzureCredential

            _credential = DefaultAzureCredential()
        return _credential


def _get_token() -> str:
    """Acquire a Microsoft Entra ID access token for the configured scope."""
    scope = _get_entra_scope()
    try:
        access_token = _get_credential().get_token(scope)
    except MlflowException:
        raise
    except Exception as e:
        raise MlflowException(
            f"Failed to acquire a Microsoft Entra ID token for scope '{scope}': {e}. "
            "Ensure that valid Azure credentials are available, e.g. by running "
            "`az login`, or by setting the AZURE_CLIENT_ID, AZURE_TENANT_ID and "
            "AZURE_CLIENT_SECRET environment variables."
        )
    return access_token.token


class EntraAuth:
    """Custom authentication class for Microsoft Entra ID.

    This class is callable and will be invoked by the requests library
    to add an ``Authorization`` header to each request.
    """

    def __call__(self, request):
        """Add a Microsoft Entra ID bearer token to the request.

        Args:
            request: The prepared request object from the requests library.

        Returns:
            The modified request object with the authorization header.

        Raises:
            MlflowException: If no scope is configured or no token can be acquired.
        """
        if AUTHORIZATION_HEADER_NAME not in request.headers:
            request.headers[AUTHORIZATION_HEADER_NAME] = f"Bearer {_get_token()}"
        return request


class EntraRequestAuthProvider(RequestAuthProvider):
    """Provides bearer token authentication backed by Microsoft Entra ID.

    This provider adds an ``Authorization`` header with a token acquired via
    ``azure.identity.DefaultAzureCredential`` for the scope configured through
    ``MLFLOW_ENTRA_ID_SCOPE``. Use ``MLFLOW_TRACKING_AUTH=entra`` to enable.
    """

    def get_name(self) -> str:
        return "entra"

    def get_auth(self):
        _check_azure_identity_installed()
        return EntraAuth()
