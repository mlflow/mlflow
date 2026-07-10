"""Microsoft Entra ID (Azure AD) token acquisition for Azure OpenAI.

``azure-identity`` is an optional dependency and is imported lazily so that this
module can be imported without it installed.
"""

import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mlflow.exceptions import MlflowException

if TYPE_CHECKING:
    from azure.core.credentials import AccessToken

_AZURE_COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"
_TOKEN_EXPIRY_BUFFER_SECONDS = 60


@dataclass
class _CachedCredential:
    credential: Any
    token: "AccessToken | None" = None


# Gateway providers are constructed per request, so credentials and their tokens are
# cached at module level to avoid re-authenticating on every request.
_credential_cache: dict[tuple[str | None, str | None, str | None], _CachedCredential] = {}
_cache_lock = threading.Lock()


def _build_credential(client_id: str | None, tenant_id: str | None, client_secret: str | None):
    try:
        from azure.identity import ClientSecretCredential, DefaultAzureCredential
    except ImportError as e:
        raise MlflowException(
            "Using Microsoft Entra ID authentication for Azure OpenAI requires the "
            "`azure-identity` package. Install it with `pip install azure-identity` or "
            "`pip install mlflow[azure]`."
        ) from e

    if client_id and tenant_id and client_secret:
        return ClientSecretCredential(
            tenant_id=tenant_id, client_id=client_id, client_secret=client_secret
        )
    return DefaultAzureCredential()


def get_azure_openai_token(
    client_id: str | None = None,
    tenant_id: str | None = None,
    client_secret: str | None = None,
) -> str:
    """Return a valid Microsoft Entra ID bearer token for Azure Cognitive Services.

    When ``client_id``, ``tenant_id``, and ``client_secret`` are all provided, a
    ``ClientSecretCredential`` (service principal) is used; otherwise
    ``DefaultAzureCredential`` resolves credentials from the environment (environment
    variables, managed identity, Azure CLI login, etc.). Tokens are refreshed when they
    are within ``_TOKEN_EXPIRY_BUFFER_SECONDS`` of expiry.
    """
    cache_key = (client_id, tenant_id, client_secret)
    with _cache_lock:
        entry = _credential_cache.get(cache_key)
        if entry is None:
            entry = _CachedCredential(_build_credential(client_id, tenant_id, client_secret))
            _credential_cache[cache_key] = entry

        if (
            entry.token is None
            or entry.token.expires_on < time.time() + _TOKEN_EXPIRY_BUFFER_SECONDS
        ):
            from azure.core.exceptions import ClientAuthenticationError

            try:
                entry.token = entry.credential.get_token(_AZURE_COGNITIVE_SERVICES_SCOPE)
            except ClientAuthenticationError as e:
                raise MlflowException(
                    "Unable to acquire a Microsoft Entra ID token for Azure OpenAI due to "
                    f"the following error: {e.message}"
                ) from e

        return entry.token.token


def _reset_credential_cache() -> None:
    with _cache_lock:
        _credential_cache.clear()
