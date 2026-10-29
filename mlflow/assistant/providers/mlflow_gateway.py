"""MLflow AI Gateway preset of the OpenAI-compatible assistant provider."""

import base64
import contextvars
import logging
import urllib.parse
from typing import Any, ClassVar

import aiohttp
from packaging.version import Version

from mlflow.assistant.config import get_config_user
from mlflow.assistant.providers.base import ProviderNotConfiguredError
from mlflow.assistant.providers.openai_compatible import OpenAICompatibleProvider
from mlflow.environment_variables import _MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN
from mlflow.gateway.constants import MLFLOW_GATEWAY_AUTH_HEADER
from mlflow.utils.workspace_utils import WORKSPACE_HEADER_NAME

_logger = logging.getLogger(__name__)

# aiohttp's ``server_hostname`` request argument exists since 3.9.
_AIOHTTP_VERSION = Version(aiohttp.__version__)

# Where the in-server gateway is called, bound per request by the Assistant API. The URL comes
# from the server's own address (``mlflow.server.asgi_utils.get_server_self_url``): the chat
# call carries the server-internal gateway token, so it must never go to the request-derived
# tracking URI, whose host the caller controls through the ``Host`` header
# (GHSA-98p3-crmr-mxcx). The Host and workspace header values are the inbound request's own: the
# server's Host allowlist then judges the self-call exactly as it judged the request that
# triggered it (the server's address itself need not be listed under a custom
# ``--allowed-hosts``), and the gateway resolves the same workspace the caller is working in.
_server_url: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "mlflow_gateway_provider_server_url", default=None
)
_server_host_header: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "mlflow_gateway_provider_server_host_header", default=None
)
_server_workspace_header: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "mlflow_gateway_provider_server_workspace_header", default=None
)


def set_server_url(
    url: str | None, host_header: str | None = None, workspace_header: str | None = None
) -> None:
    """Bind, for the current context, the URL at which the in-server gateway is reached and the
    Host and workspace header values the call presents (the inbound request's own).
    """
    _server_url.set(url)
    _server_host_header.set(host_header)
    _server_workspace_header.set(workspace_header)


class MlflowGatewayProvider(OpenAICompatibleProvider):
    """OpenAI-compatible provider backed by an in-server MLflow AI Gateway."""

    # Provider name for the in-server MLflow AI Gateway backend. The frontend
    # mirrors this literal in `server/js/src/assistant/constants.ts`
    # (GATEWAY_PROVIDER_ID); keep the two in sync.
    GATEWAY_PROVIDER_NAME: ClassVar[str] = "mlflow_gateway"

    @staticmethod
    def _build_chat_url(_base_url: str | None, _tracking_uri: str) -> str | None:
        """The in-server MLflow Gateway is reachable through this server itself, so the chat URL
        comes from the server's own address bound with ``set_server_url``: not from a base_url
        stored in config, and not from the tracking URI, which derives from the client's Host
        header.
        """
        server_url = _server_url.get()
        if not server_url:
            return None
        return f"{server_url.rstrip('/')}/gateway/mlflow/v1/chat/completions"

    def __init__(self) -> None:
        super().__init__(
            name=self.GATEWAY_PROVIDER_NAME,
            display_name="MLflow AI Gateway",
            description=(
                "AI-powered assistant backed by an MLflow AI Gateway endpoint "
                "configured on this server."
            ),
            connection_hint=(
                "Configure an LLM chat endpoint on the MLflow AI Gateway and select it."
            ),
            chat_url_builder=self._build_chat_url,
            allows_remote_access=True,
        )

    def _auth_headers(self, api_key: str | None) -> dict[str, str]:
        headers = super()._auth_headers(api_key)
        # The in-server gateway sits behind the same authentication as the rest of the server,
        # so this internal call has to authenticate too -- otherwise it is rejected with 401 on
        # an auth-enabled server. Reuse the server's internal gateway token (generated at startup
        # and shared across all worker processes; also used by scorer and evaluation jobs) and
        # attribute the call to the current user. The credential goes in the dedicated gateway
        # auth header so it is not forwarded upstream to the LLM provider. When there is no token
        # (a server without auth), the gateway needs no credential, so no header is added.
        token = _MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN.get()
        username = get_config_user()
        if token and username:
            credential = base64.b64encode(f"{username}:{token}".encode()).decode("ascii")
            headers[MLFLOW_GATEWAY_AUTH_HEADER] = f"Basic {credential}"
        if host_header := _server_host_header.get():
            headers["Host"] = host_header
        if workspace_header := _server_workspace_header.get():
            headers[WORKSPACE_HEADER_NAME] = workspace_header
        return headers

    def _request_kwargs(self, chat_url: str) -> dict[str, Any]:
        # On a server that terminates TLS itself the self-call is HTTPS to the server's own
        # address, whose certificate names the host clients connect to, not that address. Verify
        # it against the inbound Host name, which the client's own connection already matched;
        # the destination stays pinned, so a name the certificate does not carry only fails the
        # handshake.
        host_header = _server_host_header.get()
        if not chat_url.startswith("https://") or not host_header:
            return {}
        if _AIOHTTP_VERSION < Version("3.9"):
            raise ProviderNotConfiguredError(
                "Calling a server that terminates TLS itself needs aiohttp>=3.9 (found "
                f"{aiohttp.__version__}). Upgrade aiohttp, or terminate TLS in a proxy in front "
                "of the server."
            )
        hostname = urllib.parse.urlsplit(f"//{host_header}").hostname
        return {"server_hostname": hostname} if hostname else {}

    @staticmethod
    def _list_endpoints():
        from mlflow.tracking._tracking_service.utils import _get_store

        store = _get_store()
        try:
            return store.list_gateway_endpoints()
        except (AttributeError, NotImplementedError):
            return []

    def list_models(self, base_url: str | None = None, api_key: str | None = None) -> list[str]:
        return sorted(endpoint.name for endpoint in self._list_endpoints() if endpoint.name)

    def is_available(self) -> bool:
        try:
            return bool(self.list_models())
        except Exception:
            _logger.debug("Failed to list gateway endpoints", exc_info=True)
            return False
