"""MLflow AI Gateway preset of the OpenAI-compatible assistant provider."""

import base64
import logging
from typing import ClassVar

from mlflow.assistant.config import get_config_user
from mlflow.assistant.providers.openai_compatible import OpenAICompatibleProvider
from mlflow.environment_variables import _MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN
from mlflow.gateway.constants import MLFLOW_GATEWAY_AUTH_HEADER

_logger = logging.getLogger(__name__)


class MlflowGatewayProvider(OpenAICompatibleProvider):
    """OpenAI-compatible provider backed by an in-server MLflow AI Gateway."""

    # Provider name for the in-server MLflow AI Gateway backend. The frontend
    # mirrors this literal in `server/js/src/assistant/constants.ts`
    # (GATEWAY_PROVIDER_ID); keep the two in sync.
    GATEWAY_PROVIDER_NAME: ClassVar[str] = "mlflow_gateway"

    @staticmethod
    def _build_chat_url(_base_url: str | None, tracking_uri: str) -> str | None:
        """The in-server MLflow Gateway is reachable through the same MLflow
        server, so the chat URL is derived from the tracking URI instead of
        a separate base_url stored in config.
        """
        if not tracking_uri:
            return None
        return f"{tracking_uri.rstrip('/')}/gateway/mlflow/v1/chat/completions"

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
            # Deployable/remote: history rides with the client so the server stays stateless
            # and any replica can serve any turn. Ollama is localhost-only and keeps the
            # stateful server-side session path.
            client_carries_history=True,
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
        return headers

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
