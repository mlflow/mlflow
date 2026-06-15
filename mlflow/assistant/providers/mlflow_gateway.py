"""MLflow AI Gateway preset of the OpenAI-compatible assistant provider.

This preset proxies through the same MLflow tracking server that hosts the
assistant API, so the chat URL is derived from the ``tracking_uri`` rather
than a stand-alone ``base_url``. Model enumeration is handled by the
frontend (which calls the gateway's own endpoints API), so
``list_models_fn`` is intentionally omitted.
"""

from typing import ClassVar

from mlflow.assistant.providers.openai_compatible import OpenAICompatibleProvider


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
        )
