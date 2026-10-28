from mlflow.gateway.config import PortkeyConfig
from mlflow.gateway.providers.openai_compatible import OpenAICompatibleProvider


class PortkeyProvider(OpenAICompatibleProvider):
    DISPLAY_NAME = "Portkey"
    CONFIG_TYPE = PortkeyConfig
    DEFAULT_API_BASE = "https://api.portkey.ai/v1"

    @property
    def headers(self) -> dict[str, str]:
        # Portkey requires a routing target in addition to the Portkey API key:
        # a provider slug (x-portkey-provider), a saved config (x-portkey-config),
        # or an "@integration/model" reference in the model name. Bare provider
        # slugs also need the upstream key, passed via the Authorization header.
        headers = {"x-portkey-api-key": self._api_key}
        if portkey_provider := self._provider_config.portkey_provider:
            headers["x-portkey-provider"] = portkey_provider
        if portkey_config := self._provider_config.portkey_config:
            headers["x-portkey-config"] = portkey_config
        if provider_api_key := self._provider_config.provider_api_key:
            headers["Authorization"] = f"Bearer {provider_api_key}"
        return headers

    def _get_headers(
        self,
        headers: dict[str, str] | None = None,
    ) -> dict[str, str]:
        result = super()._get_headers(headers)
        # On the passthrough routes, the base class lets a subscription tool's
        # own Authorization header (Claude Code, Codex, Gemini CLI) pass through
        # to the upstream. For Portkey, Authorization carries the configured
        # upstream `provider_api_key`, not a Portkey credential, so an explicitly
        # configured key must win: a client Authorization targets MLflow, not
        # Portkey's upstream provider. When no key is configured, the base
        # behavior is preserved.
        if self._provider_config.provider_api_key:
            # drop any client-provided Authorization (any casing) so it cannot
            # shadow the configured upstream key
            for key in [k for k in result if k.lower() == "authorization"]:
                del result[key]
            result["Authorization"] = f"Bearer {self._provider_config.provider_api_key}"
        return result
