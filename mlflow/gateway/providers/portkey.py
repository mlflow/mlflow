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
