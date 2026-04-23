from mlflow.gateway.config import _OpenAICompatibleConfig
from mlflow.gateway.providers.openai_compatible import OpenAICompatibleProvider


class PortkeyProvider(OpenAICompatibleProvider):
    DISPLAY_NAME = "Portkey"
    CONFIG_TYPE = _OpenAICompatibleConfig

    @property
    def headers(self) -> dict[str, str]:
        return {"x-portkey-api-key": self._api_key}
