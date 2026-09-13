from mlflow.gateway.config import _OpenAICompatibleConfig
from mlflow.gateway.providers.openai_compatible import OpenAICompatibleProvider


class CrusoeProvider(OpenAICompatibleProvider):
    DISPLAY_NAME = "Crusoe"
    CONFIG_TYPE = _OpenAICompatibleConfig
    DEFAULT_API_BASE = "https://api.inference.crusoecloud.com/v1"
