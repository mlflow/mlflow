from mlflow.gateway.config import _OpenAICompatibleConfig
from mlflow.gateway.providers.openai_compatible import OpenAICompatibleProvider


class XAIProvider(OpenAICompatibleProvider):
    DISPLAY_NAME = "xAI"
    CONFIG_TYPE = _OpenAICompatibleConfig
    DEFAULT_API_BASE = "https://api.x.ai/v1"
