from mlflow.gateway.config import _OpenAICompatibleConfig
from mlflow.gateway.providers.openai_compatible import OpenAICompatibleProvider


class RequestyProvider(OpenAICompatibleProvider):
    DISPLAY_NAME = "Requesty"
    CONFIG_TYPE = _OpenAICompatibleConfig
    DEFAULT_API_BASE = "https://router.requesty.ai/v1"
