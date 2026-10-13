from mlflow.gateway.config import _OpenAICompatibleConfig
from mlflow.gateway.providers.openai_compatible import OpenAICompatibleProvider


class GroqProvider(OpenAICompatibleProvider):
    DISPLAY_NAME = "Groq"
    CONFIG_TYPE = _OpenAICompatibleConfig
    DEFAULT_API_BASE = "https://api.groq.com/openai/v1"
