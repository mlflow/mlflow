from mlflow.gateway.config import _OpenAICompatibleConfig
from mlflow.gateway.providers.openai_compatible import OpenAICompatibleProvider


class TheGridProvider(OpenAICompatibleProvider):
    DISPLAY_NAME = "The Grid"
    CONFIG_TYPE = _OpenAICompatibleConfig
    DEFAULT_API_BASE = "https://api.thegrid.ai/v1"
