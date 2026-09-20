from mlflow.gateway.config import _OpenAICompatibleConfig
from mlflow.gateway.providers.openai_compatible import OpenAICompatibleProvider


class AtlasCloudProvider(OpenAICompatibleProvider):
    DISPLAY_NAME = "Atlas Cloud"
    CONFIG_TYPE = _OpenAICompatibleConfig
    DEFAULT_API_BASE = "https://api.atlascloud.ai/v1"
