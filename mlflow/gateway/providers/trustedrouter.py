from mlflow.gateway.config import _OpenAICompatibleConfig
from mlflow.gateway.providers.openai_compatible import OpenAICompatibleProvider


class TrustedRouterProvider(OpenAICompatibleProvider):
    DISPLAY_NAME = "TrustedRouter"
    CONFIG_TYPE = _OpenAICompatibleConfig
    DEFAULT_API_BASE = "https://api.trustedrouter.com/v1"
