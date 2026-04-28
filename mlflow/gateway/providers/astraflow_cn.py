from mlflow.gateway.config import _OpenAICompatibleConfig
from mlflow.gateway.providers.openai_compatible import OpenAICompatibleProvider


class AstraflowCNProvider(OpenAICompatibleProvider):
    """Provider for Astraflow CN (UCloud ModelVerse) — China node.

    API base: https://api.modelverse.cn/v1
    API key env var: ASTRAFLOW_CN_API_KEY

    Supports DeepSeek, Claude, GPT-4, and other models available on
    the UCloud ModelVerse platform via an OpenAI-compatible API.
    """

    DISPLAY_NAME = "Astraflow CN"
    CONFIG_TYPE = _OpenAICompatibleConfig
    DEFAULT_API_BASE = "https://api.modelverse.cn/v1"
