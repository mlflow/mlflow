from mlflow.gateway.config import _OpenAICompatibleConfig
from mlflow.gateway.providers.openai_compatible import OpenAICompatibleProvider


class AstraflowProvider(OpenAICompatibleProvider):
    """Provider for Astraflow (UCloud ModelVerse) — Global node (US/CA).

    API base: https://api-us-ca.umodelverse.ai/v1
    API key env var: ASTRAFLOW_API_KEY

    Supports DeepSeek, Claude, GPT-4, and other models available on
    the UCloud ModelVerse platform via an OpenAI-compatible API.
    """

    DISPLAY_NAME = "Astraflow"
    CONFIG_TYPE = _OpenAICompatibleConfig
    DEFAULT_API_BASE = "https://api-us-ca.umodelverse.ai/v1"
