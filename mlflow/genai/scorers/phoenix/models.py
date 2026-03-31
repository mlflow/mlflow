from __future__ import annotations

from mlflow.genai.judges.constants import _DATABRICKS_DEFAULT_JUDGE_MODEL
from mlflow.genai.scorers.llm_backend import MLflowLLMBackend
from mlflow.genai.scorers.phoenix.utils import _NoOpRateLimiter, check_phoenix_installed


class GatewayPhoenixModel:
    """Phoenix model adapter using the shared MLflow LLM backend."""

    def __init__(self, backend: MLflowLLMBackend):
        self._backend = backend
        self._verbose = False
        self._rate_limiter = _NoOpRateLimiter()

    def __call__(self, prompt, **kwargs) -> str:
        prompt_str = str(prompt) if not isinstance(prompt, str) else prompt
        return self._backend.complete(prompt_str)

    def get_model_name(self) -> str:
        return self._backend.model_name


def create_phoenix_model(model_uri: str):
    """Create a Phoenix model adapter from a model URI.

    Routing:
        - ``"databricks"`` → managed judge via backend
        - ``"gateway:/endpoint"`` → gateway endpoint via backend
        - Supported providers (e.g. ``"openai:/gpt-4"``) → native provider via backend
        - Unsupported providers → litellm fallback via backend
    """
    check_phoenix_installed()

    if model_uri == _DATABRICKS_DEFAULT_JUDGE_MODEL:
        return GatewayPhoenixModel(MLflowLLMBackend(model_uri))

    return GatewayPhoenixModel(MLflowLLMBackend(model_uri))
