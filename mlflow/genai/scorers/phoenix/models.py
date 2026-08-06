from __future__ import annotations

from mlflow.genai.scorers.llm_backend import ScorerLLMClient
from mlflow.genai.scorers.phoenix.utils import _NoOpRateLimiter, check_phoenix_installed


class MlflowPhoenixModel:
    """Phoenix model adapter backed by the shared scorer LLM client.

    Routes through native providers when available, falls back to litellm.
    Implements the duck-typed callable interface Phoenix expects.
    """

    def __init__(self, backend: ScorerLLMClient):
        self._backend = backend
        self._model_name = backend.model_name
        self._verbose = False
        self._rate_limiter = _NoOpRateLimiter()

    def __call__(self, prompt, **kwargs) -> str:
        prompt_str = str(prompt) if not isinstance(prompt, str) else prompt
        return self._backend.complete_prompt(prompt_str)

    def get_model_name(self) -> str:
        return self._backend.model_name


def create_phoenix_model(model_uri: str):
    """Create a Phoenix model adapter from a model URI.

    Routing:
        - Native providers (via ``ScorerLLMClient``) -> MlflowPhoenixModel
        - All other providers -> LiteLLMModel (litellm fallback)
    """
    check_phoenix_installed()

    backend = ScorerLLMClient(model_uri)

    if backend.is_native:
        return MlflowPhoenixModel(backend)

    from phoenix.evals import LiteLLMModel

    return LiteLLMModel(
        model=backend.model_name,
        model_kwargs={"drop_params": True},
    )
