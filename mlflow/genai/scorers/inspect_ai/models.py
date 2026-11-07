"""Model adapter helpers for Inspect AI tasks.

Provides a lightweight adapter that exposes a simple `complete`/`complete_prompt`
API backed by MLflow's shared `ScorerLLMClient`. This lets Inspect AI tasks
receive a stable model-like object regardless of provider routing.
"""

from __future__ import annotations

from typing import Any, Dict

from mlflow.genai.scorers.llm_backend import ScorerLLMClient


class MlflowInspectAIModel:
    """Thin adapter exposing a minimal model interface for Inspect AI tasks.

    The class delegates to :class:`ScorerLLMClient` so callers can perform
    text completions without worrying about provider routing.
    """

    def __init__(self, model_uri: str, model_kwargs: Dict[str, Any] | None = None):
        self._backend = ScorerLLMClient(model_uri)
        self.model_kwargs = dict(model_kwargs or {})

    @property
    def model_name(self) -> str:
        return self._backend.model_name

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        call_kwargs = dict(self.model_kwargs)
        call_kwargs.update(kwargs)
        return self._backend.complete(messages, **call_kwargs)

    def complete_prompt(self, prompt: str, **kwargs: Any) -> str:
        call_kwargs = dict(self.model_kwargs)
        call_kwargs.update(kwargs)
        return self._backend.complete_prompt(prompt, **call_kwargs)


def create_inspectai_model(model_uri: str, model_kwargs: Dict[str, Any] | None = None) -> MlflowInspectAIModel:
    """Create a model adapter suitable for passing into Inspect AI tasks.

    This mirrors DeepEval's model adapter pattern and centralizes provider
    routing via `ScorerLLMClient`.
    """
    return MlflowInspectAIModel(model_uri, model_kwargs=model_kwargs)